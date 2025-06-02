"""
Adapted from Florida_WL_MLP_model_training_NEW.ipynb.
This program builds and trains an MLP for water level predictions.
Input data to the MLP is expected to be generated from generate_mlp_inputs.py, pulled from config_mlp.json.
"""

# Imports.
# Plotting
import matplotlib.pyplot as plt

# Data manipulation
import pandas as pd

# Machine Learning
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential, load_model
from sklearn.metrics import (mean_squared_error, root_mean_squared_error, mean_absolute_error, median_absolute_error,
                             r2_score, mean_absolute_percentage_error)

# Misc
import helpers
import json
from datetime import datetime


''' *********************************************** FUNCTIONS *********************************************** '''
def calculate_central_frequency_percentage(testing_label_array, predictions, cm):
    """
    Find the percentage of predictions with a central frequency (CF) of less than
    or equal to a given number of centimeters (cm)

    Args:
        testing_label_array (array): Testing labels

        predictions (array): Model predictions

        cm (int): Number of centimeters

    Returns:
        (float): central frequency (CF) percentage
    """
    less_than_cm_counter = 0

    # Convert cm to m
    cm_to_m = cm / 100

    for index, prediction in enumerate(predictions):
        if abs(testing_label_array[index] - prediction) <= cm_to_m:
            less_than_cm_counter += 1

    cf_percentage = (less_than_cm_counter / len(predictions)) * 100

    return cf_percentage


def evaluate_model(model, testing_input_array, testing_label_array):
    """
    Calculates loss, makes predictions, and calculates Central Frequency (CF),
    Mean Squared Error (MSE), Root Mean Squared Error(RMSE), Mean Absolute Error (MAE),
    Median Absolute Error, and R-squared (R2)

    Args:
    model (tf.keras.model): The trained model

    testing_input_array (array): Testing inputs

    testing_label_array (array): Testing labels
    """
    test_loss = model.evaluate(testing_input_array, testing_label_array, batch_size=len(testing_input_array))

    predictions = model.predict(testing_input_array, batch_size=len(testing_input_array))

    # Calculate evaluation metrics
    cf_15cm_percentage = calculate_central_frequency_percentage(testing_label_array, predictions, 15)
    cf_5cm_percentage = calculate_central_frequency_percentage(testing_label_array, predictions, 5)
    cf_2cm_percentage = calculate_central_frequency_percentage(testing_label_array, predictions, 2)
    cf_1cm_percentage = calculate_central_frequency_percentage(testing_label_array, predictions, 1)
    mse = mean_squared_error(testing_label_array, predictions)
    rmse = root_mean_squared_error(testing_label_array, predictions)
    mae = mean_absolute_error(testing_label_array, predictions)
    medae = median_absolute_error(testing_label_array, predictions)
    r2 = r2_score(testing_label_array, predictions)

    results_dict = {
        'Loss': test_loss,
        'Central Frequency Percentage 15cm': cf_15cm_percentage,
        'Central Frequency Percentage 5cm': cf_5cm_percentage,
        'Central Frequency Percentage 2cm': cf_2cm_percentage,
        'Central Frequency Percentage 1cm': cf_1cm_percentage,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae,
        'Median Absolute Error': medae,
        'R-squared': r2
    }

    return results_dict


def write_stats(data_arrays, filename, results_dict):
    with open(filename, 'w') as file:
        file.write(f"Training data: {data_arrays[0]}\n"
                   f"Validation data: {data_arrays[1]}\n\n")
        json.dump(results_dict, file, indent=4)
    file.close()


''' ************************************************ GET DATA *********************************************** '''
# Save off current time for file naming.
current_timestamp = datetime.now().strftime('%m%d%Y_%H%M%S')

# Get configs.
config = helpers.load_configs('config_mlp.json')

data_paths = config['data_arrays']

# Get training, validation, testing datasets.
training_data = pd.read_csv(data_paths[0], index_col=0)
validation_data = pd.read_csv(data_paths[1], index_col=0)

# Get target arrays, then drop from dataframe.
training_targets = training_data.iloc[:, -1]
validation_targets = validation_data.iloc[:, -1]
training_data_inputs = training_data.drop(columns=[training_data.columns[-1]])
validation_data_inputs = validation_data.drop(columns=[validation_data.columns[-1]])

# The remaining data are the inputs.
training_inputs = training_data_inputs.values
validation_inputs = validation_data_inputs.values

# print(training_inputs.shape, validation_inputs.shape)
# print(training_targets.shape, validation_targets.shape)


''' *********************************************** BUILD MLP *********************************************** '''
# Skip building and training if model exists ready to run/test.
if config['existing_model']:
    model = load_model(config['existing_model'])

# Build and train new model if an existing model is not being loaded to run.
else:
    config_model = config['mlp_parameters']

    # Create a simple sequential model
    model = Sequential()

    # Hidden Layer with 20 hidden neurons and relu activation function
    model.add(Dense(units=config_model['hidden_neurons'], activation='relu'))

    # Dropout Layer with a small dropout rate (good for when you have a large amount of samples).
    model.add(Dropout(rate=config_model['dropout_rate']))

    # Output Layer outputs one value, 6-minute water level.
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model using mean squared error (MSE) loss function and adam optimizer.
    model.compile(loss=config_model['loss'], optimizer='adam')


    ''' *********************************************** TRAIN MODEL *********************************************** '''
    # Define number of epochs.
    epochs = config_model['epochs']

    # Define training and validation batch sizes using batch gradient descent (BGD).
    # BGD utilizes the entire dataset during each epoch to compute the gradients.
    batch_size = len(training_inputs)
    validation_batch_size = len(validation_inputs)

    # Early stopping is meant to prevent overfitting and save training time.
    # Define early stopping callback with a 'patience' of 50 epochs that monitors validation loss.
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=50,  # If val_loss doesn't improve for 50 epochs, stop training.
                                   restore_best_weights=True,  # After stopping, revert to the model state with the
                                   # lowest val_loss.
                                   verbose=1)  # Print a message when early stopping is triggered.

    # Define model checkpoint callback that saves the model weights at the epoch that had the best validation loss.
    # In this case, it looks for the epoch with the minimum validation loss and writes the weights to the file
    # name specififed.
    model_file_name = config['model_file_name']
    checkpoint = ModelCheckpoint(model_file_name,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 verbose=1)

    # Add early stopping and model checkpoint to list of callbacks.
    callbacks = [early_stopping, checkpoint]

    # Fit the model using the training and validation sets we defined earlier.
    model_history = model.fit(training_inputs, training_targets,
                              validation_data=(validation_inputs, validation_targets),
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_batch_size=validation_batch_size,
                              callbacks=callbacks)
# End build and train model.

''' *********************************************** TEST & LOG MODEL *********************************************** '''
# Calculate and print performance metrics. Write to txt file.
results = evaluate_model(model, validation_inputs, validation_targets)
write_stats(config['data_arrays'], config['stats_file_name'], results)
predictions = model.predict(validation_inputs, batch_size=len(validation_inputs))

# Save csv of predicted vs observed.
observed_vs_predicted_df = pd.DataFrame()
observed_vs_predicted_df['timestamp'] = validation_targets.index
observed_vs_predicted_df['observed'] = validation_targets.values
observed_vs_predicted_df['predicted'] = predictions
observed_vs_predicted_df.to_csv(config['predictions_filename'], index=False)

# Log configs.
if config['logging_on']:
    with open(f"{current_timestamp}_log.txt", "w") as f:
        json.dump(config, f, indent=4)




