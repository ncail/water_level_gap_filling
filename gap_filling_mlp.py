"""
Adapted from Florida_WL_MLP_model_training_NEW.ipynb by Beto Estrada (TAMUCC) by Nevena Cail.
This program builds and trains an MLP for water level predictions.
Input data to the MLP is expected to be generated from generate_mlp_inputs.py
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

import helpers

''' *********************************************** FUNCTIONS *********************************************** '''
def plot_model_predictions(testing_labels, predictions, title, x_label, y_label,
                           plot_file_name, legend_location='best'):
    """
    Plots model predictions and compares to testing labels to evaluate the model performance

    Args:
        testing_labels (list): list of testing labels

        predictions (list): list of model predictions

        title (string): plot title

        x_label (string): x-axis label

        y_label (string): y-axis label

        plot_file_name (string): file name for plot. Must have appropriate image extension (eg. 'png', 'pdf', 'svg',
        ...)

        legend_location (string): 'best' (Axes only), 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                                  'center left', 'center right', 'lower center', 'upper center', 'center'. Defaults
                                  to 'best'
    """
    fig, ax = plt.subplots(1, figsize=(25, 10))

    # Plot the observations and model predictions on the same plot
    plt.plot(testing_labels, label='Observed')
    plt.plot(predictions, label='Predicted')

    plt.title(title, fontsize=30)

    ax.set_xlabel(x_label, fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)

    plt.xticks(fontsize=20, ha='right', rotation=45)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=20, loc=legend_location)

    plt.savefig(plot_file_name, bbox_inches='tight')

    # plt.show()

    plt.close()


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
    print("Calculating Loss:")
    test_loss = model.evaluate(testing_input_array, testing_label_array, batch_size=len(testing_input_array))

    print("Loss:", test_loss)


    print("\nGenerating output predictions with model:")
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

    print("\nCentral Frequency Percentage 15cm:", cf_15cm_percentage)
    print("\nCentral Frequency Percentage 5cm:", cf_5cm_percentage)
    print("\nCentral Frequency Percentage 2cm:", cf_2cm_percentage)
    print("\nCentral Frequency Percentage 1cm:", cf_1cm_percentage)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("Median Absolute Error:", medae)
    print("R-squared:", r2)

    return cf_15cm_percentage, cf_5cm_percentage, cf_2cm_percentage, cf_1cm_percentage, mse, rmse, mae, medae, r2


def write_stats(data_arrays, filename, cf_15cm_percentage, cf_5cm_percentage, cf_2cm_percentage, cf_1cm_percentage,
                mse, rmse, mae, medae, r2):
    with open(filename, 'w') as file:
        file.write(f"Training data: {data_arrays[0]}\n"
                   f"Validation data: {data_arrays[1]}\n\n"
                   f"Central Frequency Percentage 15cm: {cf_15cm_percentage}\n"
                   f"Central Frequency Percentage 5cm: {cf_5cm_percentage}\n"
                   f"Central Frequency Percentage 2cm: {cf_2cm_percentage}\n"
                   f"Central Frequency Percentage 1cm: {cf_1cm_percentage}\n"
                   f"Mean Squared Error: {mse}\n"
                   f"Root Mean Squared Error: {rmse}\n"
                   f"Mean Absolute Error: {mae}\n"
                   f"Median Absolute Error: {medae}\n"
                   f"R-squared: {r2}")
    file.close()


''' ************************************************ GET DATA *********************************************** '''
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
if config['existing_model']:
    model = load_model(config['existing_model'])

if not config['existing_model']:
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
# End create and train model.

''' *********************************************** TEST MODEL *********************************************** '''
# Calculate and print performance metrics.
results = evaluate_model(model, validation_inputs, validation_targets)
write_stats(config['data_arrays'], config['stats_file_name'], results[0], results[1], results[2], results[3],
            results[4], results[5], results[6], results[7], results[8])

# Plot observed vs predicted time series.
if config['plot_file_name']:
    predictions = model.predict(validation_inputs, batch_size=len(validation_inputs))
    title = 'Water Level MLP Model Observed vs Predicted'
    x_label = 'Index'
    y_label = 'Elevation in meters (Stn. Datum)'
    plot_file_name = config['plot_file_name']

    plot_model_predictions(validation_targets, predictions, title, x_label, y_label,
                           plot_file_name)






