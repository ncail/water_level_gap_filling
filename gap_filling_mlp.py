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
# from sklearn.metrics import (mean_squared_error, root_mean_squared_error, mean_absolute_error, median_absolute_error,
#                              r2_score, mean_absolute_percentage_error)

import helpers

''' *********************************************** FUNCTIONS *********************************************** '''
def plot_model_predictions(testing_labels, predictions, title, x_label, y_label,
                           plot_file_name, legend_location='best'):
    """Plots model predictions and compares to testing labels to evaluate the model performance

    Args:
        testing_labels (list): list of testing labels

        predictions (list): list of model predictions

        title (string): plot title

        x_label (string): x-axis label

        y_label (string): y-axis label

        plot_file_name (string): file name for plot. Must have appropriate image extension (eg. 'png', 'pdf', 'svg', ...)

        legend_location (string): 'best' (Axes only), 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                                  'center left', 'center right', 'lower center', 'upper center', 'center'. Defaults to 'best'
    """
    fig, ax = plt.subplots(1, figsize=(25, 10))

    # Plot the observations and model predictions on the same plot
    plt.plot(testing_labels, label='Observed')
    plt.plot(predictions, label='Predicted')

    plt.title(title, fontsize=30)

    ax.set_xlabel(x_label, fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize = 20, loc = legend_location)

    plt.savefig(plot_file_name, bbox_inches='tight')

    plt.show()

    plt.close()


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


''' *********************************************** TEST MODEL *********************************************** '''
# Plot observed vs predicted time series.
predictions = model.predict(validation_inputs, batch_size=len(validation_inputs))
title = 'Water Level MLP Model Observed vs Predicted'
x_label = 'Index'
y_label = 'Elevation in meters (NAVD88)'
plot_file_name = config['plot_file_name']

plot_model_predictions(validation_targets, predictions, title, x_label, y_label, plot_file_name)






