# water_level_gap_filling

# Overview

This project uses a Multi-Layer Perceptron (MLP) model to predict water levels values for a target tide gauge station, 
using input data from a nearby paired tide gauge station. 
The water level predictions are intended to support QA/QC of historical water level data. 
Data for this project is downloaded directly from the 
Texas *Lighthouse* database. Project is adapted by Nevena Cail from `Florida_WL_MLP_model_training_NEW.ipynb` by Beto 
Estrada (Texas A&M University-Corpus Christi).

# Features

* **Data Preprocessing**: Clean and prepare raw water level data.

* **Input Generation**: Create input features suitable for MLP models.

* **Model Training**: Train an MLP model to predict missing water level values.

* **Water Level Prediction**: Apply the trained model to predict water level values.

# Repository Structure

* `preprocess_data_files.py`: Script for preprocessing raw data files.
* `generate_mlp_inputs.py`: Generates input features for the MLP model.
* `gap_filling_mlp.py`: Applies the trained MLP model to predict water levels.
* `Florida_WL_MLP_model_training_NEW.ipynb`: Jupyter notebook demonstrating original application of the model.
* `helpers.py`: Contains helper functions used across scripts.
* `config_preprocess_data_files.json`: Configuration for data preprocessing.
* `config_mlp_input_generation.json`: Configuration for input generation.
* `config_mlp.json`: Configuration for MLP model training.