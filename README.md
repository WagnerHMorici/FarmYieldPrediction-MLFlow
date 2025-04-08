# FarmYieldPrediction-MLFlow

This project leverages MLflow to manage the lifecycle of a machine learning model designed to predict agricultural crop yields (in tons per hectare) based on various environmental and management factors.

## Project Overview
Agricultural productivity is influenced by multiple factors, including rainfall, soil quality, farm size, sunlight hours, and fertilizer usage. This project develops a regression model to predict crop yields using these variables as inputs. MLflow is utilized to track experiments, log parameters, metrics, and models, and facilitate deployment and monitoring in production environments.

## Dataset

The dataset used in this project is the Crop Yield of a Farm from Kaggle. 
https://www.kaggle.com/datasets/govindaramsriram/crop-yield-of-a-farm


### Environment Setup

git clone https://github.com/your-username/FarmYieldPrediction-MLFlow.git
cd FarmYieldPrediction-MLFlow

poetry install
poetry shell

### Usage 

mlflow server --host 127.0.0.1 --port 5000

python main_train.py

python predict.py