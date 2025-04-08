import mlflow
import mlflow.client
import pandas as pd

mlflow.set_tracking_uri('http://127.0.0.1:5000/')

client = mlflow.client.MlflowClient()

version = max([int(i.version) for i in client.get_latest_versions('crop-yield-model')])

model = mlflow.sklearn.load_model(f"models:/crop-yield-model/{version}")

df = pd.read_csv("dataset/test_data.csv", sep=',')

X = df[['rainfall_mm', 'soil_quality_index', 'farm_size_hectares',
       'sunlight_hours', 'fertilizer_kg']]

prob = model.predict(X)

print(prob)