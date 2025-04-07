import pandas as pd     

import mlflow

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics


df = pd.read_csv('dataset\crop_yield_data.csv')

X = df[['rainfall_mm', 'soil_quality_index', 'farm_size_hectares',
       'sunlight_hours', 'fertilizer_kg']]
y = df['crop_yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

print("Taxa de resposta train:", y_train.mean())
print("Taxa de resposta test:", y_test.mean())


with mlflow.start_run():

    mlflow.sklearn.autolog()

    randon_forest_model = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)

    randon_forest_model.fit(X_train, y_train)

    y_train_predict = randon_forest_model.predict(X_train)
    y_test_predict = randon_forest_model.predict(X_test)

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    acc_test = metrics.accuracy_score(y_test, y_test_predict)

    mlflow.log_metrics({"acc_train": acc_train, "acc_test": acc_test})


