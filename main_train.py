import pandas as pd     

import mlflow

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=734718012324761974)

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

    mse_train = mean_squared_error(y_train, y_train_predict)
    mae_train = mean_absolute_error(y_train, y_train_predict)
    r2_train = r2_score(y_train, y_train_predict)

    mse_test = mean_squared_error(y_test, y_test_predict)
    mae_test = mean_absolute_error(y_test, y_test_predict)
    r2_test = r2_score(y_test, y_test_predict)

    mlflow.log_metrics({
        "mse_train": mse_train,
        "mae_train": mae_train,
        "r2_train": r2_train,
        "mse_test": mse_test,
        "mae_test": mae_test,
        "r2_test": r2_test
    })

    print("Treino:")
    print("MSE:", mse_train)
    print("MAE:", mae_train)
    print("R²:", r2_train)

    print("\nTeste:")
    print("MSE:", mse_test)
    print("MAE:", mae_test)
    print("R²:", r2_test)

