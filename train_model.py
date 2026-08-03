import pandas as pd #reads and loads .csv file
import numpy as np #used for numerical operations
import joblib #saves and loads the trained model
from sklearn.ensemble import RandomForestRegressor #model
from sklearn.model_selection import train_test_split #splits data into train and test sets
from sklearn.preprocessing import OneHotEncoder #encodes categorical columns as numbers
from sklearn.pipeline import Pipeline #connects preprocessing and model into one object
from sklearn.compose import ColumnTransformer #different preprocessing for different columns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load data
data = pd.read_csv("data/used_cars.csv")

#clean data
data["milage"] = (data["milage"].str.replace(",","").str.replace(" mi.","").astype(int))
data["price"] = (data["price"].str.replace("$","", regex=False).str.replace(",","").astype(int))

#drop rows missing a fuel type
data = data.dropna(subset=["fuel_type"])
data = data.drop_duplicates()

data = data[(data["price"] >= 2000) & (data["price"] <= 200000)]
data = data[data["milage"] <= 300000]

#organizing data
x = data[["brand","model","model_year","milage","fuel_type","transmission"]]
y = np.log1p(data["price"])

#split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#categorical columns
categorical = ["brand", "model", "fuel_type", "transmission"]
preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
            remainder="passthrough")

#model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=700,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ))
])

#train
model.fit(x_train, y_train)
log_preds = model.predict(x_test)
preds = np.expm1(log_preds)
actual = np.expm1(y_test)

print("MAE:", mean_absolute_error(actual, preds))
print("MSE:", np.sqrt(mean_squared_error(actual, preds)))
print("R2:", r2_score(actual, preds))
print("Sample predictions:", preds[:5])
print("Actual values:", actual.values[:5])

#saving model
joblib.dump(model, "model.pkl")

