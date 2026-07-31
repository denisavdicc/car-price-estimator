import pandas as pd #reads and loads .csv file
import joblib #saves and loads the trained model
from sklearn.ensemble import RandomForestRegressor #model
from sklearn.model_selection import train_test_split #splits data into train/test sets
from sklearn.preprocessing import OneHotEncoder #encodes categorical columns as numbers
from sklearn.pipeline import Pipeline #connects preprocessing and model into one object
from sklearn.compose import ColumnTransformer #different preprocessing for different columns
from sklearn.metrics import mean_absolute_error

#load data
data = pd.read_csv("data/used_cars.csv")

#clean data
data["milage"] = (data["milage"].str.replace(",","").str.replace(" mi.","").astype(int))
data["price"] = (data["price"].str.replace("$","", regex=False).str.replace(",","").astype(int))

#drop rows missing a fuel type
data = data.dropna(subset=["fuel_type"])

#organizing data (x = info about a car, y = price that's supposed to be predicted)
x = data[["brand","model","model_year","milage","fuel_type","transmission"]]
y = data["price"]

#split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#categorical columns
categorical = ["brand", "model", "fuel_type", "transmission"]
preprocessor = ColumnTransformer(
            transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             categorical)],
             remainder="passthrough")

#model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ))
])

#train
model.fit(x_train, y_train)

#test
preds = model.predict(x_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("Sample predictions:", preds[:5])
print("Actual values:", y_test.values[:5])

#saving model
joblib.dump(model, "model.pkl")
