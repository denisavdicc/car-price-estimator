# Car Price Estimator

#### Description:

Car Price Estimator is a Flask web app that estimates what a used car is worth. You enter the car's brand, model, model year, mileage, fuel type, and transmission, and it returns a predicted price from a machine learning model trained on a dataset of real used car listings. The app also has full user accounts (registration, login, logout, password changes) and a history page where every prediction is saved to the account that made it, so you can review or delete past estimates later.

I built this project to combine two things: a real web application with user accounts and a database behind it, and a machine learning model performing a practical task instead of simply producing results in a terminal. Used car price prediction was a good fit, as it's a regression problem involving both numeric features (model year and mileage) and categorical features (brand, model, fuel type, and transmission). The dataset also required significant cleaning and preprocessing before it could be used for training.

## Setup

### macOS

```bash
# 1. Clone the repository
git clone https://github.com/denisavdicc/car-price-estimator.git
cd car-price-estimator

# 2. Create a virtual environment
python3.12 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model (generates model.pkl, takes a minute or two)
python train_model.py

# 6. Run the app
python app.py
```

If Python 3.12 isn't installed:
```bash
brew install python@3.12
```

If `pip install` fails with an `externally-managed-environment` error, it means the virtual environment wasn't activated first. Make sure you see `(.venv)` at the start of your terminal prompt before running step 4. If not, repeat step 3.

### Windows

```bash
# 1. Clone the repository
git clone https://github.com/denisavdicc/car-price-estimator.git
cd car-price-estimator

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model (generates model.pkl, takes a minute or two)
python train_model.py

# 6. Run the app
python app.py
```

If `python` isn't recognized, try `py` instead (`py -m venv .venv`, then `py train_model.py`, `py app.py`).

### After running the app

The database (`cars.db`) and its tables are created automatically the first time the app runs, so no manual database setup is needed. Once it's running, visit `http://127.0.0.1:5000` in your browser, register an account, and start predicting.

## The Dataset

The model is trained on `used_cars.csv`, 4,009 listings covering 57 brands and 1898 distinct models, with model years from 1974 to 2024. The raw file has 12 columns, including engine description, exterior/interior color, accident history, and clean title status, but I only used six as features: brand, model, model_year, milage, fuel_type, and transmission.

Mileage and price both arrive as strings (`"51,000 mi."`, `"$10,300"`), so both get their commas, units, and dollar signs stripped and cast to integers. Fuel type is missing on about 170 rows, and a few remaining rows have garbage values (`"–"`, `"not supported"`) instead of a real fuel type, so rows with a missing fuel type get dropped before training. The data skews heavily toward gasoline vehicles, with smaller groups of hybrid, plug-in hybrid, diesel, and E85 flex-fuel mixed in.

## Files

**app.py** defines every route. It loads the trained model with `joblib` at startup, and uses the `cs50` library's `SQL` wrapper to read/write a SQLite database (`cars.db`) storing users and prediction history. Sessions are handled with `Flask-Session`, and a `login_required` decorator (using `functools.wraps`) redirects anonymous visitors away from routes that require login.

- `/` renders `index.html`, the homepage, which links to the prediction tool or to login/register depending on whether `session["user_id"]` is set.
- `/register` and `/login` handle account creation and sign-in. Passwords are hashed with `werkzeug.security.generate_password_hash` and checked with `check_password_hash`; registration also checks that the two password fields match and the username isn't taken.
- `/logout` clears the session and redirects home.
- `/change_password` updates the password after confirming the current one.
- `/predict` is the core route. GET shows the empty form. POST reads the six fields, checks none are blank, builds a one-row pandas DataFrame with matching column names, and runs `model.predict()` on it. The result is shown to the user and inserted into the `history` table under their `user_id`.
- `/history` shows all of the current user's predictions, most recent first.
- `/clear_history` and `/delete_prediction` remove all or one saved prediction, both filtered by `user_id` so a user can't delete someone else's row.

**train_model.py** is run once, offline, to produce `model.pkl`. The Flask app only loads its output. After cleaning the CSV, it splits `brand`, `model`, `model_year`, `mileage`, `fuel_type`, and `transmission` into the feature matrix, with `price` as the target. The four categorical columns are one-hot encoded via a `ColumnTransformer` with `handle_unknown="ignore"`, so an unseen brand or model just encodes as all zeros instead of crashing the app. The encoder and a `RandomForestRegressor` are chained in one scikit-learn `Pipeline`, fit on an 80/20 train-test split, and saved with `joblib.dump`.

**templates/** - `index.html`, `login.html`, `register.html`, `predict.html`, `history.html`, and `change_password.html` all extend a shared `layout.html` for the navbar and page structure. `predict.html` and `history.html` format prices with commas and two decimals. `history.html` also splits the stored timestamp into date and time. `login.html` and `register.html` include a small JavaScript snippet toggling a "Show Password" checkbox.

**styles.css** - a few small overrides on Bootstrap defaults (accent color, rounded cards).

## Design Decisions

I used a random forest instead of linear regression because car prices don't move linearly with these features. Depreciation curves aren't straight lines, and mileage matters differently depending on the car's age. A tree-based ensemble picks up on that without hand-engineered interaction terms. I combined the encoder and regressor into one Pipeline object so training time and prediction time preprocessing can't fall out of sync. I stored prediction history in the database rather than the session since sessions don't persist across devices, and storing rows with their own IDs is what makes the individual delete/clear-history features possible.
I don't commit `cars.db` or `model.pkl` to this repository. The database would expose real account data if included, and the trained model is a large binary file that's easy to regenerate locally from the included training script and dataset.

## Data Source

The dataset (`used_cars.csv`) is the [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) by Taeef Najib on Kaggle, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It contains 4,009 vehicle listings extracted from cars.com. Values in the `milage` and `price` columns were reformatted from strings (e.g. `"51,000 mi."`, `"$10,300"`) into plain integers, and rows with a missing `fuel_type` were removed before training. No other changes were made to the original data.
