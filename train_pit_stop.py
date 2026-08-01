import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Import utils
from utils import load_raw_data, SafeLabelEncoder

# 1. Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1 dataset")

print("Loading datasets...")
results, races, drivers, constructors, _, _, pit_stops = load_raw_data(DATA_DIR)

if pit_stops is None:
    raise FileNotFoundError("pit_stops.csv is required but not found in data directory.")

print("Preprocessing pit stop data...")
# Merge pit_stops with races to get year and race name
model_df = pit_stops.merge(races[['raceId', 'year', 'name']], on='raceId', how='inner')

# Merge with results to get constructorId and grid
model_df = model_df.merge(results[['raceId', 'driverId', 'constructorId', 'grid', 'laps']], on=['raceId', 'driverId'], how='inner')

# Merge with drivers to get driver name (driver_name = forename + " " + surname)
drivers = drivers.copy()
drivers['driver_name'] = drivers['forename'] + ' ' + drivers['surname']
model_df = model_df.merge(drivers[['driverId', 'driver_name']], on='driverId', how='inner')

# Merge with constructors to get constructorRef
model_df = model_df.merge(constructors[['constructorId', 'constructorRef']], on='constructorId', how='inner')

# Compute total laps per race
results = results.copy()
results['laps'] = pd.to_numeric(results['laps'], errors='coerce')
race_laps = results.groupby('raceId')['laps'].max().reset_index()
race_laps.columns = ['raceId', 'total_laps']

# Merge total laps back to model_df
model_df = model_df.merge(race_laps, on='raceId', how='left')

# Create normalized pit lap percentage target
model_df['pit_lap_pct'] = (model_df['lap'] / model_df['total_laps']) * 100

# Drop any row with missing features or targets
model_df = model_df.dropna(subset=['pit_lap_pct', 'grid', 'stop', 'year', 'total_laps'])

print(f"Total samples available: {model_df.shape[0]}")

# Encode categorical columns safely using SafeLabelEncoder
le_driver = SafeLabelEncoder()
model_df['driver_encoded'] = le_driver.fit_transform(model_df['driver_name'])

le_constructor = SafeLabelEncoder()
model_df['constructor_encoded'] = le_constructor.fit_transform(model_df['constructorRef'])

le_race = SafeLabelEncoder()
model_df['race_encoded'] = le_race.fit_transform(model_df['name'])

# Define features and target
features = ['driver_encoded', 'constructor_encoded', 'race_encoded', 'grid', 'stop', 'year', 'total_laps']
target = 'pit_lap_pct'

# Temporal split (Train: <= 2022, Test: >= 2023)
print("Splitting data chronologically...")
train_df = model_df[model_df['year'] <= 2022]
test_df = model_df[model_df['year'] >= 2023]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Train size: {X_train.shape[0]} rows (2018-2022)")
print(f"Test size: {X_test.shape[0]} rows (2023+)")

# Model training
print("Training Default Gradient Boosting Regressor...")
gb_default = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_default.fit(X_train, y_train)

# Default Evaluation
default_preds = gb_default.predict(X_test)
default_mae = mean_absolute_error(y_test, default_preds)
default_r2 = r2_score(y_test, default_preds)

# Hyperparameter Tuning via RandomizedSearchCV
print("Tuning Gradient Boosting Regressor via RandomizedSearchCV...")
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

gb_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_
print(f"Best hyperparameters found: {gb_search.best_params_}")

# Tuned Evaluation
tuned_preds = best_gb.predict(X_test)
tuned_mae = mean_absolute_error(y_test, tuned_preds)
tuned_r2 = r2_score(y_test, tuned_preds)

print("\n=== Model Performance Comparison ===")
print(f"Default Model -> MAE: {default_mae:.2f}%, R2 Score: {default_r2:.4f}")
print(f"Tuned Model   -> MAE: {tuned_mae:.2f}%, R2 Score: {tuned_r2:.4f}")
print("====================================")

# Save model and encoders
print("\nSaving model and encoders...")
joblib.dump(best_gb, 'models/f1_pit_stop_model.joblib')
joblib.dump(le_driver, 'models/le_driver_pit.joblib')
joblib.dump(le_constructor, 'models/le_constructor_pit.joblib')
joblib.dump(le_race, 'models/le_race_pit.joblib')
print("Saved f1_pit_stop_model.joblib, le_driver_pit.joblib, le_constructor_pit.joblib, and le_race_pit.joblib successfully!")
