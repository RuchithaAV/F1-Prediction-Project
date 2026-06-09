import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1 dataset")

def time_to_seconds(t_str):
    if pd.isna(t_str) or not isinstance(t_str, str) or t_str.strip() == '' or t_str.strip() == '\\N':
        return np.nan
    try:
        parts = t_str.strip().split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except:
        return np.nan
    return np.nan

print("Loading datasets...")
results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
drivers = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
constructors = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
qualifying = pd.read_csv(os.path.join(DATA_DIR, "qualifying.csv"))

# Replace F1 Ergast SQL null placeholder '\N' with NaN/None
for df_temp in [results, races, drivers, constructors, qualifying]:
    df_temp.replace(r'\N', np.nan, inplace=True)
    df_temp.replace('\\N', np.nan, inplace=True)

# Process qualifying times
for col in ['q1', 'q2', 'q3']:
    qualifying[col + '_sec'] = qualifying[col].apply(time_to_seconds)

qualifying['best_qual_time'] = qualifying[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)
pole_times = qualifying.groupby('raceId')['best_qual_time'].transform('min')
qualifying['qual_gap_to_pole'] = qualifying['best_qual_time'] - pole_times

# 2. Convert column data types
results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')
results['points'] = pd.to_numeric(results['points'], errors='coerce')

races['year'] = pd.to_numeric(races['year'], errors='coerce')
races['round'] = pd.to_numeric(races['round'], errors='coerce')
races['date'] = pd.to_datetime(races['date'], errors='coerce')

drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')

# 3. Merge data
print("Merging datasets...")
# Merge results with race info
df = results.merge(races[['raceId', 'year', 'round', 'circuitId', 'date']], on='raceId', how='inner')
# Merge with driver info
df = df.merge(drivers[['driverId', 'driverRef', 'dob', 'nationality']], on='driverId', how='inner')
# Merge with constructor info
df = df.merge(constructors[['constructorId', 'constructorRef']], on='constructorId', how='inner')
# Merge with qualifying info
df = df.merge(qualifying[['raceId', 'driverId', 'position', 'qual_gap_to_pole']].rename(columns={'position': 'qual_position'}), on=['raceId', 'driverId'], how='left')
df['qual_position'] = df['qual_position'].fillna(df['grid'])
df['qual_gap_to_pole'] = df['qual_gap_to_pole'].fillna(5.0)

# Filter for modern era (post-2000)
df = df[df['year'] >= 2000].copy()

# Sort chronologically to engineer prior history features
df.sort_values(by=['date', 'round', 'positionOrder'], inplace=True)

# 4. Target Definition
df['podium_finish'] = (df['positionOrder'] <= 3).astype(int)

# 5. Feature Engineering
print("Engineering features...")

# Feature: Driver age at the time of the race
df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

# Feature: Cumulative prior points and wins in the current season (leakage-free)
# We compute cumulative points/wins group by (year, driverId/constructorId), then shift by 1 to exclude current race
df['win'] = (df['positionOrder'] == 1).astype(int)

# Driver prior stats in the season
df['driver_prior_pts_season'] = df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']
df['driver_prior_wins_season'] = df.groupby(['year', 'driverId'])['win'].cumsum() - df['win']

# Constructor prior stats in the season
df['constructor_prior_pts_season'] = df.groupby(['year', 'constructorId'])['points'].cumsum() - df['points']
df['constructor_prior_wins_season'] = df.groupby(['year', 'constructorId'])['win'].cumsum() - df['win']

# Feature: Driver and Constructor recent form (rolling podium count in previous 3 races overall)
# Sort by driver and date to get absolute rolling shift
driver_history = df.sort_values('date').groupby('driverId')
df['driver_recent_podiums'] = driver_history['podium_finish'].shift(1).rolling(3, min_periods=1).sum().fillna(0)

constructor_history = df.sort_values('date').groupby('constructorId')
df['constructor_recent_podiums'] = constructor_history['podium_finish'].shift(1).rolling(3, min_periods=1).sum().fillna(0)

# Categorical label encoding
le_driver = LabelEncoder()
df['driver_encoded'] = le_driver.fit_transform(df['driverRef'])

le_constructor = LabelEncoder()
df['constructor_encoded'] = le_constructor.fit_transform(df['constructorRef'])

# 6. Train/Test Split (Temporal)
print("Splitting data chronologically...")
train_df = df[df['year'] < 2022].copy()
test_df = df[df['year'] >= 2022].copy()

features = [
    'grid',
    'qual_position',
    'qual_gap_to_pole',
    'driver_encoded',
    'constructor_encoded',
    'circuitId',
    'driver_age',
    'driver_prior_pts_season',
    'driver_prior_wins_season',
    'constructor_prior_pts_season',
    'constructor_prior_wins_season',
    'driver_recent_podiums',
    'constructor_recent_podiums'
]
target = 'podium_finish'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# Drop any row with nan in features
X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index]
X_test_clean = X_test.dropna()
y_test_clean = y_test.loc[X_test_clean.index]

print(f"Train size: {X_train_clean.shape[0]} rows (2000-2021)")
print(f"Test size: {X_test_clean.shape[0]} rows (2022+)")

# 7. Model training
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_clean, y_train_clean)

# 8. Evaluation
preds = model.predict(X_test_clean)
probs = model.predict_proba(X_test_clean)[:, 1]

print("\n=== Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test_clean, preds):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test_clean, probs):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_clean, preds))

# Feature importances
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== Feature Importances ===")
for col, val in importances.items():
    print(f"{col:<30}: {val:.4f}")

# 9. Save model and preprocessing artifacts
print("\nSaving model and encoders...")
joblib.dump(model, 'f1_podium_model.joblib')
joblib.dump(le_driver, 'le_driver.joblib')
joblib.dump(le_constructor, 'le_constructor.joblib')
print("Saved f1_podium_model.joblib, le_driver.joblib, and le_constructor.joblib successfully!")

