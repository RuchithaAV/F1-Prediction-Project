import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# Import utils
from utils import load_raw_data, engineer_podium_features, SafeLabelEncoder

# 1. Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1 dataset")

print("Loading datasets...")
results, races, drivers, constructors, qualifying, _, _ = load_raw_data(DATA_DIR)

print("Engineering features...")
df = engineer_podium_features(results, races, drivers, constructors, qualifying)

# Filter for modern era (post-2000)
df = df[df['year'] >= 2000].copy()

# Encoding categorical columns safely
le_driver = SafeLabelEncoder()
df['driver_encoded'] = le_driver.fit_transform(df['driverRef'])

le_constructor = SafeLabelEncoder()
df['constructor_encoded'] = le_constructor.fit_transform(df['constructorRef'])

# 2. Train/Test Split (Temporal)
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

# 3. Model training
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_clean, y_train_clean)

# 4. Evaluation
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

# 5. Save model and preprocessing artifacts
print("\nSaving model and encoders...")
joblib.dump(model, 'models/f1_podium_model.joblib')
joblib.dump(le_driver, 'models/le_driver.joblib')
joblib.dump(le_constructor, 'models/le_constructor.joblib')
print("Saved f1_podium_model.joblib, le_driver.joblib, and le_constructor.joblib successfully!")
