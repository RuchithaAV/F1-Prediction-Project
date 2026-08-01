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
    'constructor_recent_podiums',
    'driver_recent_avg_finish',
    'driver_recent_avg_qual',
    'constructor_recent_avg_finish',
    'driver_circuit_avg_finish',
    'constructor_circuit_avg_finish',
    'driver_season_dnf_rate',
    'constructor_season_dnf_rate'
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

# Import additional ML model and optimization tools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Explaining class imbalance context
total_pos = y_test_clean.sum()
total_neg = len(y_test_clean) - total_pos
print("\n=== Class Imbalance Information ===")
print(f"Test set podiums (1s): {total_pos} ({total_pos/len(y_test_clean)*100:.1f}%)")
print(f"Test set non-podiums (0s): {total_neg} ({total_neg/len(y_test_clean)*100:.1f}%)")
print("CRITICAL INSIGHT: Because only ~15% of driver entries finish on the podium, accuracy alone is a misleading metric.")
print("A naive classifier predicting all 0s (no podium) would achieve ~85% accuracy but yield zero predictive value.")
print("We prioritize ROC-AUC, Precision, Recall, and F1-score to gauge actual performance.\n")

# --- MODEL 1: Baseline Logistic Regression ---
print("Training Baseline Logistic Regression...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train_clean)

lr_preds = lr_model.predict(X_test_scaled)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

# --- MODEL 2: Default Random Forest Classifier ---
print("Training Default Random Forest Classifier...")
rf_default = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_default.fit(X_train_clean, y_train_clean)

rf_def_preds = rf_default.predict(X_test_clean)
rf_def_probs = rf_default.predict_proba(X_test_clean)[:, 1]

# --- MODEL 3: Tuned Random Forest Classifier ---
print("Tuning Random Forest via RandomizedSearchCV (Temporal-safe configuration)...")
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
rf_search.fit(X_train_clean, y_train_clean)
best_rf = rf_search.best_estimator_
print(f"Best hyperparameters found: {rf_search.best_params_}")

rf_tuned_preds = best_rf.predict(X_test_clean)
rf_tuned_probs = best_rf.predict_proba(X_test_clean)[:, 1]

# --- EVALUATION COMPARISON ---
models_results = {
    "Logistic Regression (Baseline)": (lr_preds, lr_probs),
    "Random Forest (Default)": (rf_def_preds, rf_def_probs),
    "Random Forest (Tuned)": (rf_tuned_preds, rf_tuned_probs)
}

print("\n================== MODEL COMPARISON REPORT ==================")
for name, (preds, probs) in models_results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy : {accuracy_score(y_test_clean, preds):.4f}")
    print(f"Precision: {precision_score(y_test_clean, preds):.4f}")
    print(f"Recall   : {recall_score(y_test_clean, preds):.4f}")
    print(f"F1-Score : {f1_score(y_test_clean, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test_clean, probs):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_clean, preds))
print("=============================================================")

# Plot & Save Feature Importance for the best model
print("\nGenerating feature importance plot...")
importances = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("F1 Podium Simulator - Feature Importance Breakdown")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
os.makedirs('models', exist_ok=True)
plt.savefig('models/feature_importance.png')
plt.close()
print("Saved feature importance chart to models/feature_importance.png successfully!")

# 5. Save model and preprocessing artifacts
print("\nSaving final model and encoders...")
joblib.dump(best_rf, 'models/f1_podium_model.joblib')
joblib.dump(le_driver, 'models/le_driver.joblib')
joblib.dump(le_constructor, 'models/le_constructor.joblib')
# Save scaler for Logistic Regression in case it's used elsewhere
joblib.dump(scaler, 'models/scaler_podium.joblib')
print("Saved f1_podium_model.joblib, le_driver.joblib, and le_constructor.joblib successfully!")
