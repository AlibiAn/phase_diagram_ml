import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('features_with_liquidus_curves.csv')

target_cols = [f'liquidus_temp_at_{comp}pct' for comp in [10, 20, 30, 40,
                                                         50, 60, 70, 80,
                                                         90]]

y = df[target_cols].copy()

# Filter out incomplete rows
complete_mask = y.notna().all(axis = 1)
y = y[complete_mask]
df = df[complete_mask]

drop_columns = [
    # Metadata
    'entry_id',
    'elements_str',
    'composition',
    # Our targets + features that may leak info
    'liquidus_temp_at_10pct',
    'liquidus_temp_at_20pct',
    'liquidus_temp_at_30pct',
    'liquidus_temp_at_40pct',
    'liquidus_temp_at_50pct',
    'liquidus_temp_at_60pct',
    'liquidus_temp_at_70pct',
    'liquidus_temp_at_80pct',
    'liquidus_temp_at_90pct',
    'liquidus_temp_min',
    'liquidus_temp_max',
    'liquidus_temp_range',
    'liquidus_temp_mean',
    'liquidus_temp_std',
]

X = df.drop(columns = drop_columns, errors = 'ignore')
# Convert to numeric and fill missing with medians
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors = 'coerce')

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

base_model = XGBRegressor(
    n_estimators = 1000,
    max_depth = 12,
    learning_rate = 0.05,
    n_jobs = -1,
)

model = MultiOutputRegressor(base_model)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

overall_mae = mean_absolute_error(y_test, y_pred)
overall_r2 = r2_score(y_test, y_pred)

compositions = [10, 20, 30, 40, 50, 60, 70, 80, 90]

for i,comp in enumerate(compositions):
    mae_at_comp = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2_at_comp = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{comp:2d}%: MAE = {mae_at_comp:6.1f} K, R2 = {r2_at_comp:.3f}")

errors = np.abs(y_test.values - y_pred).mean(axis=1)
best_idx = errors.argmin()
# Choose example with lowest error to display
plt.figure(figsize=(8, 6))
plt.plot(compositions, y_test.iloc[best_idx].values, 'o-', label='Actual', linewidth=2)
plt.plot(compositions, y_pred[best_idx], 's--', label='Predicted', linewidth=2)
plt.xlabel('Composition (%)', fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.title(f'Liquidus Curve Prediction\nMAE = {errors[best_idx]:.1f}K', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('liquidus_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Model MAE: {overall_mae:.1f}K")
