
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

df = pd.read_csv('phase_diagram_features_with_metadata.csv')

y = df['n_phases']

# Drop str labels, and data that might give away the answer
drop_columns = [
    'n_phases',
    'entry_id',
    'composition',
    'elements_str',
    'n_solid_phases',
    'n_liquid_phases',
    'n_eutectics',   
    'has_eutectic',
    'eutectic_temp',
    'eutectic_comp',
    'lowest_eutectic_temp',
    'highest_eutectic_temp',
]
X = df.drop(columns=drop_columns, errors = 'ignore')

# Convert each col to numeric and fill NaN values with the Median value
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors = 'coerce')
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

# Use LGBMBoost Regressor model, tuned for best R2 score
# From testing, LGBM performs slightly better than RGBoost for this metric
model = LGBMRegressor(
    n_estimators = 2000,
    max_depth = 15,
    learning_rate = 0.04,
    n_jobs=-1,
    verbose =-1 
    )
model.fit(X_train, y_train)

# Make predictions on test set and compare with actual values
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'The R2 score: {r2:.4f}')

# Check feature importance:
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Check if there is a significant feature(Likely leaking the y value)
if importance.iloc[0]['importance'] > 0.3:
    print(importance.iloc[0])

# Plot the results
plt.figure(figsize = (8, 6))
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha = 0.3)
plt.savefig('n_phases.png', dpi = 150, bbox_inches = 'tight')
plt.close()
