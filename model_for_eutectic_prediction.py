import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('phase_diagram_features_with_metadata.csv')
df = df[df['has_eutectic'] == 1].copy()

# Drop columns that might give away the answer
# Assign eutectic_temp as our y 
y = df['eutectic_temp']
drop_columns = [
    'entry_id', 'elements_str', 'composition',
    'eutectic_temp',
    'eutectic_comp',
    'n_eutectics',
    'has_eutectic',
    'first_eutectic_temp',     
    'first_eutectic_comp',      
    'lowest_eutectic_temp',     
    'highest_eutectic_temp',    
    'eutectic_temp_min',        
    'eutectic_temp_max',        
    'eutectic_temp_mean',       
    'eutectic_temp_std',        
    'eutectic_comp_min',       
    'eutectic_comp_max',        
    'eutectic_comp_mean',     
    'eutectic_comp_std',       
    'eutectic_spacing_mean',   
    'eutectic_spacing_min',    
    'eutectic_spacing_max',  
]
X = df.drop(columns = drop_columns, errors = 'ignore')

# Make sure everything is a number, fill NaN with median value
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors = 'coerce')
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Use XGBRegressor model for it's speed and high accuracy
model = XGBRegressor(n_estimators = 200, max_depth = 8, learning_rate = 0.05)
model.fit(X_train, y_train)

# Make predictions and compute the mean squared error as well as the r2 score
# R2 score is used as the more accurate metric
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2:.3f}")

# Plot predicted results vs. actual results
plt.figure(figsize = (8, 6))
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2)
plt.xlabel('Actual Temp(K)')
plt.ylabel('Predicted Temp(K)')
plt.grid(True, alpha = 0.3)
plt.savefig('plot.png', dpi = 150, bbox_inches = 'tight')
plt.close()
