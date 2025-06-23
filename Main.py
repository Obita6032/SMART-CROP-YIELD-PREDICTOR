import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load data from Our World in Data
url = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'
df = pd.read_csv(url)

# Use only necessary columns
df = df[['country', 'year', 'co2_per_capita']]
df.columns = ['Country', 'Year', 'CO2_per_capita']

# Preprocess
df = df.sort_values(['Country','Year'])
df = df.dropna(subset=['CO2_per_capita'])

# Feature engineering
df['lag1'] = df.groupby('Country')['CO2_per_capita'].shift(1)
df['lag3_avg'] = df.groupby('Country')['CO2_per_capita'].shift(1).rolling(3).mean().reset_index(drop=True)

df = df.dropna()
features = ['lag1','lag3_avg']
X = df[features]
y = df['CO2_per_capita']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        'MAE': mean_absolute_error(y_test, preds),
        'R2': r2_score(y_test, preds),
        'preds': preds
    }

# Print evaluation results
print("\nModel Evaluation Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: MAE={metrics['MAE']:.3f}, R²={metrics['R2']:.3f}")

# Plot predictions vs actual (Random Forest)
plt.figure(figsize=(8,5))
plt.scatter(y_test, results['RandomForest']['preds'], alpha=0.3)
plt.xlabel('Actual CO₂ per capita')
plt.ylabel('Predicted CO₂ per capita')
plt.title('Random Forest Predictions vs Actual')
plt.grid(True)
plt.show()
