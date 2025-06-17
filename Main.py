import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv("combined_crop_data.csv")

# Encode categorical columns
label_enc = LabelEncoder()
df['Crop'] = label_enc.fit_transform(df['Crop'])
df['Soil Type'] = label_enc.fit_transform(df['Soil Type'])
df['Region'] = label_enc.fit_transform(df['Region'])

# Features and target
X = df[['Crop', 'Soil Type', 'Rainfall', 'Temperature', 'Region']]
y = df['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
