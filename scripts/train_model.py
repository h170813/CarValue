import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv('../data/used_cars.csv')

# Makes mileage a string
if 'milage' in data.columns:
    data['milage'] = data['milage'].astype(str).fillna('')
    data['milage'] = data['milage'].str.replace(' mi.', '', regex=False)
    data['milage'] = data['milage'].str.replace(',', '', regex=False).astype(float)
else:
    print("Column 'milage' not found in the dataset")

# Drop columns with high missing values
data = data.dropna(axis=1, thresh=len(data) * 0.8)  # Keep columns with at least 80% non-null values
print("Columns after dropping high-NA columns:", data.columns.tolist())

# Clean and preprocess specific columns
def clean_numeric(value, remove_chars):
    if isinstance(value, str):
        for char in remove_chars:
            value = value.replace(char, '')
        try:
            return float(value)
        except ValueError:
            return None
    return value

# Clean price column
data['price'] = data['price'].apply(clean_numeric, remove_chars=['$', ','])

# Drop rows with missing target variable
data = data.dropna(subset=['price'])
print("Dataset shape after cleaning:", data.shape)

# Check if columns exist before one-hot encoding
columns_to_encode = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col']
available_columns = [col for col in columns_to_encode if col in data.columns]

if available_columns:
    data = pd.get_dummies(data, columns=available_columns, drop_first=True)
else:
    print(f"None of the specified columns for encoding found: {columns_to_encode}")


# Verify column existence and debug preprocessing
print("Initial columns:", data.columns.tolist())

# Drop columns with high missing values
data = data.dropna(axis=1, thresh=len(data) * 0.8)
print("Columns after dropping high-NA columns:", data.columns.tolist())

# Rename or standardize column names if needed
data.rename(columns={"milage": "mileage"}, inplace=True)

# Ensure target and key columns are present
columns_to_encode = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col']
available_columns = [col for col in columns_to_encode if col in data.columns]
print(f"Columns available for encoding: {available_columns}")

# Apply one-hot encoding
if available_columns:
    data = pd.get_dummies(data, columns=available_columns, drop_first=True)
else:
    print(f"None of the specified columns for encoding found: {columns_to_encode}")

# Feature selection
numeric_features = ['model_year', 'milage', 'engine']  # Add other relevant numeric features
encoded_features = [col for col in data.columns if col.startswith(('brand_', 'fuel_type_', 'transmission_'))]
selected_features = numeric_features + encoded_features

# Ensure selected features exist in the dataset
selected_features = [col for col in selected_features if col in data.columns]

# Define features (X) and target variable (y)
X = data[selected_features]
y = data['price']

print(f"Selected features for training: {selected_features}")
print(f"Feature matrix shape: {X.shape}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



# Optional: Train a Random Forest Regressor for better performance
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
print("Random Forest Mean Squared Error:", rf_mse)

# Save the best model
joblib.dump(rf_model if rf_mse < mse else model, '../static/car_price_model.pkl')
print("Best model saved as 'car_price_model.pkl'")
