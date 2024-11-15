from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and prepare dataset
def train_model():
    data = pd.read_csv('data/processed_used_cars.csv')
    data = pd.get_dummies(data, drop_first=True)

    # Split data into features and target
    X = data.drop('price', axis=1)
    y = data['price']

    # Split dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'static/car_price_model.pkl')

    # Display top 10 most important features
    feature_importances = model.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
    print("Top 10 Most Important Features:")
    print(top_features)

# Uncomment to train model
# train_model()

# Flask web application
app = Flask(__name__)

# Load the saved model
model = joblib.load('static/car_price_model.pkl')

# List of valid car brands based on the dataset
valid_car_brands = [
    'BMW', 'Audi', 'Ford', 'Toyota', 'Honda', 'Mercedes-Benz', 'Volkswagen',
    'Chevrolet', 'Nissan', 'Hyundai', 'Kia', 'Subaru', 'Mazda', 'Jeep', 'Lexus',
    'Dodge', 'Porsche', 'Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce'
]

@app.route('/')
def home():
    return render_template('index.html', car_brands=valid_car_brands)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        model_year = int(request.form['model_year'])
        car_brand = request.form['car_brand'].capitalize()
        fuel_type = request.form['fuel_type'].capitalize()
        milage = int(request.form['milage'])

        # Validate car brand
        if car_brand not in valid_car_brands:
            return render_template('index.html', error=f"Invalid car brand: {car_brand}. Please choose from the listed car brands.", car_brands=valid_car_brands)

        # Create input DataFrame with dynamic brand, fuel type, and milage
        new_data = pd.DataFrame({
            'model_year': [model_year],
            f'brand_{car_brand}': [1],
            f'fuel_type_{fuel_type}': [1],
            'milage': [milage],
        })

        # Align new_data with model's expected features
        expected_features = model.feature_names_in_
        new_data = new_data.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        predicted_price = model.predict(new_data)[0]
        return render_template('index.html', prediction=f"${predicted_price:,.2f}", car_brands=valid_car_brands)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}", car_brands=valid_car_brands)

if __name__ == '__main__':
    app.run(debug=True)
