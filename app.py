from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Base directory and paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'CarPricePredictor.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Cleaned_Car_data.csv')

# Validate assets
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset file not found at: {DATA_PATH}")

# Load model and data
model = pickle.load(open(MODEL_PATH, 'rb'))
car = pd.read_csv(DATA_PATH)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, years=years,
                           fuel_types=fuel_types, prediction_text=None)

@app.route('/get_models')
def get_models():
    company = request.args.get('company')
    models = sorted(car[car['company'] == company]['name'].unique())
    return jsonify({'models': models})

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_data)[0]
    price = f"Estimated Price: ₹ {round(prediction, 2)} Lakh"

    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    return render_template('index.html', companies=companies,
                           years=years, fuel_types=fuel_types,
                           prediction_text=price)

if __name__ == "__main__":
    app.run(debug=True)

