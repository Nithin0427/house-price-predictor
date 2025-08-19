import joblib, pandas as pd

model = joblib.load("house_price_model.pkl")

sample = {
    "SquareFeet": 1600,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Neighborhood": "Suburb",
    "YearBuilt": 2015
}

df = pd.DataFrame([sample])
pred = float(model.predict(df)[0])
print(f"Predicted Price: ₹{pred:,.2f}")
