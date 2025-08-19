# 🏠 House Price Predictor

A **Machine Learning + Flask Web App** that predicts house prices based on square footage, bedrooms, bathrooms, neighborhood, and year built.

---

## 🚀 Features

- 📊 Data preprocessing and feature engineering  
- 🤖 Trained ML models (Linear Regression, Gradient Boosting)  
- 🌐 Flask backend for predictions  
- 🎨 Dark-themed frontend (HTML + CSS)  
- 🖥️ Interactive floating result popup  

---


## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
   cd house-price-predictor
   ```
2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model:
    ```bash
    python train_model.py
    ```
4. Run the app:
    ```bash
    python app.py
    # Open http://localhost:5000
    ```

---


## 📊 Dataset

- SquareFeet: 1000–3000
- Bedrooms: 2–5
- Bathrooms: 1–3
- Neighborhood: Urban, Suburb, Rural
- YearBuilt: 1950–2021
- Price: 0–5 lakhs (cleaned)

---

## 🌟 Future Enhancements

- Add larger datasets for more realistic predictions
- Deploy app on Render/Heroku for live demo
- Add visualizations for trends and insights
- Extend to crores with richer datasets

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy, Scikit-learn
- Flask
- HTML, CSS
