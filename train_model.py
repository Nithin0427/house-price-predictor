import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys

DATA_PATH = "housing_price_dataset.csv"
MODEL_PATH = "house_price_model.pkl"

def rupee(x: float) -> str:
    return f"₹{x:,.2f}"

def fail(msg: str):
    print(f"\n❌ {msg}")
    sys.exit(1)

def main():
    # -------- Load --------
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        fail(f"Could not read {DATA_PATH}: {e}")

    # Normalize column names (strip spaces, exact expected names)
    df.columns = [c.strip() for c in df.columns]

    required = ["SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"CSV missing required columns: {missing}")

    # -------- Coerce types safely --------
    numeric_cols = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "Price"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Neighborhood"] = df["Neighborhood"].astype(str)

    # Drop rows with NaNs in required fields
    before = len(df)
    df = df.dropna(subset=required).copy()
    after = len(df)
    if after == 0:
        fail("All rows dropped after type coercion / NaN removal. Check your CSV content.")
    if before != after:
        print(f"ℹ️ Dropped {before - after} rows with missing/invalid required values.")

    # -------- Basic sanity filters --------
    current_year = datetime.now().year
    # Year bounds
    df = df[(df["YearBuilt"] >= 1800) & (df["YearBuilt"] <= current_year)]
    # SquareFeet realistic bounds (allow very large but cap truly absurd)
    df = df[(df["SquareFeet"] > 100) & (df["SquareFeet"] <= 1_000_000)]
    # Remove non-positive targets (break log1p)
    neg = (df["Price"] <= 0).sum()
    if neg > 0:
        print(f"ℹ️ Dropping {neg} rows with non-positive Price (<= 0) to enable log-transform.")
        df = df[df["Price"] > 0]

    if len(df) < 100:
        print("⚠️ Very small dataset after cleaning. Model may be unstable.")

    # Feature engineering
    df["HouseAge"] = current_year - df["YearBuilt"]

    # -------- Split X/y --------
    y = np.log1p(df["Price"])  # safe now (Price > 0)
    X = df.drop(columns=["Price"])

    categorical = ["Neighborhood"]
    numerical = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "HouseAge"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # -------- Model --------
    regressor = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        random_state=42,
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    # -------- Train/test --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training Gradient Boosting model...")
    model.fit(X_train, y_train)
    print("✅ Training complete!\n")

    # -------- Evaluate (inverse from log scale) --------
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    r2 = r2_score(y_true, y_pred)

    print("📊 Model Performance (Price scale)")
    print(f"MAE:  {rupee(mae)}")
    print(f"RMSE: {rupee(rmse)}")
    print(f"R²:   {r2:.4f}")

    # -------- Feature importances (optional, informative) --------
    try:
        # Build feature names from preprocessor
        pre = model.named_steps["preprocessor"]
        num_names = numerical
        cat_encoder = pre.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(categorical).tolist()
        all_feature_names = num_names + cat_names

        importances = model.named_steps["regressor"].feature_importances_
        pairs = sorted(zip(all_feature_names, importances), key=lambda t: t[1], reverse=True)
        top = pairs[:10]
        print("\n⭐ Top 10 features:")
        for name, val in top:
            print(f"  {name:20s}  {val:.4f}")
    except Exception as e:
        print(f"\n(Feature importance display skipped: {e})")

    # -------- Save --------
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Saved model to '{MODEL_PATH}'")

if __name__ == "__main__":
    main()
