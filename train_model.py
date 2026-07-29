import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("cars.csv")

# Create encoders
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()

df["Brand"] = brand_encoder.fit_transform(df["Brand"])
df["Model"] = model_encoder.fit_transform(df["Model"])

# Features
X = df[[
    "Brand",
    "Model",
    "Year",
    "Kms_Driven"
]]

# Target
y = df["Selling_Price"]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model and encoders
joblib.dump(model, "car_price_model.pkl")
joblib.dump(brand_encoder, "brand_encoder.pkl")
joblib.dump(model_encoder, "model_encoder.pkl")

print("Model trained successfully!")
print("Files created:")
print("- car_price_model.pkl")
print("- brand_encoder.pkl")
print("- model_encoder.pkl")
