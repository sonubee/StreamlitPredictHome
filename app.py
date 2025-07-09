import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 1) Load your model
model = tf.keras.models.load_model('home_price_model.keras')

# 2) (Re)load your scaler or recreate it — you can pickle it in Colab similarly:
import pickle
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🏠 Sacramento Home Price Predictor")

# 3) Build a simple form for inputs
bedrooms   = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathrooms  = st.number_input("Full Baths", min_value=0, max_value=10, value=2)
sqft       = st.number_input("Square Footage", min_value=100, value=1000)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
lot_size   = st.number_input("Lot Size (sqft)", min_value=0, value=5000)

# load a tiny CSV just to get the column names (or load from a saved JSON)
df_cols = pd.read_csv("MLS_Sold_Data_For_Training_Zips_Only.csv", nrows=0)
zip_cols = [c for c in df_cols.columns if c.startswith("ZipCode_")]
zipcodes = [int(c.split("_")[1]) for c in zip_cols]

zipcode = st.selectbox("Zip Code", options=zipcodes)

# 4) Encode & scale inputs exactly as in training
input_df = pd.DataFrame([{
    'Bedrooms': bedrooms,
    'FullBaths': bathrooms,
    'LotSize': lot_size,
    'YearBuilt': year_built,
    # … plus one-hot for ZipCode …
}])
# If you one-hot-encoded zips during training, you need the same columns here:
for z in [/*list of all zip codes except the first*/]:
    input_df[f'ZipCode_{z}'] = 1 if z==zipcode else 0

X_scaled = scaler.transform(input_df)

# 5) Make prediction
pred_price = model.predict(X_scaled)[0,0]

st.write(f"## Estimated Sale Price: ${pred_price:,.0f}")