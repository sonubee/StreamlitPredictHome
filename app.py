import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# 1) Load your trained model & scaler
model = tf.keras.models.load_model('home_price_model.keras')
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# 2) Load the shortened CSV for stats (min/max/median/mode)
df_form = pd.read_csv('MLS_Sold_Data_For_Training_Shortened.csv')

# 3) Grab the exact feature names/order that the scaler was fit on
feature_names = list(scaler.feature_names_in_)

# 4) Split into zip‐code dummies vs. true numeric features
zip_features     = [f for f in feature_names if f.startswith('ZipCode_')]
numeric_features = [f for f in feature_names if f not in zip_features]

# 5) Build the Streamlit form
st.title("🏠 Sacramento Home Price Predictor")

# 5a) Numeric and binary features
input_data = {}
for feat in numeric_features:
    vals = df_form[feat].dropna().unique()
    # binary flag?
    if set(vals) <= {0,1}:
        # default to the mode
        default = bool(df_form[feat].mode().iloc[0])
        input_data[feat] = int(st.checkbox(feat, value=default))
    else:
        mn = float(df_form[feat].min())
        mx = float(df_form[feat].max())
        md = float(df_form[feat].median())
        input_data[feat] = st.number_input(
            label=feat,
            min_value=mn,
            max_value=mx,
            value=md
        )

# 5b) ZipCode selector
zipcodes = sorted(int(f.split("_")[1]) for f in zip_features)
zipcode  = st.selectbox("Zip Code", zipcodes)

# 5c) One-hot your selected zip
for f in zip_features:
    z = int(f.split("_")[1])
    input_data[f] = 1 if z == zipcode else 0

# 6) Assemble DataFrame in correct order & predict
input_df = pd.DataFrame([input_data], columns=feature_names)
X_scaled = scaler.transform(input_df)
pred_price = model.predict(X_scaled)[0,0]

# 7) Show result
st.subheader("Estimated Sale Price")
st.write(f"💰 **${pred_price:,.0f}**")
