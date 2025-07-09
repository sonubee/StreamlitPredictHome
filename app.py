import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# 1) Load model & scaler
model = tf.keras.models.load_model('home_price_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2) Load the shortened CSV for defaults & feature discovery
df_form = pd.read_csv('MLS_Sold_Data_For_Training_Shortened.csv')

# 3) Get the exact feature order expected by the scaler
feature_names = list(scaler.feature_names_in_)

# 4) Split zip-code dummies vs. true numeric features
zip_feats = [f for f in feature_names if f.startswith('ZipCode_')]
num_feats = [f for f in feature_names if f not in zip_feats]

# 5) Build the form
st.title("🏠 Sacramento Home Price Predictor")

# 5a) Numeric & binary inputs
input_data = {}
for feat in num_feats:
    vals = df_form[feat].dropna().unique()
    # checkbox for binary flags
    if set(vals) <= {0, 1}:
        default = bool(df_form[feat].mode().iloc[0])
        input_data[feat] = int(st.checkbox(feat, value=default))
    else:
        mn = int(df_form[feat].min())
        mx = int(df_form[feat].max())
        md = int(df_form[feat].median())
        input_data[feat] = st.number_input(
            label=feat,
            min_value=mn,
            max_value=mx,
            value=md,
            step=1
        )

# 5b) Zip code selector
zipcodes = sorted(int(f.split("_")[1]) for f in zip_feats)
zipcode = st.selectbox("Zip Code", zipcodes)

# 5c) One-hot encode ZIP
for f in zip_feats:
    z = int(f.split("_")[1])
    input_data[f] = 1 if z == zipcode else 0

# 6) Predict button
if st.button("Predict Price"):
    # assemble in correct order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    X_scaled = scaler.transform(input_df)
    pred_price = model.predict(X_scaled)[0, 0]

    st.subheader("Estimated Sale Price")
    st.write(f"💰 **${int(round(pred_price)):,.0f}**")
