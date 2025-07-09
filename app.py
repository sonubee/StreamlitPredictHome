import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# 1) Load your model & scaler
model = tf.keras.models.load_model('home_price_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2) Sample CSV for defaults & feature discovery
df_form = pd.read_csv('MLS_Sold_Data_For_Training_Shortened.csv')

# 3) Get scaler’s exact feature names
feature_names = list(scaler.feature_names_in_)

# 4) Split out ZIP dummies vs the rest
zip_feats = [f for f in feature_names if f.startswith('ZipCode_')]
num_feats = [f for f in feature_names if f not in zip_feats]

st.title("🏠 Sacramento Home Price Predictor")

# 5) Build inputs for numeric & binary features
input_data = {}
for feat in num_feats:
    vals = df_form[feat].dropna().unique()
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
            step=1,
            format="%d"
        )

# 6) ZIP code selector
zipcodes = sorted(int(f.split("_")[1]) for f in zip_feats)
selected_zip = st.selectbox("Zip Code", zipcodes)

# 7) One-hot encode ZIP
for f in zip_feats:
    z = int(f.split("_")[1])
    input_data[f] = 1 if z == selected_zip else 0

# 8) Session-state for prediction
if "pred_price" not in st.session_state:
    st.session_state.pred_price = None

# 9) Predict button
if st.button("Predict Price"):
    # Build DataFrame in exact order
    df_input = pd.DataFrame([input_data], columns=feature_names)
    X_scaled = scaler.transform(df_input)
    st.session_state.pred_price = model.predict(X_scaled)[0, 0]

# 10) Display result if available
if st.session_state.pred_price is not None:
    price = int(round(st.session_state.pred_price))
    st.subheader("Estimated Sale Price")
    st.write(f"💰 **${price:,.0f}**")
