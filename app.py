import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# —1) Load your trained model & scaler—
model = tf.keras.models.load_model('home_price_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# —2) Load sample data for form generation—
#    This CSV must contain ALL the columns you trained on,
#    even if it only has a few rows.
df_form = pd.read_csv('MLS_Sold_Data_For_Training_Shortened.csv')

# —3) ZIP code handling—
zipcodes = sorted(df_form['ZipCode'].dropna().astype(int).unique())
base_zip   = zipcodes[0]        # drop_first in one-hot during training
dummy_zips = zipcodes[1:]

# —4) Select only numeric & binary features (drop text/object columns)—
all_numeric = df_form.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in all_numeric if c not in ['SalePrice', 'ZipCode']]

st.title("🏠 Sacramento Home Price Predictor")

# —5) Dynamically create form inputs for each numeric feature—
input_data = {}
for col in feature_cols:
    vals = df_form[col].dropna().unique()
    # If only 0/1 → checkbox
    if set(vals) <= {0, 1}:
        input_data[col] = int(st.checkbox(col, value=bool(df_form[col].mode()[0])))
    else:
        mn = float(df_form[col].min())
        mx = float(df_form[col].max())
        md = float(df_form[col].median())
        input_data[col] = st.number_input(
            label=col,
            min_value=mn,
            max_value=mx,
            value=md
        )

# —6) Zip code selector—
zipcode = st.selectbox("Zip Code", zipcodes)

# —7) One-hot encode zip exactly as during training—
for z in dummy_zips:
    input_data[f"ZipCode_{z}"] = 1 if zipcode == z else 0

# —8) Build DataFrame, scale, and predict—
input_df = pd.DataFrame([input_data])
X_scaled = scaler.transform(input_df)
pred_price = model.predict(X_scaled)[0, 0]

# —9) Display result—
st.subheader("Estimated Sale Price")
st.write(f"💰 **${pred_price:,.0f}**")