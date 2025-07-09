import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# 1) Load model & scaler (inference-only)
model = tf.keras.models.load_model('home_price_model.keras', compile=False)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2) Short CSV for defaults & feature discovery
df_form = pd.read_csv('MLS_Sold_Data_For_Training_Shortened.csv')

# 3) Exact feature names and split ZIP dummies vs. numeric
feature_names = list(scaler.feature_names_in_)
zip_feats     = [f for f in feature_names if f.startswith('ZipCode_')]
num_feats     = [f for f in feature_names if f not in zip_feats]

# 4) Prepare ZIP list for selector
zipcodes = sorted(int(f.split("_")[1]) for f in zip_feats)

st.title("🏠 Sacramento Home Price Predictor")

# 5) Build a form so button and inputs rerun together
with st.form("predict_form"):
    inputs = {}
    for feat in num_feats:
        vals = df_form[feat].dropna().unique()
        if set(vals) <= {0, 1}:  # binary flags
            default = bool(df_form[feat].mode().iloc[0])
            inputs[feat] = st.checkbox(feat, value=default)
        else:                   # integer-valued features
            mn = int(df_form[feat].min())
            mx = int(df_form[feat].max())
            md = int(df_form[feat].median())
            inputs[feat] = st.number_input(
                label=feat,
                min_value=mn,
                max_value=mx,
                value=md,
                step=1,
                format="%d"
            )

    selected_zip = st.selectbox("Zip Code", zipcodes)
    submit = st.form_submit_button("Predict Price")

# 6) On submit, assemble, scale, predict, and display
if submit:
    # convert inputs to the right types
    input_data = {feat: int(val) for feat, val in inputs.items()}
    # one-hot encode ZIP dummies
    for f in zip_feats:
        z = int(f.split("_")[1])
        input_data[f] = 1 if z == selected_zip else 0

    # build DataFrame in exact feature order
    df_input = pd.DataFrame([input_data], columns=feature_names)
    X_scaled = scaler.transform(df_input)
    pred = model.predict(X_scaled)[0,0]
    price = int(round(pred))

    st.subheader("Estimated Sale Price")
    st.write(f"💰 **${price:,.0f}**")
