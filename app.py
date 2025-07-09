import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# —1) Load your trained model & scaler—
model = tf.keras.models.load_model('home_price_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# —2) Load all zip codes from your helper CSV—
zip_df   = pd.read_csv('MLS_Sold_Data_For_Training_Zips_Only.csv')
zipcodes = sorted(zip_df['ZipCode'].astype(int).unique())

# If you used drop_first=True when one-hoting during training,
# the first zip in this list was “dropped” as the base case:
base_zip   = zipcodes[0]
dummy_zips = zipcodes[1:]

st.title("🏠 Sacramento Home Price Predictor")

# —3) Build the input form—
bedrooms   = st.number_input("Bedrooms",      min_value=0,  max_value=10,   value=3)
bathrooms  = st.number_input("Full Baths",    min_value=0,  max_value=10,   value=2)
sqft       = st.number_input("Square Footage",min_value=100,value=1000)
year_built = st.number_input("Year Built",    min_value=1900,max_value=2025,value=2000)
lot_size   = st.number_input("Lot Size (sqft)",min_value=0,  value=5000)
zipcode    = st.selectbox("Zip Code", zipcodes)

# —4) Assemble the feature vector—
# NOTE: replace 'SquareFootage' below with the exact column name your model expects
input_dict = {
    'Bedrooms':      bedrooms,
    'FullBaths':     bathrooms,
    'SquareFootage': sqft,        # ← adjust if your column is named differently
    'YearBuilt':     year_built,
    'LotSize':       lot_size,
}

# Add one-hot columns for every zip *except* the base_zip
for z in dummy_zips:
    input_dict[f'ZipCode_{z}'] = 1 if zipcode == z else 0

input_df = pd.DataFrame([input_dict])

# —5) Scale & predict—
X_scaled   = scaler.transform(input_df)
pred_price = model.predict(X_scaled)[0][0]

# —6) Display result—
st.subheader("Estimated Sale Price")
st.write(f"💰 ${pred_price:,.0f}")