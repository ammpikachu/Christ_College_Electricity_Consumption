
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("electricity_bill.pkl")
poly = joblib.load("polynomial_features.pkl")

st.title("⚡ Electricity Bill Prediction")

st.write(
    "Predict your electricity bill based on AC consumption units "
    "using Polynomial Regression."
)

# Input
ac_units = st.number_input(
    "Enter AC Consumption (AC Units)",
    min_value=0.0,
    value=100.0,
    step=1.0
)

# Show warning immediately for invalid input
if ac_units <= 0:
    st.error("⚠️ AC Units cannot be 0 or less. Please enter a value between 1 and 150.")

elif ac_units > 150:
    st.error("⚠️ AC Units cannot be more than 150. Please enter a value between 1 and 150.")

else:
    # Predict only when input is valid
    if st.button("Predict Bill"):

        new_data = pd.DataFrame({
            "AC_Units": [ac_units]
        })

        # Convert input into polynomial features
        new_data_poly = poly.transform(new_data)

        # Make prediction
        prediction = model.predict(new_data_poly)

        st.success("Model predicted successfully!")

        st.metric(
            "Predicted Electricity Bill",
            f"₹{prediction[0]:,.2f}"
        )

