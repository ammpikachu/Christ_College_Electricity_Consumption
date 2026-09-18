
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("electricity_bill.pkl")
poly = joblib.load("polynomial_features.pkl")

# Page title
st.title("⚡ Electricity Bill Prediction")

st.write(
    "Predict your electricity bill based on AC consumption units "
    "using Polynomial Regression."
)

# AC Units input
ac_units = st.number_input(
    "Enter AC Consumption (AC Units)",
    min_value=-100000.0,
    max_value=100000.0,
    value=100.0,
    step=1.0
)

# Validate input
if ac_units < 0 or ac_units > 150:

    st.error(
        "⚠️ Min value should be 0 and max value should be 150. "
        "Please enter a value within this range."
    )

else:

    # Prediction button
    if st.button("Predict Bill"):

        # Create input DataFrame
        new_data = pd.DataFrame({
            "AC_Units": [ac_units]
        })

        # Convert input into polynomial features
        new_data_poly = poly.transform(new_data)

        # Predict electricity bill
        prediction = model.predict(new_data_poly)

        # Success message
        st.success("Model predicted successfully!")

        # Display prediction
        st.metric(
            "Predicted Electricity Bill",
            f"₹{prediction[0]:,.2f}"
        )

