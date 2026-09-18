```python
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

# AC Units input: minimum 1, maximum 150
ac_units = st.number_input(
    "Enter AC Consumption (AC Units)",
    min_value=1.0,
    max_value=150.0,
    value=100.0,
    step=1.0
)

if st.button("Predict Bill"):

    # Create DataFrame with the same column name used during training
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
```
