import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load Model and Dataset
# -----------------------------
model = joblib.load("fraud_detection_model.pkl")
df = pd.read_csv("creditcard.csv")


# -----------------------------
# Streamlit Page Design
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Fraud Detection using Machine Learning")
st.write(
    "This system predicts whether a credit card transaction is genuine or fraudulent "
    "using machine learning models trained on transaction data."
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Transaction Input Method")

option = st.sidebar.radio(
    "Choose input method:",
    ["Select transaction from dataset", "Enter custom transaction values"]
)


# -----------------------------
# Option 1: Select Existing Transaction
# -----------------------------
if option == "Select transaction from dataset":
    st.subheader("Select a Transaction from Dataset")

    row_number = st.number_input(
        "Enter row number from dataset:",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    selected_row = df.iloc[[row_number]]

    actual_class = selected_row["Class"].values[0]
    input_data = selected_row.drop("Class", axis=1)

    st.write("Selected Transaction Data:")
    st.dataframe(selected_row)

    if st.button("Predict Transaction"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Fraudulent Transaction Detected")
        else:
            st.success("Genuine Transaction")

        st.write("Fraud Probability:", round(probability * 100, 2), "%")

        if actual_class == 1:
            st.warning("Actual Class in Dataset: Fraud")
        else:
            st.info("Actual Class in Dataset: Genuine")


# -----------------------------
# Option 2: Enter Custom Values
# -----------------------------
else:
    st.subheader("Enter Custom Transaction Values")

    st.write(
        "Note: V1 to V28 are anonymized PCA features. "
        "For real testing, selecting a transaction from the dataset is easier."
    )

    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)

    feature_values = {}

    for i in range(1, 29):
        feature_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    input_dict = {"Time": time}

    for i in range(1, 29):
        input_dict[f"V{i}"] = feature_values[f"V{i}"]

    input_dict["Amount"] = amount

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict Custom Transaction"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Fraudulent Transaction Detected")
        else:
            st.success("Genuine Transaction")

        st.write("Fraud Probability:", round(probability * 100, 2), "%")