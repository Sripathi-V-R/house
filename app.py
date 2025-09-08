import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
from datetime import datetime
import tempfile

# -----------------------------
# Load artifacts
MODEL_PATH = "catboost_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "preprocessed.csv"

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)
data = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Chennai House Price Prediction", layout="wide")
st.title("🏠 Chennai House Price Prediction")

categorical_cols = ['Locality', 'Sale_Condition', 'Parking_Facility', 
                    'Building_Type', 'Utilities_Available', 'Street_Type', 'Zoning_Type']

numeric_cols = ['Interior_SqFt', 'Distance_To_Main_Road', 'Num_Bedrooms', 'Num_Bathrooms',
                'Total_Rooms', 'Quality_Score_Rooms', 'Quality_Score_Bathroom', 
                'Quality_Score_Bedroom', 'Quality_Score_Overall', 'Registration_Fee', 
                'Commission', 'Sale_Year', 'Build_Year', 'Building_Age']

# Sidebar inputs for categorical features
user_input = {}
for col in categorical_cols:
    options = data[col].unique().tolist()
    user_input[col] = st.sidebar.selectbox(col, options)

# Numeric inputs
for col in numeric_cols:
    default = int(data[col].median())
    user_input[col] = st.number_input(col, value=default)

# Encode categorical features
for col in categorical_cols:
    le = label_encoders[col]
    user_input[col] = le.transform([user_input[col]])[0]

# Prepare DataFrame
input_df = pd.DataFrame([user_input], columns=categorical_cols + numeric_cols)

# Scale features
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Sale Price"):
    try:
        prediction = model.predict(input_scaled)
        predicted_price = prediction[0]
        st.success(f"Predicted House Price: {predicted_price:,.2f} INR")

        # Generate textual report
        report = f"""
House Sale Report - Chennai

Area: {user_input['Locality']}
Interior SqFt: {user_input['Interior_SqFt']}
Bedrooms: {user_input['Num_Bedrooms']}, Bathrooms: {user_input['Num_Bathrooms']}, Rooms: {user_input['Total_Rooms']}
Building Type: {user_input['Building_Type']}, Park Facility: {user_input['Parking_Facility']}
Quality Scores - Rooms: {user_input['Quality_Score_Rooms']}, Bathroom: {user_input['Quality_Score_Bathroom']}, Bedroom: {user_input['Quality_Score_Bedroom']}, Overall: {user_input['Quality_Score_Overall']}
Registration Fee: {user_input['Registration_Fee']}, Commission: {user_input['Commission']}
Sale Year: {user_input['Sale_Year']}, Build Year: {user_input['Build_Year']}, Building Age: {user_input['Building_Age']}

Predicted House Price: {predicted_price:,.2f} INR
"""

        # Generate bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        features_plot = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
        values_plot = [user_input[f] for f in features_plot]
        ax.bar(features_plot, values_plot, color='skyblue')
        ax.set_title("House Features Overview")
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=180)
        plt.close(fig)
        img_buf.seek(0)

        # Generate PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Chennai House Price Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, report)
        pdf.ln(4)
        pdf.image(img_buf, x=30, w=150)
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)
        temp_pdf.seek(0)
        with open(temp_pdf.name, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"HousePrice_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")
