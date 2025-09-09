# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
import tempfile
import os
import io
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Paths
MODEL_PATH = "catboost_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
PREPROCESSED_PATH = "preprocessed.csv"

# -----------------------------
# Load artifacts
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
raw_data = pd.read_csv(PREPROCESSED_PATH)

# -----------------------------
# Columns
categorical_cols = ['Locality', 'Sale_Condition', 'Parking_Facility',
                    'Building_Type', 'Utilities_Available', 'Street_Type', 'Zoning_Type']

numeric_cols_manual = ['Registration_Fee', 'Commission', 'Sale_Year', 'Build_Year', 'Interior_SqFt', 'Distance_To_Main_Road']
numeric_cols_other = ['Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms',
                      'Quality_Score_Rooms', 'Quality_Score_Bathroom',
                      'Quality_Score_Bedroom', 'Quality_Score_Overall']

# -----------------------------
# Streamlit Config
st.set_page_config(page_title="Chennai House Valuation", layout="wide")

# -----------------------------
# Session state defaults
if "user_input" not in st.session_state:
    st.session_state.user_input = {col: None for col in categorical_cols + numeric_cols_manual + numeric_cols_other}
    for col in numeric_cols_manual + numeric_cols_other:
        st.session_state.user_input[col] = 0

if "property_image" not in st.session_state:
    st.session_state.property_image = None

# -----------------------------
# Helpers
def safe_str(x):
    return str(x).encode('latin-1', 'replace').decode('latin-1')

# -----------------------------
# Validation function with rules
def validate_inputs(inputs):
    missing = []
    errors = []

    # 1) Check required fields
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            missing.append(k)

    # 2) Building Year vs Sale Year
    build_year = inputs.get("Build_Year", 0)
    sale_year = inputs.get("Sale_Year", 0)
    if build_year and sale_year:
        if build_year > sale_year:
            errors.append("Build_Year cannot be after Sale_Year")
        if build_year < 1800 or build_year > datetime.now().year:
            errors.append(f"Build_Year must be between 1800 and {datetime.now().year}")

    # 3) Quality scores 0-5
    quality_cols = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
    for q in quality_cols:
        val = inputs.get(q, 0)
        if val < 0 or val > 5:
            errors.append(f"{q} must be between 0 and 5")

    # 4) Registration fee <= 7% of Sale_Price, commission <= 10%
    sale_price = inputs.get("Sale_Price", 0)
    if sale_price > 0:
        reg_fee = inputs.get("Registration_Fee", 0)
        if reg_fee > 0.07 * sale_price:
            errors.append("Registration_Fee cannot exceed 7% of Sale Price")
        commission = inputs.get("Commission", 0)
        if commission > 0.10 * sale_price:
            errors.append("Commission cannot exceed 10% of Sale Price")

    # 5) Number of bedrooms for commercial or others
    building_type = inputs.get("Building_Type", "").lower()
    if building_type in ["commercial", "others"]:
        inputs['Num_Bedrooms'] = 0  # forcibly set to 0

    # 6) Bedrooms limit <= 10
    if inputs.get("Num_Bedrooms", 0) > 10:
        errors.append("Num_Bedrooms cannot exceed 10")

    # 7) Distance <= 1000
    if inputs.get("Distance_To_Main_Road", 0) > 1000:
        errors.append("Distance_To_Main_Road cannot exceed 1000 meters")

    # Return result
    if missing or errors:
        msg = ""
        if missing:
            msg += f"Missing fields: {', '.join(missing)}. "
        if errors:
            msg += "Errors: " + "; ".join(errors)
        return False, msg.strip()
    return True, ""

# -----------------------------
# Input section (with image upload & preview)
def input_section():
    st.markdown("<h2 style='text-align:center;'>🏠 House Details</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Categorical inputs
    with col1:
        for col in categorical_cols[:len(categorical_cols)//2]:
            options = ["--Select--"] + sorted(raw_data[col].dropna().unique().tolist())
            selected = st.selectbox(col, options, index=0, key=col)
            st.session_state.user_input[col] = selected if selected != "--Select--" else None

    with col2:
        for col in categorical_cols[len(categorical_cols)//2:]:
            options = ["--Select--"] + sorted(raw_data[col].dropna().unique().tolist())
            selected = st.selectbox(col, options, index=0, key=col)
            st.session_state.user_input[col] = selected if selected != "--Select--" else None

    # Numeric inputs
    with col1:
        for col in numeric_cols_manual[:len(numeric_cols_manual)//2]:
            st.session_state.user_input[col] = st.number_input(col, value=st.session_state.user_input[col], key=col)

    with col2:
        for col in numeric_cols_manual[len(numeric_cols_manual)//2:]:
            st.session_state.user_input[col] = st.number_input(col, value=st.session_state.user_input[col], key=col)

    with col1:
        for col in numeric_cols_other[:len(numeric_cols_other)//2]:
            st.session_state.user_input[col] = st.number_input(col, value=st.session_state.user_input[col], key=col)

    with col2:
        for col in numeric_cols_other[len(numeric_cols_other)//2:]:
            st.session_state.user_input[col] = st.number_input(col, value=st.session_state.user_input[col], key=col)

    # Building age
    if st.session_state.user_input["Build_Year"] and st.session_state.user_input["Sale_Year"]:
        st.session_state.user_input["Building_Age"] = max(st.session_state.user_input["Sale_Year"] - st.session_state.user_input["Build_Year"], 0)
    else:
        st.session_state.user_input["Building_Age"] = 0

    # Image Upload + Preview
    uploaded = st.file_uploader("📸 Upload Property Image (jpg / png)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        st.session_state.property_image = uploaded
        st.image(uploaded, caption="Uploaded property image preview", use_column_width=True)

# -----------------------------
# Keep all other functions (charts, PDF generation) as in your previous code:
# create_plotly_charts, create_pdf_images, generate_pdf
# ...
# -----------------------------
# Pages / Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Prediction", "Charts", "Insights", "Report"])

if page == "Prediction":
    input_section()
    st.subheader("💰 Predicted House Price")
    valid, msg = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning(msg)
    else:
        input_encoded = {}
        for col in categorical_cols:
            val = st.session_state.user_input[col]
            try:
                input_encoded[col] = label_encoders[col].transform([val])[0]
            except Exception:
                input_encoded[col] = 0
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input.get(col, 0)
        input_df = pd.DataFrame([input_encoded], columns=list(input_encoded.keys()))
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")

elif page == "Charts":
    input_section()
    st.subheader("📊 Interactive Charts")
    valid, msg = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning(msg)
    else:
        figs = create_plotly_charts(st.session_state.user_input, raw_data)
        keys = list(figs.keys())
        for i in range(0, len(keys), 2):
            cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx < len(keys):
                    cols[j].plotly_chart(figs[keys[idx]], use_container_width=True)

elif page == "Insights":
    input_section()
    st.subheader("📈 Insights & Suggestions")
    valid, msg = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning(msg)
    else:
        st.markdown("### 🔍 Key Insights")
        st.write(f"- **Interior Space:** {st.session_state.user_input['Interior_SqFt']} sq.ft.")
        st.write(f"- **Bedrooms/Bathrooms:** {st.session_state.user_input['Num_Bedrooms']} / {st.session_state.user_input['Num_Bathrooms']}")
        st.write(f"- **Quality Scores:**")
        for score in ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']:
            st.write(f"  - {score}: {st.session_state.user_input[score]}")
        st.write(f"- **Distance to Road:** {st.session_state.user_input['Distance_To_Main_Road']} meters")
        st.write(f"- **Financials:** Registration {st.session_state.user_input['Registration_Fee']}, Commission {st.session_state.user_input['Commission']}")
        figs = create_plotly_charts(st.session_state.user_input, raw_data)
        st.plotly_chart(figs['bar_features'], use_container_width=True)
        st.plotly_chart(figs['donut_rooms'], use_container_width=True)

elif page == "Report":
    input_section()
    st.subheader("📄 Generate PDF Report")
    valid, msg = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning(msg)
    else:
        input_encoded = {}
        for col in categorical_cols:
            val = st.session_state.user_input[col]
            try:
                input_encoded[col] = label_encoders[col].transform([val])[0]
            except Exception:
                input_encoded[col] = 0
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input.get(col, 0)
        input_df = pd.DataFrame([input_encoded])
        prediction = model.predict(input_df)[0]
        pdf_images = create_pdf_images(st.session_state.user_input, raw_data)
        pdf_bytes = generate_pdf(st.session_state.user_input, pdf_images, prediction, st.session_state.property_image)
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"HouseValuation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
        st.success("PDF generated — click the download button above.")
