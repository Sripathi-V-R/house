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
# Session State
if "user_input" not in st.session_state:
    st.session_state.user_input = {col: None for col in categorical_cols + numeric_cols_manual + numeric_cols_other}
    for col in numeric_cols_manual + numeric_cols_other:
        st.session_state.user_input[col] = 0

if "property_image" not in st.session_state:
    st.session_state.property_image = None

# -----------------------------
# Validate Inputs
def validate_inputs(inputs):
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return False
    return True

# -----------------------------
# Input Section
def input_section():
    st.markdown("<h2 style='text-align:center;'>🏠 House Details</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Categorical
    with col1:
        for col in categorical_cols[:len(categorical_cols)//2]:
            options = ["--Select--"] + list(raw_data[col].unique())
            selected = st.selectbox(col, options, index=0, key=col)
            st.session_state.user_input[col] = selected if selected != "--Select--" else None

    with col2:
        for col in categorical_cols[len(categorical_cols)//2:]:
            options = ["--Select--"] + list(raw_data[col].unique())
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

    # Image Upload
    st.session_state.property_image = st.file_uploader("📸 Upload Property Image", type=["png", "jpg", "jpeg"])

# -----------------------------
# Charts
def create_charts(user_input):
    figs = []

    # Bar Chart: Key Features
    features_plot = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
    values_plot = [user_input[f] for f in features_plot]
    fig_bar = px.bar(x=features_plot, y=values_plot, text=values_plot, color=features_plot,
                     title="Key House Features", height=400)
    figs.append(fig_bar)

    # Radar Chart: Quality
    quality_scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
    scores = [user_input[q] for q in quality_scores]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=quality_scores + [quality_scores[0]],
                                        fill='toself', name='Quality Scores'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                            title="Quality Assessment Radar", height=400)
    figs.append(fig_radar)

    # Histogram: Interior Space
    fig_hist = px.histogram(raw_data, x="Interior_SqFt", nbins=30, title="Distribution of Interior SqFt")
    figs.append(fig_hist)

    # Donut Chart: Building Type
    fig_donut = px.pie(raw_data, names="Building_Type", hole=0.4, title="Building Type Distribution")
    figs.append(fig_donut)

    # Box Plot: Registration Fee
    fig_box = px.box(raw_data, y="Registration_Fee", title="Registration Fee Spread")
    figs.append(fig_box)

    return figs

# -----------------------------
# PDF Generation
def generate_pdf(user_input, figs, prediction, property_image):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # Header
    pdf.set_fill_color(0, 102, 204)
    pdf.rect(0, 0, 210, 20, 'F')
    pdf.set_xy(0, 5)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 10, "Sunrise Property Valuation Agency", 0, 1, 'C')

    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Professional House Valuation Report", ln=True, align="C")
    pdf.ln(5)

    # Property Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Property Information", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in user_input.items():
        safe_v = str(v).encode('latin-1', 'replace').decode('latin-1')
        safe_k = str(k).encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(60, 6, f"{safe_k}", border=1)
        pdf.cell(0, 6, f"{safe_v}", border=1, ln=True)
    pdf.ln(5)

    # Property Image
    if property_image:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img = Image.open(property_image).convert("RGB")
        img.save(tmp_img.name, "PNG")
        pdf.image(tmp_img.name, x=30, w=120)
        tmp_img.close()
        os.unlink(tmp_img.name)
        pdf.ln(5)

    # Prediction
    pdf.set_fill_color(255, 204, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C")
    pdf.ln(5)

    # Charts
    for fig in figs:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img_bytes = fig.to_image(format="png")  # avoids kaleido dependency
        img = Image.open(io.BytesIO(img_bytes))
        img.save(tmp_file.name, "PNG")
        pdf.image(tmp_file.name, x=20, w=170)
        os.unlink(tmp_file.name)
        pdf.ln(5)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, "Sunrise Property Valuation Agency", align="C")

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    temp_pdf.seek(0)
    with open(temp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
    os.unlink(temp_pdf.name)
    return pdf_bytes

# -----------------------------
# Pages
page = st.sidebar.radio("Navigation", ["Prediction", "Charts", "Insights", "Report"])

if page == "Prediction":
    input_section()
    st.subheader("💰 Predicted House Price")
    if validate_inputs(st.session_state.user_input):
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded])
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")
    else:
        st.warning("Please complete all inputs.")

elif page == "Charts":
    input_section()
    if validate_inputs(st.session_state.user_input):
        figs = create_charts(st.session_state.user_input)
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please fill all inputs.")

elif page == "Insights":
    input_section()
    st.subheader("📈 Insights & Suggestions")
    if validate_inputs(st.session_state.user_input):
        st.markdown("### 🔍 Key Insights")
        st.write(f"- **Interior Space:** {st.session_state.user_input['Interior_SqFt']} sq.ft.")
        st.write(f"- **Bedrooms/Bathrooms:** {st.session_state.user_input['Num_Bedrooms']} / {st.session_state.user_input['Num_Bathrooms']}")
        st.write(f"- **Quality Scores:**")
        for score in ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']:
            st.write(f"  - {score}: {st.session_state.user_input[score]}")
        st.write(f"- **Distance to Road:** {st.session_state.user_input['Distance_To_Main_Road']} meters")
        st.write(f"- **Financials:** Registration {st.session_state.user_input['Registration_Fee']}, Commission {st.session_state.user_input['Commission']}")
    else:
        st.warning("Please fill all inputs.")

elif page == "Report":
    input_section()
    st.subheader("📄 Generate Professional PDF Report")
    if validate_inputs(st.session_state.user_input):
        figs = create_charts(st.session_state.user_input)
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded])
        prediction = model.predict(input_df)[0]

        pdf_bytes = generate_pdf(st.session_state.user_input, figs, prediction, st.session_state.property_image)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"HouseValuation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please fill all inputs before generating report.")
