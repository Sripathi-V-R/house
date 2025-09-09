import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

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
categorical_cols = [
    'Locality', 'Sale_Condition', 'Parking_Facility',
    'Building_Type', 'Utilities_Available', 'Street_Type', 'Zoning_Type'
]

numeric_cols_manual = [
    'Registration_Fee', 'Commission', 'Sale_Year',
    'Build_Year', 'Interior_SqFt', 'Distance_To_Main_Road'
]

numeric_cols_other = [
    'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms',
    'Quality_Score_Rooms', 'Quality_Score_Bathroom',
    'Quality_Score_Bedroom', 'Quality_Score_Overall'
]

# -----------------------------
# Streamlit Config
st.set_page_config(page_title="🏡 Chennai House Valuation", layout="wide")

# Hero Section
st.markdown("""
    <div style="background-color:#0066cc;padding:20px;border-radius:10px;margin-bottom:20px">
        <h1 style="color:white;text-align:center;">Sunrise Property Valuation</h1>
        <p style="color:white;text-align:center;font-size:18px;">AI-powered House Valuation & Professional PDF Reports</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("📌 Navigation", ["Prediction", "Charts", "Insights", "Report"])

# -----------------------------
# Initialize user input
if "user_input" not in st.session_state:
    st.session_state.user_input = {col: None for col in categorical_cols + numeric_cols_manual + numeric_cols_other}
    for col in numeric_cols_manual + numeric_cols_other:
        st.session_state.user_input[col] = 0

if "property_image" not in st.session_state:
    st.session_state.property_image = None

# -----------------------------
# Input Section
def input_section():
    st.subheader("🏠 Enter Property Details")

    # Two-column layout
    col1, col2 = st.columns(2)

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

    st.markdown("### 📏 Numeric Inputs")
    for col in numeric_cols_manual + numeric_cols_other:
        st.session_state.user_input[col] = st.number_input(col, value=st.session_state.user_input[col], key=col)

    # Auto-calculate building age
    if st.session_state.user_input["Build_Year"] and st.session_state.user_input["Sale_Year"]:
        st.session_state.user_input["Building_Age"] = max(
            st.session_state.user_input["Sale_Year"] - st.session_state.user_input["Build_Year"], 0
        )
    else:
        st.session_state.user_input["Building_Age"] = 0

    # Property Image Upload
    uploaded_file = st.file_uploader("📸 Upload Property Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.session_state.property_image = uploaded_file
        st.image(uploaded_file, caption="Uploaded Property Image", use_column_width=True)

# -----------------------------
# Validate Inputs
def validate_inputs(inputs):
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return False
    return True

# -----------------------------
# Dynamic Plotly Charts
def create_charts(user_input, raw_data):
    figs = []

    # 1. Bar Chart
    bar_fig = px.bar(
        x=['Interior SqFt', 'Bedrooms', 'Bathrooms', 'Total Rooms'],
        y=[user_input['Interior_SqFt'], user_input['Num_Bedrooms'],
           user_input['Num_Bathrooms'], user_input['Total_Rooms']],
        title="Key House Features",
        labels={'x': 'Feature', 'y': 'Value'},
        color=['Interior SqFt', 'Bedrooms', 'Bathrooms', 'Total Rooms']
    )
    figs.append(bar_fig)

    # 2. Radar Chart
    quality_scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom',
                      'Quality_Score_Bedroom', 'Quality_Score_Overall']
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[user_input[q] for q in quality_scores],
        theta=quality_scores,
        fill='toself',
        name='Quality Scores'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title="Quality Scores Radar"
    )
    figs.append(radar_fig)

    # 3. Histogram (Interior SqFt)
    hist_fig = px.histogram(
        raw_data, x="Interior_SqFt", nbins=30,
        title="Distribution of Interior SqFt"
    )
    hist_fig.add_vline(x=user_input['Interior_SqFt'], line_width=3,
                       line_dash="dash", line_color="red")
    figs.append(hist_fig)

    # 4. Donut / Pie (Room distribution)
    pie_fig = px.pie(
        names=["Bedrooms", "Bathrooms", "Other Rooms"],
        values=[user_input['Num_Bedrooms'],
                user_input['Num_Bathrooms'],
                max(user_input['Total_Rooms'] - (
                    user_input['Num_Bedrooms'] + user_input['Num_Bathrooms']), 0)],
        hole=0.4,
        title="Room Distribution"
    )
    figs.append(pie_fig)

    # 5. Box Plot (Rooms vs Price if available)
    if "Total_Rooms" in raw_data.columns and "Sale_Price" in raw_data.columns:
        box_fig = px.box(
            raw_data, x="Total_Rooms", y="Sale_Price",
            title="Price Distribution by Total Rooms"
        )
        figs.append(box_fig)

    return figs

# -----------------------------
# PDF Generator
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
        fig.write_image(tmp_file.name, format="png")
        tmp_file.close()
        pdf.image(tmp_file.name, x=20, w=170)
        os.unlink(tmp_file.name)
        pdf.ln(5)

    # Insights
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Insights & Suggestions", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"- Interior Space: {user_input['Interior_SqFt']} sq.ft.\n"
                          f"- Bedrooms/Bathrooms: {user_input['Num_Bedrooms']} / {user_input['Num_Bathrooms']}.\n"
                          f"- Quality Scores: Rooms {user_input['Quality_Score_Rooms']}, "
                          f"Bathroom {user_input['Quality_Score_Bathroom']}, "
                          f"Bedroom {user_input['Quality_Score_Bedroom']}, "
                          f"Overall {user_input['Quality_Score_Overall']}.\n"
                          f"- Distance from Main Road: {user_input['Distance_To_Main_Road']} meters.\n"
                          f"- Registration Fee: {user_input['Registration_Fee']}, "
                          f"Commission: {user_input['Commission']}.\n")

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
if page == "Prediction":
    input_section()
    st.subheader("💰 Predicted House Price")
    if validate_inputs(st.session_state.user_input):
        # Encode
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded], columns=categorical_cols + numeric_cols_manual + numeric_cols_other + ["Building_Age"])
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")
    else:
        st.warning("Please fill all inputs.")

elif page == "Charts":
    input_section()
    if validate_inputs(st.session_state.user_input):
        figs = create_charts(st.session_state.user_input, raw_data)
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please fill inputs to see charts.")

elif page == "Insights":
    input_section()
    if validate_inputs(st.session_state.user_input):
        st.subheader("📈 Insights & Suggestions")
        st.write(f"- **Interior Space:** {st.session_state.user_input['Interior_SqFt']} sq.ft.")
        st.write(f"- **Bedrooms & Bathrooms:** {st.session_state.user_input['Num_Bedrooms']} / {st.session_state.user_input['Num_Bathrooms']}")
        st.write(f"- **Quality Scores:** Rooms {st.session_state.user_input['Quality_Score_Rooms']}, "
                 f"Bathroom {st.session_state.user_input['Quality_Score_Bathroom']}, "
                 f"Bedroom {st.session_state.user_input['Quality_Score_Bedroom']}, "
                 f"Overall {st.session_state.user_input['Quality_Score_Overall']}")
        st.write(f"- **Proximity:** {st.session_state.user_input['Distance_To_Main_Road']} meters from main road")
        st.write(f"- **Financials:** Registration Fee {st.session_state.user_input['Registration_Fee']}, "
                 f"Commission {st.session_state.user_input['Commission']}")
    else:
        st.warning("Please fill inputs.")

elif page == "Report":
    input_section()
    st.subheader("📄 Generate PDF Report")
    if validate_inputs(st.session_state.user_input):
        # Prediction
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded], columns=categorical_cols + numeric_cols_manual + numeric_cols_other + ["Building_Age"])
        prediction = model.predict(input_df)[0]

        figs = create_charts(st.session_state.user_input, raw_data)
        pdf_bytes = generate_pdf(st.session_state.user_input, figs, prediction, st.session_state.property_image)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"HouseValuation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please fill inputs before generating report.")
