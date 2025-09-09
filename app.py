import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import tempfile, os, io, base64
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
categorical_cols = ['Locality', 'Sale_Condition', 'Parking_Facility', 
                    'Building_Type', 'Utilities_Available', 'Street_Type', 'Zoning_Type']

numeric_cols_manual = ['Registration_Fee', 'Commission', 'Sale_Year', 'Build_Year', 'Interior_SqFt', 'Distance_To_Main_Road']
numeric_cols_other = ['Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms',
                      'Quality_Score_Rooms', 'Quality_Score_Bathroom', 
                      'Quality_Score_Bedroom', 'Quality_Score_Overall']

# -----------------------------
# Streamlit config
st.set_page_config(page_title="Chennai House Valuation", layout="wide")

# Hero header
st.markdown(
    """
    <div style="background-color:#0066cc;padding:15px;border-radius:8px;margin-bottom:20px">
        <h1 style="color:white;margin:0">🏡 Sunrise Property Valuation</h1>
        <p style="color:white;margin:0">AI-powered professional valuation reports for Chennai properties</p>
    </div>
    """, unsafe_allow_html=True
)

# Navigation
page = st.sidebar.radio("Navigation", ["Prediction", "Charts", "Insights", "Report"])

# -----------------------------
# Initialize user input dictionary
if "user_input" not in st.session_state:
    st.session_state.user_input = {col: None for col in categorical_cols + numeric_cols_manual + numeric_cols_other}
    for col in numeric_cols_manual + numeric_cols_other:
        st.session_state.user_input[col] = 0
    st.session_state.property_image = None

# -----------------------------
# Validation
def validate_inputs(inputs):
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return False
    return True

# -----------------------------
# Input Section
def input_section():
    st.subheader("Enter House Details")
    col1, col2 = st.columns(2)
    with col1:
        for col in categorical_cols:
            options = ["--Select--"] + list(raw_data[col].unique())
            selected = st.selectbox(col, options, index=0, key=col)
            st.session_state.user_input[col] = selected if selected != "--Select--" else None
    with col2:
        for col in numeric_cols_manual + numeric_cols_other:
            st.session_state.user_input[col] = st.number_input(col, value=st.session_state.user_input[col], key=col)

    # Building age
    if st.session_state.user_input["Build_Year"] and st.session_state.user_input["Sale_Year"]:
        st.session_state.user_input["Building_Age"] = max(st.session_state.user_input["Sale_Year"] - st.session_state.user_input["Build_Year"], 0)
    else:
        st.session_state.user_input["Building_Age"] = 0

    # Property image upload
    uploaded_file = st.file_uploader("Upload Property Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.property_image = uploaded_file
        st.image(uploaded_file, caption="Uploaded Property Image", use_column_width=True)

# -----------------------------
# Dynamic Plotly Charts
def create_plotly_charts(user_input):
    figs = {}

    # Bar chart
    feats = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
    fig_bar = px.bar(x=feats, y=[user_input[f] for f in feats], title="Key House Features")
    figs['bar'] = fig_bar

    # Radar chart
    scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[user_input[s] for s in scores] + [user_input[scores[0]]],
        theta=scores + [scores[0]],
        fill='toself',
        name='Quality'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), title="Quality Scores Radar")
    figs['radar'] = fig_radar

    # Histogram (simulate comps)
    comps = np.random.normal(loc=user_input['Interior_SqFt']*100, scale=50000, size=80)
    fig_hist = px.histogram(comps, nbins=20, title="Comparable Market Prices")
    figs['hist'] = fig_hist

    # Donut chart (cost split)
    parts = pd.DataFrame({
        "part": ["Registration Fee","Commission","Interior Value"],
        "amount": [user_input['Registration_Fee'], user_input['Commission'], user_input['Interior_SqFt']*100]
    })
    fig_pie = px.pie(parts, names="part", values="amount", hole=0.4, title="Cost Breakdown")
    figs['pie'] = fig_pie

    # Box plot (quality vs rooms)
    df_box = pd.DataFrame({
        "Rooms": [user_input['Num_Bedrooms'], user_input['Num_Bathrooms'], user_input['Total_Rooms']],
        "Quality": [user_input['Quality_Score_Rooms'], user_input['Quality_Score_Bathroom'], user_input['Quality_Score_Bedroom']]
    })
    fig_box = px.box(df_box, y="Quality", x="Rooms", points="all", title="Quality vs Rooms")
    figs['box'] = fig_box

    return figs

# -----------------------------
# PDF Generator
def generate_pdf(user_input, figs, prediction, image_file=None):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # Header
    pdf.set_fill_color(0, 102, 204)
    pdf.rect(0, 0, 210, 20, 'F')
    pdf.set_xy(0, 5)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 10, "Sunrise Property Valuation Agency", 0, 1, 'C')

    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Professional House Valuation Report", ln=True, align="C")

    # Property info table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Property Information", ln=True)
    pdf.set_font("Arial", "", 10)
    for k, v in user_input.items():
        safe_k = str(k).encode('latin-1','replace').decode('latin-1')
        safe_v = str(v).encode('latin-1','replace').decode('latin-1')
        pdf.cell(60, 6, safe_k, border=1)
        pdf.cell(0, 6, safe_v, border=1, ln=True)
    pdf.ln(4)

    # Prediction highlight
    pdf.set_fill_color(255, 204, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C")

    # Property image
    if image_file:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_img.write(image_file.read())
        tmp_img.close()
        pdf.image(tmp_img.name, x=30, w=120)
        os.unlink(tmp_img.name)
        pdf.ln(5)

    # Embed charts
    for fig_key in figs:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        figs[fig_key].write_image(tmp.name, width=600, height=400, scale=2)
        pdf.image(tmp.name, x=20, w=170)
        os.unlink(tmp.name)
        pdf.ln(5)

    # Insights
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Insights & Suggestions", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, f"- Interior: {user_input['Interior_SqFt']} sq.ft\n"
                         f"- Bedrooms/Bathrooms: {user_input['Num_Bedrooms']} / {user_input['Num_Bathrooms']}\n"
                         f"- Quality: Overall {user_input['Quality_Score_Overall']}\n"
                         f"- Distance to main road: {user_input['Distance_To_Main_Road']} meters\n"
                         f"- Registration Fee: {user_input['Registration_Fee']} INR, Commission: {user_input['Commission']} INR")

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, "Sunrise Property Valuation Agency", align="C")

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
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
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded])
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")
    else:
        st.warning("Fill all required fields to get prediction.")

elif page == "Charts":
    input_section()
    if validate_inputs(st.session_state.user_input):
        figs = create_plotly_charts(st.session_state.user_input)
        for f in figs.values():
            st.plotly_chart(f, use_container_width=True)

elif page == "Insights":
    input_section()
    st.subheader("📈 Insights & Suggestions")
    if validate_inputs(st.session_state.user_input):
        st.write(f"- Interior: {st.session_state.user_input['Interior_SqFt']} sq.ft")
        st.write(f"- Bedrooms/Bathrooms: {st.session_state.user_input['Num_Bedrooms']} / {st.session_state.user_input['Num_Bathrooms']}")
        st.write(f"- Quality Scores: Overall {st.session_state.user_input['Quality_Score_Overall']}")
        st.write(f"- Distance: {st.session_state.user_input['Distance_To_Main_Road']} meters from main road")
        st.write(f"- Fees: Registration {st.session_state.user_input['Registration_Fee']} INR, Commission {st.session_state.user_input['Commission']} INR")
    else:
        st.warning("Fill inputs to see insights.")

elif page == "Report":
    input_section()
    st.subheader("📄 Generate Professional PDF Report")
    if validate_inputs(st.session_state.user_input):
        figs = create_plotly_charts(st.session_state.user_input)
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded])
        prediction = model.predict(input_df)[0]

        pdf_bytes = generate_pdf(st.session_state.user_input, figs, prediction, st.session_state.property_image)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"HouseValuation_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Fill all inputs before generating report.")
