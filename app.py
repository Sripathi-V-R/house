# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

def validate_inputs(inputs):
    errors = []
    # Required fields
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            errors.append(f"{k} is not filled")
    # Year validations
    if inputs["Build_Year"] > inputs["Sale_Year"]:
        errors.append("Build Year cannot be after Sale Year")
    if inputs["Build_Year"] < 1800 or inputs["Build_Year"] > datetime.now().year:
        errors.append("Build Year must be between 1800 and current year")
    # Quality score
    for q in ['Quality_Score_Rooms','Quality_Score_Bathroom','Quality_Score_Bedroom','Quality_Score_Overall']:
        if not (0 <= inputs[q] <= 5):
            errors.append(f"{q} must be between 0 and 5")
    # Financial limits
    if inputs["Registration_Fee"] > 0.07 * inputs.get("Sale_Price",1):
        errors.append("Registration Fee cannot exceed 7% of Sale Price")
    if inputs["Commission"] > 0.10 * inputs.get("Sale_Price",1):
        errors.append("Commission cannot exceed 10% of Sale Price")
    # Bedroom limits
    if inputs["Building_Type"].lower() not in ["residential","house"] and inputs["Num_Bedrooms"]>0:
        errors.append("Bedrooms cannot be included for Commercial/Others")
    if inputs["Num_Bedrooms"] > 10:
        errors.append("Number of Bedrooms cannot exceed 10")
    # Distance
    if inputs["Distance_To_Main_Road"] > 1000:
        errors.append("Distance to main road cannot exceed 1000 meters")
    return len(errors)==0, errors

# -----------------------------
# Input Section
def input_section():
    st.markdown("<h2 style='text-align:center;'>🏠 House Details</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Categorical
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
        st.session_state.user_input["Building_Age"] = max(st.session_state.user_input["Sale_Year"] - st.session_state.user_input["Build_Year"],0)
    else:
        st.session_state.user_input["Building_Age"] = 0

    # Property Image
    uploaded = st.file_uploader("📸 Upload Property Image", type=["png","jpg","jpeg"])
    if uploaded is not None:
        st.session_state.property_image = uploaded
        st.image(uploaded, caption="Uploaded property image preview", use_column_width=True)

# -----------------------------
# Charts (Matplotlib for PDF)
def create_pdf_images(user_input, raw_data):
    images = []

    # Bar chart
    feats = ['Interior_SqFt','Num_Bedrooms','Num_Bathrooms','Total_Rooms']
    vals = [user_input[f] for f in feats]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(feats, vals, color='C0', edgecolor='black')
    ax.set_title("Key House Features")
    for i,v in enumerate(vals):
        ax.text(i, v+0.5, str(v), ha='center')
    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # Radar chart
    q_labels = ['Rooms','Bathroom','Bedroom','Overall']
    scores = [user_input['Quality_Score_Rooms'], user_input['Quality_Score_Bathroom'],
              user_input['Quality_Score_Bedroom'], user_input['Quality_Score_Overall']]
    angles = np.linspace(0,2*np.pi,len(scores),endpoint=False).tolist()
    scores_closed = scores + [scores[0]]
    angles_closed = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(5,5),subplot_kw=dict(polar=True))
    ax.plot(angles_closed, scores_closed, 'o-', linewidth=2)
    ax.fill(angles_closed, scores_closed, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), q_labels)
    ax.set_ylim(0,5)
    ax.set_title("Quality Radar")
    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # Interior SqFt histogram
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(raw_data['Interior_SqFt'].dropna(), bins=30, color='C1', edgecolor='black')
    ax.axvline(user_input['Interior_SqFt'], color='red', linestyle='--')
    ax.set_title("Interior SqFt Distribution")
    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # Rooms histogram
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(raw_data['Total_Rooms'].dropna(), bins=15, color='C2', edgecolor='black')
    ax.axvline(user_input['Total_Rooms'], color='red', linestyle='--')
    ax.set_title("Total Rooms Distribution")
    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    return images

# -----------------------------
# PDF Generation
def generate_pdf(user_input, pdf_images, prediction, property_image):
    pdf = FPDF('P','mm','A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(0,102,204)
    pdf.rect(0,0,210,20,'F')
    pdf.set_xy(0,5)
    pdf.set_font("Arial","B",16)
    pdf.set_text_color(255,255,255)
    pdf.cell(210,10,"Sunrise Property Valuation Agency",0,1,'C')
    pdf.set_text_color(0,0,0)
    pdf.ln(6)
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"Professional House Valuation Report",ln=True, align="C")
    pdf.ln(6)

    # Property Info
    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Property Information", ln=True)
    pdf.set_font("Arial","",10)
    for k,v in user_input.items():
        pdf.cell(60,6,safe_str(k),1)
        pdf.cell(0,6,safe_str(v),1, ln=True)
    pdf.ln(4)

    # Property image
    if property_image is not None:
        tmp_img = tempfile.NamedTemporaryFile(delete=False,suffix=".png")
        img = Image.open(property_image).convert("RGB")
        img.save(tmp_img.name,"PNG"); tmp_img.close()
        pdf.image(tmp_img.name, x=30, w=150)
        os.unlink(tmp_img.name)
        pdf.ln(6)

    # Prediction
    pdf.set_fill_color(255,204,0)
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C")
    pdf.ln(6)

    # Charts
    for buf in pdf_images:
        tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".png")
        tmp.write(buf.getbuffer()); tmp.close()
        pdf.image(tmp.name,x=15,w=180)
        os.unlink(tmp.name)
        pdf.ln(5)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial","I",9)
    pdf.cell(0,6,"Sunrise Property Valuation Agency", align="C")

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")
    pdf.output(tmp_pdf.name)
    with open(tmp_pdf.name,"rb") as f:
        pdf_bytes = f.read()
    os.unlink(tmp_pdf.name)
    return pdf_bytes

# -----------------------------
# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Prediction","Report"])

if page=="Prediction":
    input_section()
    st.subheader("💰 Predicted House Price")
    valid, errors = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning("Please correct the following:")
        for e in errors:
            st.write(f"- {e}")
    else:
        input_encoded = {}
        for col in categorical_cols:
            val = st.session_state.user_input[col]
            try:
                input_encoded[col] = label_encoders[col].transform([val])[0]
            except:
                input_encoded[col] = 0
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded])
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")

elif page=="Report":
    input_section()
    st.subheader("📄 Generate PDF Report")
    valid, errors = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning("Please correct the following:")
        for e in errors:
            st.write(f"- {e}")
    else:
        input_encoded = {}
        for col in categorical_cols:
            val = st.session_state.user_input[col]
            try:
                input_encoded[col] = label_encoders[col].transform([val])[0]
            except:
                input_encoded[col] = 0
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
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
        st.success("PDF generated successfully!")
