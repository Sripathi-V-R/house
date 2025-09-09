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
# Paths (change if needed)
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
# Enhanced validation
def validate_inputs(inputs):
    errors = []

    # Required fields
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            errors.append(f"Missing value: {k}")

    # Building year checks
    build_year = inputs.get("Build_Year", 0)
    sale_year = inputs.get("Sale_Year", 0)
    current_year = datetime.now().year
    if build_year > sale_year:
        errors.append("Build_Year cannot be after Sale_Year")
    if build_year < 1800 or build_year > current_year:
        errors.append(f"Build_Year must be between 1800 and {current_year}")

    # Quality scores: 0-5
    for q in ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']:
        val = inputs.get(q, 0)
        if val < 0 or val > 5:
            errors.append(f"{q} must be between 0 and 5")

    # Registration Fee <= 7% of sale price (if sale price entered)
    sale_price = inputs.get("Sale_Price", 0)
    reg_fee = inputs.get("Registration_Fee", 0)
    if sale_price > 0 and reg_fee > 0.07 * sale_price:
        errors.append("Registration_Fee cannot exceed 7% of Sale Price")

    # Commission <= 10%
    commission = inputs.get("Commission", 0)
    if sale_price > 0 and commission > 0.10 * sale_price:
        errors.append("Commission cannot exceed 10% of Sale Price")

    # Bedrooms restrictions
    building_type = inputs.get("Building_Type", "").lower() if inputs.get("Building_Type") else ""
    bedrooms = inputs.get("Num_Bedrooms", 0)
    if building_type in ["commercial", "others"] and bedrooms > 0:
        errors.append("Number of bedrooms should not be included for commercial/other buildings")
    if bedrooms > 10:
        errors.append("Number of bedrooms cannot exceed 10")

    # Distance
    distance = inputs.get("Distance_To_Main_Road", 0)
    if distance > 1000:
        errors.append("Distance_To_Main_Road cannot exceed 1000 meters")

    if errors:
        return False, errors
    return True, []

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
# Plotly charts for UI
def create_plotly_charts(user_input, raw_data):
    figs = {}

    # Bar features
    features = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
    vals = [user_input[f] for f in features]
    figs['bar_features'] = px.bar(x=features, y=vals, text=vals, title="Key House Features")

    # Radar
    quality_scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
    scores = [user_input[q] for q in quality_scores]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=quality_scores + [quality_scores[0]], fill='toself'))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), title="Quality Radar")
    figs['radar'] = radar

    # Histograms
    hist_interior = px.histogram(raw_data, x="Interior_SqFt", nbins=30, title="Interior SqFt distribution")
    hist_interior.add_vline(x=user_input['Interior_SqFt'], line_dash="dash", line_color="red")
    figs['hist_interior'] = hist_interior

    hist_rooms = px.histogram(raw_data, x="Total_Rooms", nbins=15, title="Total Rooms distribution")
    hist_rooms.add_vline(x=user_input['Total_Rooms'], line_dash="dash", line_color="red")
    figs['hist_rooms'] = hist_rooms

    # Donut rooms
    bedrooms = user_input['Num_Bedrooms'] or 0
    bathrooms = user_input['Num_Bathrooms'] or 0
    other = max(user_input['Total_Rooms'] - (bedrooms + bathrooms), 0)
    figs['donut_rooms'] = px.pie(names=['Bedrooms', 'Bathrooms', 'Other Rooms'],
                                 values=[bedrooms, bathrooms, other], hole=0.4, title="Room distribution")

    return figs

# -----------------------------
# Matplotlib images for PDF
def create_pdf_images(user_input, raw_data):
    images = []

    # Bar features
    feats = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
    vals = [user_input[f] for f in feats]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(feats, vals, color='C0', edgecolor='black')
    ax.set_ylabel("Value"); ax.set_title("Key House Features")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.02 if max(vals) else v+0.5, str(v), ha='center')
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # Radar
    q_labels = ['Rooms','Bathroom','Bedroom','Overall']
    scores = [user_input['Quality_Score_Rooms'], user_input['Quality_Score_Bathroom'],
              user_input['Quality_Score_Bedroom'], user_input['Quality_Score_Overall']]
    angles = np.linspace(0, 2*np.pi, len(scores), endpoint=False).tolist()
    scores_closed = scores + [scores[0]]
    angles_closed = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, scores_closed, 'o-', linewidth=2)
    ax.fill(angles_closed, scores_closed, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), q_labels); ax.set_ylim(0,5)
    ax.set_title("Quality Radar")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # Interior SqFt histogram
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(raw_data['Interior_SqFt'].dropna(), bins=30, color='C1', edgecolor='black')
    ax.axvline(user_input['Interior_SqFt'], color='red', linestyle='--', linewidth=2)
    ax.set_title("Interior SqFt distribution"); ax.set_xlabel("Interior_SqFt")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # Rooms histogram
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(raw_data['Total_Rooms'].dropna(), bins=15, color='C2', edgecolor='black')
    ax.axvline(user_input['Total_Rooms'], color='red', linestyle='--', linewidth=2)
    ax.set_title("Total Rooms distribution")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # Donut
    bedrooms = user_input['Num_Bedrooms'] or 0
    bathrooms = user_input['Num_Bathrooms'] or 0
    other = max(user_input['Total_Rooms'] - (bedrooms + bathrooms), 0)
    labels = ['Bedrooms', 'Bathrooms', 'Other Rooms']
    sizes = [bedrooms, bathrooms, other]
    fig, ax = plt.subplots(figsize=(6,4))
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.4), startangle=-40)
    ax.legend(wedges, labels, title="Rooms", loc="center left"); ax.set_title("Room distribution (this property)")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    return images

# -----------------------------
# PDF Generation
def generate_pdf(user_input, pdf_image_buffers, prediction, property_image):
    pdf = FPDF('P', 'mm', 'A4'); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()

    # Header
    pdf.set_fill_color(0, 102, 204); pdf.rect(0, 0, 210, 20, 'F')
    pdf.set_xy(0, 5); pdf.set_font("Arial", "B", 16); pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 10, "Sunrise Property Valuation Agency", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0); pdf.ln(6); pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Professional House Valuation Report", ln=True, align="C"); pdf.ln(6)

    # Property table
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, "Property Information", ln=True)
    pdf.set_font("Arial", "", 10)
    for k, v in user_input.items():
        pdf.cell(60, 6, safe_str(k), border=1)
        pdf.cell(0, 6, safe_str(v), border=1, ln=True)
    pdf.ln(4)

    # Property image
    if property_image is not None:
        try:
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img = Image.open(property_image).convert("RGB"); img.save(tmp_img.name, "PNG"); tmp_img.close()
            pdf.image(tmp_img.name, x=30, w=150); os.unlink(tmp_img.name); pdf.ln(6)
        except Exception as e:
            print("Image embedding failed:", e)

    # Prediction
    pdf.set_fill_color(255, 204, 0); pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C"); pdf.ln(6)

    # Charts
    for buf in pdf_image_buffers:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png"); tmp.write(buf.getbuffer()); tmp.close()
        try: pdf.image(tmp.name, x=15, w=180)
        except Exception as e: print("Failed to add chart to PDF:", e)
        os.unlink(tmp.name); pdf.ln(5)

    # Footer insights
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, "Insights & Suggestions", ln=True)
    pdf.set_font("Arial", "", 10)
    insights_text = (
        f"- Interior Space: {user_input['Interior_SqFt']} sq.ft.\n"
        f"- Bedrooms/Bathrooms: {user_input['Num_Bedrooms']} / {user_input['Num_Bathrooms']}\n"
        f"- Quality Scores: Rooms {user_input['Quality_Score_Rooms']}, Bathroom {user_input['Quality_Score_Bathroom']}, "
        f"Bedroom {user_input['Quality_Score_Bedroom']}, Overall {user_input['Quality_Score_Overall']}\n"
        f"- Distance from Main Road: {user_input['Distance_To_Main_Road']} meters\n"
        f"- Registration Fee: {user_input['Registration_Fee']}, Commission: {user_input['Commission']}\n"
    )
    pdf.multi_cell(0, 6, safe_str(insights_text))
    pdf.set_y(-20); pdf.set_font("Arial", "I", 9); pdf.cell(0, 6, "Sunrise Property Valuation Agency", align="C")

    # Return PDF bytes
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf"); pdf.output(tmp_pdf.name)
    with open(tmp_pdf.name, "rb") as f: pdf_bytes = f.read()
    os.unlink(tmp_pdf.name); return pdf_bytes

# -----------------------------
# Streamlit Pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Prediction", "Charts", "Insights", "Report"])

if page == "Prediction":
    input_section()
    st.subheader("💰 Predicted House Price")
    valid, errors = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning("Please correct the following errors:")
        for e in errors: st.write(f"- {e}")
    else:
        input_encoded = {}
        for col in categorical_cols:
            val = st.session_state.user_input[col]
            try: input_encoded[col] = label_encoders[col].transform([val])[0]
            except: input_encoded[col] = 0
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input.get(col, 0)
        input_df = pd.DataFrame([input_encoded], columns=list(input_encoded.keys()))
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")

elif page == "Charts":
    input_section()
    st.subheader("📊 Interactive Charts")
    valid, errors = validate_inputs(st.session_state.user_input)
    if not valid:
        st.warning("Please correct the following errors:")
        for e in errors: st.write(f"- {e}")
    else:
        figs = create_plotly_charts(st.session_state.user_input, raw_data)
        keys = list(figs.keys())
        for i in range(0, len(keys), 2):
            cols = st.columns(2)
            for j in range(2
