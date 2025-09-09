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
def validate_inputs(inputs):
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return False
    return True

def safe_str(x):
    return str(x).encode('latin-1', 'replace').decode('latin-1')

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
        # Show preview
        st.image(uploaded, caption="Uploaded property image preview", use_column_width=True)

# -----------------------------
# Plotly charts for the UI
def create_plotly_charts(user_input, raw_data):
    figs = {}

    # Bar - key features (combined)
    features = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
    vals = [user_input[f] for f in features]
    figs['bar_features'] = px.bar(x=features, y=vals, text=vals, title="Key House Features")

    # Radar - quality
    quality_scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
    scores = [user_input[q] for q in quality_scores]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=quality_scores + [quality_scores[0]], fill='toself'))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), title="Quality Radar")
    figs['radar'] = radar

    # Histogram - Interior SqFt with user's value
    hist = px.histogram(raw_data, x="Interior_SqFt", nbins=30, title="Interior SqFt distribution")
    hist.add_vline(x=user_input['Interior_SqFt'], line_dash="dash", line_color="red")
    figs['hist_interior'] = hist

    # Separate chart for Number of Rooms (histogram)
    if 'Total_Rooms' in raw_data.columns:
        hist_rooms = px.histogram(raw_data, x="Total_Rooms", nbins=15, title="Total Rooms distribution")
        hist_rooms.add_vline(x=user_input['Total_Rooms'], line_dash="dash", line_color="red")
        figs['hist_rooms'] = hist_rooms
    else:
        figs['hist_rooms'] = px.histogram(x=[user_input['Total_Rooms']], title="Total Rooms (sample)")

    # Donut - room split of the current property
    bedrooms = user_input['Num_Bedrooms'] or 0
    bathrooms = user_input['Num_Bathrooms'] or 0
    other = max(user_input['Total_Rooms'] - (bedrooms + bathrooms), 0)
    figs['donut_rooms'] = px.pie(names=['Bedrooms', 'Bathrooms', 'Other Rooms'], values=[bedrooms, bathrooms, other], hole=0.4,
                                title="Room distribution for this property")

    # Box - Registration fee vs dataset (if present)
    if 'Registration_Fee' in raw_data.columns:
        figs['box_reg'] = px.box(raw_data, y='Registration_Fee', title='Registration Fee spread (dataset)')
    else:
        figs['box_reg'] = px.box(y=[user_input['Registration_Fee']], title='Registration Fee (sample)')

    # Separate interactive chart for Interior SqFt (scatter vs price if available)
    if 'Sale_Price' in raw_data.columns:
        figs['sqft_vs_price'] = px.scatter(raw_data, x='Interior_SqFt', y='Sale_Price', trendline="ols",
                                          title="Interior SqFt vs Sale Price")
    else:
        figs['sqft_vs_price'] = px.scatter(x=[user_input['Interior_SqFt']], y=[0], title="Interior SqFt (sample)")

    # Separate interactive chart for Number of Rooms (box vs price)
    if 'Total_Rooms' in raw_data.columns and 'Sale_Price' in raw_data.columns:
        figs['rooms_vs_price'] = px.box(raw_data, x='Total_Rooms', y='Sale_Price', title="Rooms vs Price")
    else:
        figs['rooms_vs_price'] = px.box(y=[0], title="Rooms vs Price (sample)")

    return figs

# -----------------------------
# Matplotlib image generation for PDF (reliable, no kaleido)
def create_pdf_images(user_input, raw_data):
    """
    Return a list of BytesIO objects (PNG) to embed into the PDF.
    We'll produce:
      - bar features
      - radar (matplotlib)
      - interior sqft histogram (with vertical line)
      - rooms histogram (with vertical line)
      - donut (matplotlib pie)
      - box plot (registration fee)
    """
    images = []

    # 1) Bar features (matplotlib)
    feats = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
    vals = [user_input[f] for f in feats]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(feats, vals, color='C0', edgecolor='black')
    ax.set_ylabel("Value")
    ax.set_title("Key House Features")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.02 if max(vals) else v+0.5, str(v), ha='center')
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # 2) Radar (matplotlib)
    # basic radar using polar axes
    q_labels = ['Rooms','Bathroom','Bedroom','Overall']
    scores = [user_input['Quality_Score_Rooms'], user_input['Quality_Score_Bathroom'],
              user_input['Quality_Score_Bedroom'], user_input['Quality_Score_Overall']]
    angles = np.linspace(0, 2*np.pi, len(scores), endpoint=False).tolist()
    scores_closed = scores + [scores[0]]
    angles_closed = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, scores_closed, 'o-', linewidth=2)
    ax.fill(angles_closed, scores_closed, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), q_labels)
    ax.set_ylim(0,10)
    ax.set_title("Quality Radar")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # 3) Interior SqFt histogram (matplotlib)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(raw_data['Interior_SqFt'].dropna(), bins=30, color='C1', edgecolor='black')
    ax.axvline(user_input['Interior_SqFt'], color='red', linestyle='--', linewidth=2)
    ax.set_title("Interior SqFt distribution")
    ax.set_xlabel("Interior_SqFt")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # 4) Rooms histogram (matplotlib)
    fig, ax = plt.subplots(figsize=(8,4))
    if 'Total_Rooms' in raw_data.columns:
        ax.hist(raw_data['Total_Rooms'].dropna(), bins=15, color='C2', edgecolor='black')
    else:
        ax.bar([0], [user_input['Total_Rooms']])
    ax.axvline(user_input['Total_Rooms'], color='red', linestyle='--', linewidth=2)
    ax.set_title("Total Rooms distribution")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # 5) Donut (matplotlib)
    bedrooms = user_input['Num_Bedrooms'] or 0
    bathrooms = user_input['Num_Bathrooms'] or 0
    other = max(user_input['Total_Rooms'] - (bedrooms + bathrooms), 0)
    labels = ['Bedrooms', 'Bathrooms', 'Other Rooms']
    sizes = [bedrooms, bathrooms, other]
    fig, ax = plt.subplots(figsize=(6,4))
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.4), startangle=-40)
    ax.legend(wedges, labels, title="Rooms", loc="center left")
    ax.set_title("Room distribution (this property)")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    # 6) Box plot for Registration Fee (matplotlib)
    fig, ax = plt.subplots(figsize=(6,4))
    if 'Registration_Fee' in raw_data.columns:
        ax.boxplot(raw_data['Registration_Fee'].dropna(), vert=False)
    else:
        ax.boxplot([user_input['Registration_Fee']], vert=False)
    ax.set_title("Registration Fee distribution")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0); images.append(buf)

    return images

# -----------------------------
# PDF Generation (uses matplotlib PNGs + uploaded image)
def generate_pdf(user_input, pdf_image_buffers, prediction, property_image):
    # pdf_image_buffers: list of BytesIO PNGs created by create_pdf_images
    pdf = FPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(0, 102, 204)
    pdf.rect(0, 0, 210, 20, 'F')
    pdf.set_xy(0, 5)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 10, "Sunrise Property Valuation Agency", 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Professional House Valuation Report", ln=True, align="C")
    pdf.ln(6)

    # Property table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Property Information", ln=True)
    pdf.set_font("Arial", "", 10)
    for k, v in user_input.items():
        pdf.cell(60, 6, safe_str(k), border=1)
        pdf.cell(0, 6, safe_str(v), border=1, ln=True)
    pdf.ln(4)

    # Property image - convert and embed
    if property_image is not None:
        try:
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img = Image.open(property_image).convert("RGB")
            img.save(tmp_img.name, "PNG")
            tmp_img.close()
            # center image
            pdf.image(tmp_img.name, x=30, w=150)
            os.unlink(tmp_img.name)
            pdf.ln(6)
        except Exception as e:
            # if conversion fails, skip but continue
            print("Image embedding failed:", e)

    # Prediction highlight
    pdf.set_fill_color(255, 204, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C")
    pdf.ln(6)

    # Embed matplotlib-generated PNGs
    for buf in pdf_image_buffers:
        # write buffer to temp file so fpdf can include
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(buf.getbuffer())
        tmp.close()
        try:
            pdf.image(tmp.name, x=15, w=180)
        except Exception as e:
            print("Failed to add chart to PDF:", e)
        os.unlink(tmp.name)
        pdf.ln(5)

    # Insights
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Insights & Suggestions", ln=True)
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

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 6, "Sunrise Property Valuation Agency", align="C")

    # Return bytes
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    with open(tmp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
    os.unlink(tmp_pdf.name)
    return pdf_bytes

# -----------------------------
# Pages / Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Prediction", "Charts", "Insights", "Report"])

if page == "Prediction":
    input_section()
    st.subheader("💰 Predicted House Price")
    if validate_inputs(st.session_state.user_input):
        # encode categoricals safely (assumes encoder exists)
        input_encoded = {}
        for col in categorical_cols:
            # if user selected invalid category, handle gracefully
            val = st.session_state.user_input[col]
            try:
                input_encoded[col] = label_encoders[col].transform([val])[0]
            except Exception:
                # fallback: try to find mapping or set zero
                input_encoded[col] = 0
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input.get(col, 0)
        input_df = pd.DataFrame([input_encoded], columns=list(input_encoded.keys()))
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")
    else:
        st.warning("Please complete all inputs.")

elif page == "Charts":
    input_section()
    st.subheader("📊 Interactive Charts")
    if validate_inputs(st.session_state.user_input):
        figs = create_plotly_charts(st.session_state.user_input, raw_data)
        # Show charts in a 2-column grid
        keys = list(figs.keys())
        for i in range(0, len(keys), 2):
            cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx < len(keys):
                    cols[j].plotly_chart(figs[keys[idx]], use_container_width=True)
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
        # show quick charts inline
        figs = create_plotly_charts(st.session_state.user_input, raw_data)
        st.plotly_chart(figs['bar_features'], use_container_width=True)
        st.plotly_chart(figs['donut_rooms'], use_container_width=True)
    else:
        st.warning("Please fill all inputs.")

elif page == "Report":
    input_section()
    st.subheader("📄 Generate PDF Report")
    if validate_inputs(st.session_state.user_input):
        # prepare prediction
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

        # create matplotlib images for PDF (reliable)
        pdf_images = create_pdf_images(st.session_state.user_input, raw_data)

        # generate pdf bytes (includes uploaded image if provided)
        pdf_bytes = generate_pdf(st.session_state.user_input, pdf_images, prediction, st.session_state.property_image)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"HouseValuation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

        st.success("PDF generated — click the download button above.")
    else:
        st.warning("Please fill all inputs before generating report.")
