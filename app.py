import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import tempfile
import os

# -----------------------------
# Paths
MODEL_PATH = "catboost_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
PREPROCESSED_PATH = "preprocessed.csv"  # Raw dataset before encoding

# -----------------------------
# Load artifacts
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
raw_data = pd.read_csv(PREPROCESSED_PATH)

# -----------------------------
# Streamlit layout
st.set_page_config(page_title="Chennai House Price Prediction", layout="wide")
st.title("🏠 Chennai House Price Prediction - Valuation Report")

categorical_cols = ['Locality', 'Sale_Condition', 'Parking_Facility', 
                    'Building_Type', 'Utilities_Available', 'Street_Type', 'Zoning_Type']

numeric_cols = ['Interior_SqFt', 'Distance_To_Main_Road', 'Num_Bedrooms', 'Num_Bathrooms',
                'Total_Rooms', 'Quality_Score_Rooms', 'Quality_Score_Bathroom', 
                'Quality_Score_Bedroom', 'Quality_Score_Overall', 'Registration_Fee', 
                'Commission', 'Sale_Year', 'Build_Year']

# -----------------------------
# Sidebar inputs
st.sidebar.header("🏠 Enter House Details")
user_input = {}

# Categorical inputs
for col in categorical_cols:
    options = raw_data[col].unique().tolist()
    user_input[col] = st.sidebar.selectbox(col, options)

# Numeric inputs
for col in numeric_cols:
    default = int(raw_data[col].median())
    user_input[col] = st.sidebar.number_input(col, value=default)

# Calculate Building Age
user_input['Building_Age'] = user_input['Sale_Year'] - user_input['Build_Year']

# -----------------------------
# Encode categorical
input_encoded = {col: label_encoders[col].transform([user_input[col]])[0] for col in categorical_cols}
for col in numeric_cols + ['Building_Age']:
    input_encoded[col] = user_input.get(col, 0)

input_df = pd.DataFrame([input_encoded], columns=categorical_cols + numeric_cols + ['Building_Age'])

# Prediction
prediction = model.predict(input_df)[0]

# -----------------------------
# Display prediction
st.subheader("💰 Predicted House Price")
st.success(f"{prediction:,.2f} INR")

# Display selected inputs side by side
st.subheader("📝 Selected Inputs")
cols = st.columns(2)
for i, (k, v) in enumerate(user_input.items()):
    with cols[i % 2]:
        st.write(f"**{k}:** {v}")

# -----------------------------
# Charts
# Bar chart for key features
features_plot = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
values_plot = [user_input[f] for f in features_plot]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(features_plot, values_plot, color='skyblue')
ax.set_ylabel("Value")
ax.set_title("Key House Features")
plt.tight_layout()
st.pyplot(fig)

# Radar chart for quality scores
quality_scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
scores = [user_input[q] for q in quality_scores]

angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
scores += scores[:1]
angles += angles[:1]

fig2, ax2 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax2.plot(angles, scores, 'o-', linewidth=2)
ax2.fill(angles, scores, alpha=0.25)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(quality_scores)
ax2.set_yticks([1, 3, 5])
ax2.set_yticklabels(["Low", "Medium", "High"])
st.subheader("📊 Quality Scores Radar")
st.pyplot(fig2)

# -----------------------------
# Professional PDF report
def generate_professional_pdf(user_input, predicted_price, bar_fig, radar_fig):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Sunrise Property Valuation Agency", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 6, "123 Real Estate Avenue, Chennai", ln=True, align="C")
    pdf.cell(0, 6, "Phone: +91-98765-43210 | Email: contact@sunriseval.com", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Chennai House Valuation Report", ln=True, align="C")
    pdf.ln(2)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%d-%m-%Y')}", ln=True)
    pdf.ln(4)

    # Property details table
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Property Information", ln=True)
    pdf.set_font("Arial", "", 12)
    for k, v in user_input.items():
        pdf.cell(60, 6, f"{k}", border=1)
        pdf.cell(0, 6, f"{v}", border=1, ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Predicted Market Value: {predicted_price:,.2f} INR", ln=True)
    pdf.ln(4)

    # Insert bar chart
    tmp_bar = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    bar_fig.savefig(tmp_bar.name, dpi=180, bbox_inches='tight')
    tmp_bar.close()
    pdf.image(tmp_bar.name, x=20, w=170)
    os.unlink(tmp_bar.name)

    # Insert radar chart
    tmp_radar = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    radar_fig.savefig(tmp_radar.name, dpi=180, bbox_inches='tight')
    tmp_radar.close()
    pdf.image(tmp_radar.name, x=40, w=130)
    os.unlink(tmp_radar.name)

    # Notes and recommendations
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Notes & Recommendations", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6,
        "⚠️ This report is generated based on model predictions and property inputs provided. "
        "It is intended for reference purposes only. "
        "Actual market value may vary due to location, market conditions, and other factors."
    )

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, "Sunrise Property Valuation Agency - www.sunriseval.com", align="C")

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    temp_pdf.seek(0)
    with open(temp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
    os.unlink(temp_pdf.name)
    return pdf_bytes

# -----------------------------
pdf_bytes = generate_professional_pdf(user_input, prediction, fig, fig2)

# Download
st.subheader("📄 Download Professional Valuation Report")
st.download_button(
    label="Download PDF Report",
    data=pdf_bytes,
    file_name=f"HouseValuation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    mime="application/pdf"
)
