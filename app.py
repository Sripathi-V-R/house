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
PREPROCESSED_PATH = "preprocessed.csv"

# -----------------------------
# Load artifacts
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
raw_data = pd.read_csv(PREPROCESSED_PATH)

# -----------------------------
# Streamlit layout
st.set_page_config(page_title="Chennai House Price Prediction", layout="wide")
st.title("Chennai House Price Prediction - Valuation Report")

categorical_cols = ['Locality', 'Sale_Condition', 'Parking_Facility', 
                    'Building_Type', 'Utilities_Available', 'Street_Type', 'Zoning_Type']

manual_numeric_cols = ['Registration_Fee', 'Commission', 'Sale_Year', 'Build_Year', 'Interior_SqFt', 'Distance_To_Main_Road']

other_numeric_cols = ['Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms',
                      'Quality_Score_Rooms', 'Quality_Score_Bathroom', 
                      'Quality_Score_Bedroom', 'Quality_Score_Overall']

# -----------------------------
# Sidebar inputs
st.sidebar.header("House Details Input")
user_input = {}

# Categorical inputs
for col in categorical_cols:
    options = raw_data[col].unique().tolist()
    user_input[col] = st.sidebar.selectbox(col, options, key=col)

# Manual numeric inputs
for col in manual_numeric_cols:
    user_input[col] = st.sidebar.number_input(col, value=int(raw_data[col].median()), key=col)

# Other numeric inputs
for col in other_numeric_cols:
    user_input[col] = st.sidebar.number_input(col, value=int(raw_data[col].median()), key=col)

# Building Age
user_input['Building_Age'] = user_input['Sale_Year'] - user_input['Build_Year']

# -----------------------------
# Encode categorical
input_encoded = {col: label_encoders[col].transform([user_input[col]])[0] for col in categorical_cols}
for col in manual_numeric_cols + other_numeric_cols + ['Building_Age']:
    input_encoded[col] = user_input[col]

input_df = pd.DataFrame([input_encoded], columns=categorical_cols + manual_numeric_cols + other_numeric_cols + ['Building_Age'])

# Prediction
prediction = model.predict(input_df)[0]

# -----------------------------
# Display prediction
st.subheader("Predicted House Price")
st.success(f"{prediction:,.2f} INR")

# Display selected inputs side by side
st.subheader("Selected Inputs")
cols = st.columns(2)
for i, (k, v) in enumerate(user_input.items()):
    with cols[i % 2]:
        st.write(f"**{k}:** {v}")

# -----------------------------
# Charts
st.subheader("Property Feature Charts")

# Bar Chart
features_plot = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
values_plot = [user_input[f] for f in features_plot]

fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
ax_bar.bar(features_plot, values_plot, color='dodgerblue')
ax_bar.set_ylabel("Value")
ax_bar.set_title("Key House Features")
st.pyplot(fig_bar, clear_figure=True)

# Radar Chart
quality_scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
scores = [user_input[q] for q in quality_scores]
angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
scores += scores[:1]
angles += angles[:1]

fig_radar, ax_radar = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax_radar.plot(angles, scores, 'o-', linewidth=2, color='green')
ax_radar.fill(angles, scores, alpha=0.25, color='green')
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(quality_scores)
ax_radar.set_yticks([1, 3, 5])
ax_radar.set_yticklabels(["Low", "Medium", "High"])
st.pyplot(fig_radar, clear_figure=True)

# Optional: Pie chart for room distribution
st.subheader("Room Distribution")
rooms = [user_input['Num_Bedrooms'], user_input['Num_Bathrooms'], user_input['Total_Rooms'] - (user_input['Num_Bedrooms'] + user_input['Num_Bathrooms'])]
labels = ['Bedrooms', 'Bathrooms', 'Other Rooms']
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(rooms, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'orange', 'lightgreen'])
ax_pie.set_title("Room Distribution")
st.pyplot(fig_pie, clear_figure=True)

# -----------------------------
# PDF Report
def generate_professional_pdf(user_input, predicted_price, bar_fig, radar_fig, pie_fig):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Sunrise Property Valuation Agency", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, "123 Real Estate Avenue, Chennai", ln=True, align="C")
    pdf.cell(0, 5, "Phone: +91-98765-43210 | Email: contact@sunriseval.com", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Chennai House Valuation Report", ln=True, align="C")
    pdf.ln(2)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 5, f"Report Date: {datetime.now().strftime('%d-%m-%Y')}", ln=True)
    pdf.ln(5)

    # Property Information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Property Information", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in user_input.items():
        pdf.cell(60, 6, f"{k}", border=1)
        pdf.cell(0, 6, f"{v}", border=1, ln=True)
    pdf.ln(3)

    # Predicted Price Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, f"Predicted Market Value: {predicted_price:,.2f} INR", ln=True)
    pdf.ln(4)

    # Description / Suggestions
    pdf.set_font("Arial", "", 11)
    description = (
        "This valuation report is generated based on selected property features.\n\n"
        "The predicted price represents an estimated market value considering property size, quality, "
        "location, and sale conditions.\n\n"
        "Suggestions:\n"
        "- Ensure property condition matches the quality scores.\n"
        "- Consider local market trends and amenities when finalizing pricing.\n"
        "- Registration fee and commission are typical estimates.\n"
        "- Distance to main road can influence accessibility and valuation."
    )
    pdf.multi_cell(0, 6, description)
    pdf.ln(3)

    # Insert charts
    for fig, width in zip([bar_fig, radar_fig, pie_fig], [170, 130, 150]):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp_file.name, dpi=180, bbox_inches='tight')
        tmp_file.close()
        pdf.image(tmp_file.name, x=20, w=width)
        os.unlink(tmp_file.name)
        pdf.ln(5)

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

# Generate PDF
pdf_bytes = generate_professional_pdf(user_input, prediction, fig_bar, fig_radar, fig_pie)

# Download button
st.subheader("Download Professional Valuation Report")
st.download_button(
    label="Download PDF Report",
    data=pdf_bytes,
    file_name=f"HouseValuation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    mime="application/pdf"
)
