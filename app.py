import streamlit as st
import pandas as pd
import numpy as np
import tempfile, os
from fpdf import FPDF
import matplotlib.pyplot as plt

# -------------------------------
# Safe string for PDF
# -------------------------------
def safe_str(s):
    return str(s).encode("latin-1", "replace").decode("latin-1")

# -------------------------------
# Your original PDF generator (kept same style)
# -------------------------------
def generate_pdf(user_input, figs, prediction):
    pdf = FPDF("P", "mm", "A4")
    pdf.add_page()

    # Header
    pdf.set_fill_color(0, 102, 204)
    pdf.rect(0, 0, 210, 20, "F")
    pdf.set_xy(0, 5)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 10, "Sunrise Property Valuation Agency", 0, 1, "C")

    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Professional House Valuation Report", ln=True, align="C")
    pdf.ln(5)

    # Property Information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Property Information", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in user_input.items():
        pdf.cell(60, 6, safe_str(k), border=1)
        pdf.cell(0, 6, safe_str(v), border=1, ln=True)
    pdf.ln(5)

    # Prediction
    pdf.set_fill_color(255, 204, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C")
    pdf.ln(5)

    # Charts
    for fig, width in figs:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp_file.name, dpi=180, bbox_inches="tight")
        tmp_file.close()
        pdf.image(tmp_file.name, x=20, w=width)
        os.unlink(tmp_file.name)
        pdf.ln(5)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 6, "Sunrise Property Valuation Agency", align="C")

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    with open(tmp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
    os.unlink(tmp_pdf.name)
    return pdf_bytes


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Sunrise Property Valuation", layout="wide")

st.title("🏠 Sunrise Property Valuation")

# Sidebar inputs with defaults: 0 for numbers, NaN for text
st.sidebar.header("Enter Property Details")

user_input = {
    "Num_Bedrooms": st.sidebar.number_input("Num_Bedrooms", min_value=0, max_value=10, value=0),
    "Num_Bathrooms": st.sidebar.number_input("Num_Bathrooms", min_value=0, max_value=10, value=0),
    "Interior_SqFt": st.sidebar.number_input("Interior_SqFt", min_value=0, max_value=20000, value=0),
    "Quality_Score_Rooms": st.sidebar.number_input("Quality_Score_Rooms", min_value=0, max_value=10, value=0),
    "Quality_Score_Bathroom": st.sidebar.number_input("Quality_Score_Bathroom", min_value=0, max_value=10, value=0),
    "Quality_Score_Bedroom": st.sidebar.number_input("Quality_Score_Bedroom", min_value=0, max_value=10, value=0),
    "Quality_Score_Overall": st.sidebar.number_input("Quality_Score_Overall", min_value=0, max_value=10, value=0),
    "Distance_To_Main_Road": st.sidebar.number_input("Distance_To_Main_Road (meters)", min_value=0, max_value=2000, value=0),
    "Registration_Fee": st.sidebar.number_input("Registration_Fee (INR)", min_value=0, max_value=2000000, value=0),
    "Commission": st.sidebar.number_input("Commission (INR)", min_value=0, max_value=2000000, value=0),
    "City": st.sidebar.text_input("City", value=np.nan),
    "Locality": st.sidebar.text_input("Locality", value=np.nan),
    "Year_Built": st.sidebar.number_input("Year_Built", min_value=1900, max_value=2025, value=0),
}

# Simple prediction logic (same working style)
prediction = user_input["Interior_SqFt"] * 3500 - (user_input["Registration_Fee"] + user_input["Commission"])

# Charts
figs = []

# Bar chart (area vs quality vs fees)
fig1, ax1 = plt.subplots()
ax1.bar(["Area Value", "Fees"], [user_input["Interior_SqFt"]*3500, -(user_input["Registration_Fee"]+user_input["Commission"])])
ax1.set_title("Value Breakdown")
figs.append((fig1, 170))

# Radar chart (quality)
labels = ["Rooms", "Bathroom", "Bedroom", "Overall"]
scores = [
    user_input["Quality_Score_Rooms"],
    user_input["Quality_Score_Bathroom"],
    user_input["Quality_Score_Bedroom"],
    user_input["Quality_Score_Overall"],
]
fig2, ax2 = plt.subplots(subplot_kw={"polar": True})
theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
scores += scores[:1]
theta += theta[:1]
ax2.plot(theta, scores, "o-", linewidth=2)
ax2.fill(theta, scores, alpha=0.25)
ax2.set_xticks(theta[:-1])
ax2.set_xticklabels(labels)
ax2.set_title("Quality Scores")
figs.append((fig2, 130))

# Histogram (interior space distribution simulation)
fig3, ax3 = plt.subplots()
sample_sizes = np.random.normal(loc=user_input["Interior_SqFt"], scale=200, size=100)
ax3.hist(sample_sizes, bins=20, color="gray")
ax3.set_title("Comparable Interior Sizes")
figs.append((fig3, 170))

st.subheader("📊 Charts")
for f, _ in figs:
    st.pyplot(f)

st.subheader("💰 Predicted Price")
st.success(f"{prediction:,.2f} INR")

# PDF download
if st.button("Generate PDF Report"):
    pdf_bytes = generate_pdf(user_input, figs, prediction)
    st.download_button("Download PDF", data=pdf_bytes, file_name="valuation_report.pdf", mime="application/pdf")
