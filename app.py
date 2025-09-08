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
# Columns
categorical_cols = ['Locality', 'Sale_Condition', 'Parking_Facility', 
                    'Building_Type', 'Utilities_Available', 'Street_Type', 'Zoning_Type']

numeric_cols_manual = ['Registration_Fee', 'Commission', 'Sale_Year', 'Build_Year', 'Interior_SqFt', 'Distance_To_Main_Road']
numeric_cols_other = ['Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms',
                      'Quality_Score_Rooms', 'Quality_Score_Bathroom', 
                      'Quality_Score_Bedroom', 'Quality_Score_Overall']

# -----------------------------
# Streamlit page selection
st.set_page_config(page_title="Chennai House Valuation", layout="wide")
page = st.sidebar.radio("Navigation", ["Prediction", "Charts", "Insights", "Report"])

# -----------------------------
# Initialize user input dictionary
if "user_input" not in st.session_state:
    st.session_state.user_input = {col: None for col in categorical_cols + numeric_cols_manual + numeric_cols_other}
    for col in numeric_cols_manual + numeric_cols_other:
        st.session_state.user_input[col] = 0

# -----------------------------
# Function to validate input
def validate_inputs(inputs):
    for k, v in inputs.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return False
    return True

# -----------------------------
# Input Section
def input_section():
    st.header("🏠 Enter House Details")
    for col in categorical_cols:
        options = ["--Select--"] + list(raw_data[col].unique())
        selected = st.selectbox(col, options, index=0, key=col)
        st.session_state.user_input[col] = selected if selected != "--Select--" else None
    
    for col in numeric_cols_manual + numeric_cols_other:
        st.session_state.user_input[col] = st.number_input(col, value=st.session_state.user_input[col], key=col)

    # Auto-calculate building age
    if st.session_state.user_input["Build_Year"] and st.session_state.user_input["Sale_Year"]:
        st.session_state.user_input["Building_Age"] = max(st.session_state.user_input["Sale_Year"] - st.session_state.user_input["Build_Year"], 0)
    else:
        st.session_state.user_input["Building_Age"] = 0

# -----------------------------
# Function to create charts dynamically
def create_charts(user_input):
    # Bar Chart
    features_plot = ['Interior_SqFt', 'Num_Bedrooms', 'Num_Bathrooms', 'Total_Rooms']
    values_plot = [user_input[f] for f in features_plot]

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    ax_bar.bar(features_plot, values_plot, color='#1f77b4', edgecolor='black')
    ax_bar.set_ylabel("Value")
    ax_bar.set_title("Key House Features")
    plt.tight_layout()

    # Radar Chart
    quality_scores = ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']
    scores = [user_input[q] for q in quality_scores]
    angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, scores, 'o-', linewidth=2, color='#ff7f0e')
    ax_radar.fill(angles, scores, alpha=0.25, color='#ff7f0e')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(quality_scores)
    ax_radar.set_yticks([1, 3, 5])
    ax_radar.set_yticklabels(["Low", "Medium", "High"])
    plt.tight_layout()
    return fig_bar, fig_radar

# -----------------------------
# Page: Prediction
if page == "Prediction":
    input_section()
    st.subheader("💰 Predicted House Price")
    if validate_inputs(st.session_state.user_input):
        # Encode categorical
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded], columns=categorical_cols + numeric_cols_manual + numeric_cols_other + ["Building_Age"])
        prediction = model.predict(input_df)[0]
        st.success(f"{prediction:,.2f} INR")
    else:
        st.warning("Please select/fill all required fields above to see prediction.")

# -----------------------------
# Page: Charts
if page == "Charts":
    input_section()
    if validate_inputs(st.session_state.user_input):
        fig_bar, fig_radar = create_charts(st.session_state.user_input)
        st.pyplot(fig_bar, clear_figure=True)
        st.pyplot(fig_radar, clear_figure=True)
    else:
        st.warning("Please fill all inputs to see charts.")

# -----------------------------
# Page: Insights
if page == "Insights":
    input_section()
    st.subheader("📈 Insights & Suggestions")
    if validate_inputs(st.session_state.user_input):
        st.markdown("### 🔍 Key Insights")
        st.write(f"- **Interior Space:** {st.session_state.user_input['Interior_SqFt']} sq.ft, consider increasing usable area to improve valuation.")
        st.write(f"- **Bedrooms & Bathrooms:** {st.session_state.user_input['Num_Bedrooms']} bedrooms, {st.session_state.user_input['Num_Bathrooms']} bathrooms. Balanced number improves resale value.")
        st.write(f"- **Quality Scores:**")
        for score in ['Quality_Score_Rooms', 'Quality_Score_Bathroom', 'Quality_Score_Bedroom', 'Quality_Score_Overall']:
            st.write(f"  - {score}: {st.session_state.user_input[score]}")
        st.write(f"- **Proximity:** {st.session_state.user_input['Distance_To_Main_Road']} meters from main road; closer improves marketability.")
        st.write(f"- **Financials:** Registration Fee {st.session_state.user_input['Registration_Fee']}, Commission {st.session_state.user_input['Commission']}")
    else:
        st.warning("Please fill all inputs to see insights.")

# -----------------------------
# Page: Report
if page == "Report":
    input_section()
    st.subheader("📄 Generate Professional PDF Report")
    if validate_inputs(st.session_state.user_input):
        fig_bar, fig_radar = create_charts(st.session_state.user_input)

        def generate_pdf(user_input, bar_fig, radar_fig, prediction):
            pdf = FPDF('P', 'mm', 'A4')
            pdf.add_page()
            # Header
            pdf.set_fill_color(0, 102, 204)  # Blue header
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
            
            # Property info
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Property Information", ln=True)
            pdf.set_font("Arial", "", 11)
            for k, v in user_input.items():
                pdf.cell(60, 6, f"{k}", border=1)
                pdf.cell(0, 6, f"{v}", border=1, ln=True)
            pdf.ln(5)
            
            # Prediction Box
            pdf.set_fill_color(255, 204, 0)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"💰 Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C")
            pdf.ln(5)
            
            # Charts
            for fig, width in zip([bar_fig, radar_fig], [170, 130]):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp_file.name, dpi=180, bbox_inches='tight')
                tmp_file.close()
                pdf.image(tmp_file.name, x=20, w=width)
                os.unlink(tmp_file.name)
                pdf.ln(5)
            
            # Insights Section
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Insights & Suggestions", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, f"- Interior Space: {user_input['Interior_SqFt']} sq.ft. Consider increasing usable area.\n"
                                  f"- Bedrooms/Bathrooms: {user_input['Num_Bedrooms']} / {user_input['Num_Bathrooms']}. Balanced number improves resale value.\n"
                                  f"- Quality Scores: Rooms {user_input['Quality_Score_Rooms']}, Bathroom {user_input['Quality_Score_Bathroom']}, Bedroom {user_input['Quality_Score_Bedroom']}, Overall {user_input['Quality_Score_Overall']}.\n"
                                  f"- Distance from Main Road: {user_input['Distance_To_Main_Road']} meters.\n"
                                  f"- Registration Fee: {user_input['Registration_Fee']}, Commission: {user_input['Commission']}.")

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

        # Generate and download PDF
        input_encoded = {col: label_encoders[col].transform([st.session_state.user_input[col]])[0] for col in categorical_cols}
        for col in numeric_cols_manual + numeric_cols_other + ["Building_Age"]:
            input_encoded[col] = st.session_state.user_input[col]
        input_df = pd.DataFrame([input_encoded], columns=categorical_cols + numeric_cols_manual + numeric_cols_other + ["Building_Age"])
        prediction = model.predict(input_df)[0]

        pdf_bytes = generate_pdf(st.session_state.user_input, fig_bar, fig_radar, prediction)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"HouseValuation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please fill all inputs before generating report.")
