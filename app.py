# app_streamlit_improved.py
import streamlit as st
import pandas as pd
import numpy as np
import tempfile, os, io, math
from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Helper: ASCII-safe conversion
# ---------------------------
def safe_str(s):
    # convert to string and ensure latin-1 fallback to avoid unicode errors in PDF
    return str(s).encode('latin-1', 'replace').decode('latin-1')

# ---------------------------
# Improved PDF generator (keeps ASCII-safe fonts)
# ---------------------------
def generate_pdf(user_input, figs, prediction, logo_url=None, property_image_url=None):
    """
    user_input: dict of property values
    figs: list of tuples (pil_or_path, width_mm) OR (bytes_io, width_mm) for images to embed
    prediction: numeric
    logo_url: optional URL (we will download temporarily)
    property_image_url: optional url to include photo
    returns: bytes of PDF
    """
    pdf = FPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header bar
    pdf.set_fill_color(0, 102, 204)  # blue
    pdf.rect(0, 0, 210, 20, 'F')
    pdf.set_xy(5, 4)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Sunrise Property Valuation Agency", ln=1)
    pdf.set_text_color(0, 0, 0)
    
    # Optional logo (download temporarily)
    try:
        if logo_url:
            import requests
            r = requests.get(logo_url, timeout=6)
            if r.status_code == 200:
                tmp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp_logo.write(r.content)
                tmp_logo.close()
                pdf.image(tmp_logo.name, x=160, y=4, w=35)
                os.unlink(tmp_logo.name)
    except Exception:
        pass  # no network or failed -> continue safely
    
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Professional House Valuation Report", ln=True, align='C')
    pdf.ln(4)
    
    # Property basic table
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, "Property Information", ln=True)
    pdf.set_font("Arial", "", 10)
    # Create two-column table
    for k, v in user_input.items():
        safe_k = safe_str(k)
        safe_v = safe_str(v)
        pdf.cell(60, 6, safe_k, border=1)
        pdf.cell(0, 6, safe_v, border=1, ln=True)
    pdf.ln(3)
    
    # Prediction block
    pdf.set_fill_color(255, 204, 0)  # yellow
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align='C')
    pdf.ln(4)
    
    # Insert property image if provided
    try:
        if property_image_url:
            import requests
            r = requests.get(property_image_url, timeout=6)
            if r.status_code == 200:
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp_img.write(r.content)
                tmp_img.close()
                pdf.image(tmp_img.name, x=20, w=80)
                os.unlink(tmp_img.name)
    except Exception:
        pass
    
    # Embed charts (figs is list of (bytes_io_or_path, width_mm))
    for idx, (img_obj, width_mm) in enumerate(figs):
        # img_obj can be path or BytesIO or PIL Image
        if hasattr(img_obj, "read"):
            # BytesIO-like
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_obj.getbuffer())
            tmp.close()
            pdf.image(tmp.name, x=20, w=width_mm)
            os.unlink(tmp.name)
        elif isinstance(img_obj, str) and os.path.exists(img_obj):
            pdf.image(img_obj, x=20, w=width_mm)
        else:
            # try if PIL image
            try:
                if hasattr(img_obj, "save"):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img_obj.save(tmp.name, format="PNG")
                    tmp.close()
                    pdf.image(tmp.name, x=20, w=width_mm)
                    os.unlink(tmp.name)
            except Exception:
                continue
        pdf.ln(4)
    
    # Insights
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Insights & Suggestions", ln=True)
    pdf.set_font("Arial", "", 10)
    # Build the insights text safely
    insp_lines = []
    try:
        insp_lines.append(f"- Interior Space: {user_input.get('Interior_SqFt','N/A')} sq.ft.")
        insp_lines.append(f"- Bedrooms/Bathrooms: {user_input.get('Num_Bedrooms','N/A')} / {user_input.get('Num_Bathrooms','N/A')}.")
        qrooms = user_input.get('Quality_Score_Rooms','N/A')
        qbath = user_input.get('Quality_Score_Bathroom','N/A')
        qbed = user_input.get('Quality_Score_Bedroom','N/A')
        qoverall = user_input.get('Quality_Score_Overall','N/A')
        insp_lines.append(f"- Quality Scores: Rooms {qrooms}, Bathroom {qbath}, Bedroom {qbed}, Overall {qoverall}.")
        insp_lines.append(f"- Distance from Main Road: {user_input.get('Distance_To_Main_Road','N/A')} meters.")
        insp_lines.append(f"- Registration Fee: {user_input.get('Registration_Fee','N/A')}, Commission: {user_input.get('Commission','N/A')}.")
    except Exception:
        pass
    pdf.multi_cell(0, 6, "\n".join([safe_str(l) for l in insp_lines]))
    
    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 6, "Sunrise Property Valuation Agency - Generated report", align="C")
    
    # Return bytes
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    with open(tmp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
    os.unlink(tmp_pdf.name)
    return pdf_bytes

# ---------------------------
# Small utility to convert plotly fig to bytes (PNG)
# ---------------------------
def fig_to_bytes(fig, width=800, height=450, scale=2):
    # returns io.BytesIO PNG
    import io
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    return io.BytesIO(img_bytes)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Sunrise - Property Valuation", layout="wide", initial_sidebar_state="expanded")

# Top hero header (site-like)
with st.container():
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("<h1 style='margin:0; padding:0'>Sunrise Property Valuation Agency</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:gray; margin-top:0'>Professional house valuations — interactive, fast and clear</p>", unsafe_allow_html=True)
    with col2:
        # small logo using URL (replace with your URL)
        logo_url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/House_%28simple_icon%29.svg"
        st.image(logo_url, width=90)

st.markdown("---")

# Sidebar inputs (keeps your working style)
st.sidebar.header("Property inputs")
# Provide defaults for interactivity
Num_Bedrooms = st.sidebar.number_input("Num_Bedrooms", min_value=0, max_value=10, value=3)
Num_Bathrooms = st.sidebar.number_input("Num_Bathrooms", min_value=0, max_value=10, value=2)
Interior_SqFt = st.sidebar.number_input("Interior_SqFt", min_value=100, max_value=20000, value=1200)
Quality_Score_Rooms = st.sidebar.slider("Quality_Score_Rooms", 0, 10, 7)
Quality_Score_Bathroom = st.sidebar.slider("Quality_Score_Bathroom", 0, 10, 6)
Quality_Score_Bedroom = st.sidebar.slider("Quality_Score_Bedroom", 0, 10, 7)
Quality_Score_Overall = st.sidebar.slider("Quality_Score_Overall", 0, 10, 7)
Distance_To_Main_Road = st.sidebar.number_input("Distance_To_Main_Road (meters)", min_value=0, max_value=2000, value=150)
Registration_Fee = st.sidebar.number_input("Registration_Fee (INR)", min_value=0, max_value=2000000, value=15000)
Commission = st.sidebar.number_input("Commission (INR)", min_value=0, max_value=2000000, value=50000)
City = st.sidebar.text_input("City", value="Mumbai")
Locality = st.sidebar.text_input("Locality", value="Andheri")
Year_Built = st.sidebar.number_input("Year_Built", min_value=1900, max_value=2025, value=2005)

# Simple model-like prediction (keeps behaviour similar to your original)
base_rate = 3500  # INR per sq ft as a baseline
quality_factor = (Quality_Score_Overall / 7.0)  # simple factor
age_penalty = max(0, (2025 - Year_Built) * 5)  # rupees per sq ft penalty
predicted_price = (Interior_SqFt * (base_rate + quality_factor*200 - age_penalty)) - (Registration_Fee + Commission)

# Build user_input dict (same keys you used)
user_input = {
    "Num_Bedrooms": Num_Bedrooms,
    "Num_Bathrooms": Num_Bathrooms,
    "Interior_SqFt": Interior_SqFt,
    "Quality_Score_Rooms": Quality_Score_Rooms,
    "Quality_Score_Bathroom": Quality_Score_Bathroom,
    "Quality_Score_Bedroom": Quality_Score_Bedroom,
    "Quality_Score_Overall": Quality_Score_Overall,
    "Distance_To_Main_Road": Distance_To_Main_Road,
    "Registration_Fee": Registration_Fee,
    "Commission": Commission,
    "City": City,
    "Locality": Locality,
    "Year_Built": Year_Built
}

# Main page layout
left_col, right_col = st.columns([2,1])

with left_col:
    st.subheader("Property summary")
    st.markdown(f"**Location:** {Locality}, {City}")
    st.markdown(f"**Built year:** {Year_Built}")
    st.markdown(f"**Interior area:** {Interior_SqFt} sq.ft.")
    st.markdown(f"**Bedrooms / Bathrooms:** {Num_Bedrooms} / {Num_Bathrooms}")
    st.markdown(f"**Predicted Price:** **{predicted_price:,.2f} INR**")
    
    # Property image (sample link, swap for your preferred hosted picture)
    property_image_url = "https://cdn.pixabay.com/photo/2016/11/29/09/32/house-1867187_1280.jpg"
    st.image(property_image_url, caption="Property photo (preview)", use_column_width=True)
    
    st.markdown("### Dynamic charts")
    # 1) Bar chart: component value contributions
    components = pd.DataFrame({
        "component": ["Area", "Quality", "Age_penalty", "Fees"],
        "value": [
            Interior_SqFt * base_rate,
            Interior_SqFt * (quality_factor*200),
            -Interior_SqFt * age_penalty,
            -(Registration_Fee + Commission)
        ]
    })
    fig_bar = px.bar(components, x='component', y='value', title="Price component breakdown", text='value')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 2) Radar chart: quality distribution
    quality_df = pd.DataFrame({
        "category": ["Rooms","Bathroom","Bedroom","Overall"],
        "score": [Quality_Score_Rooms, Quality_Score_Bathroom, Quality_Score_Bedroom, Quality_Score_Overall]
    })
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=quality_df['score'].tolist(),
        theta=quality_df['category'].tolist(),
        fill='toself',
        name='Quality Scores'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])), showlegend=False, title="Quality radar")
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # 3) Histogram / distribution of "comparable prices"
    # Simulate comparables
    comps = np.random.normal(loc=(base_rate*Interior_SqFt)*(quality_factor), scale=0.1*base_rate*Interior_SqFt, size=80)
    comps = np.clip(comps, a_min=50000, a_max=None)
    fig_hist = px.histogram(comps, nbins=20, title="Comparable sales (simulated)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 4) Donut chart: cost split
    parts = pd.DataFrame({
        "part": ["Base value","Quality uplift","Fees"],
        "amount": [Interior_SqFt*base_rate, Interior_SqFt*(quality_factor*200), Registration_Fee+Commission]
    })
    fig_pie = px.pie(parts, names='part', values='amount', hole=0.45, title="Value split")
    st.plotly_chart(fig_pie, use_container_width=True)

with right_col:
    st.subheader("Quick insights")
    st.metric("Predicted Price (INR)", f"{predicted_price:,.2f}")
    st.write("- Distance to main road increases buyer convenience." )
    st.write("- Consider minor renovations if Quality_Score_Overall < 7 to raise price.")
    st.write(f"- Estimated per sq.ft. base: {base_rate} INR")
    st.write("- Use the charts to the left to inspect drivers.")
    st.markdown("----")
    
    # Detail table
    df_tab = pd.DataFrame([user_input])
    st.table(df_tab.T.rename(columns={0:"value"}))
    
    # Controls for PDF export
    st.markdown("### Export report")
    logo_url_input = st.text_input("Report logo URL (optional)", value="https://upload.wikimedia.org/wikipedia/commons/6/6a/House_%28simple_icon%29.svg")
    property_image_url_input = st.text_input("Property image URL (optional)", value=property_image_url)
    
    # Prepare images for PDF
    # Convert plotly figs to bytes buffers
    fig_bar_bytes = fig_to_bytes(fig_bar)
    radar_bytes = fig_to_bytes(radar_fig)
    hist_bytes = fig_to_bytes(fig_hist)
    pie_bytes = fig_to_bytes(fig_pie)
    
    figs_for_pdf = [
        (fig_bar_bytes, 170),
        (radar_bytes, 130),
        (hist_bytes, 170),
        (pie_bytes, 130)
    ]
    
    if st.button("Generate & Download PDF"):
        pdf_bytes = generate_pdf(user_input, figs_for_pdf, predicted_price, logo_url=logo_url_input, property_image_url=property_image_url_input)
        st.download_button("Download valuation report (PDF)", data=pdf_bytes, file_name="property_valuation_report.pdf", mime="application/pdf")

# Footer like website
st.markdown("---")
st.markdown("<small style='color:gray'>Sunrise Property Valuation Agency • Accurate. Transparent. Fast.</small>", unsafe_allow_html=True)
