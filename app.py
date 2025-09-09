from PIL import Image

def generate_pdf(user_input, figs, prediction, property_image):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # --- Header ---
    pdf.set_fill_color(0, 102, 204)  # Blue
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

    # --- Property Information ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Property Information", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in user_input.items():
        safe_v = str(v).encode('latin-1', 'replace').decode('latin-1')
        safe_k = str(k).encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(60, 6, f"{safe_k}", border=1)
        pdf.cell(0, 6, f"{safe_v}", border=1, ln=True)
    pdf.ln(5)

    # --- Property Image ---
    if property_image:
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            # Convert uploaded image to PNG using PIL
            img = Image.open(property_image).convert("RGB")
            img.save(tmp_img.name, "PNG")
            pdf.image(tmp_img.name, x=30, w=120)
        finally:
            tmp_img.close()
            os.unlink(tmp_img.name)
        pdf.ln(5)

    # --- Prediction ---
    pdf.set_fill_color(255, 204, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Price: {prediction:,.2f} INR", ln=True, fill=True, align="C")
    pdf.ln(5)

    # --- Charts ---
    for fig, width in zip(figs, [170, 130, 130, 130, 130]):  # adjust sizes
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.write_image(tmp_file.name, format="png")  # Plotly export
        tmp_file.close()
        pdf.image(tmp_file.name, x=20, w=width)
        os.unlink(tmp_file.name)
        pdf.ln(5)

    # --- Insights ---
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

    # --- Footer ---
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
