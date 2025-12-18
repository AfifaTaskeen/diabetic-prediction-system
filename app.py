from flask import Flask, render_template, request, send_file, session
import joblib
import numpy as np
import os
from fpdf import FPDF
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load Assets from your specific subfolder 
MODEL_PATH = os.path.join('diabetic_prediction', 'diabetes_model.pkl')
SCALER_PATH = os.path.join('diabetic_prediction', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from HTML Form
    data = {
        "name": request.form['name'],
        "preg": float(request.form['preg']),
        "glu": float(request.form['glu']),
        "bp": float(request.form['bp']),
        "skin": float(request.form['skin']),
        "ins": float(request.form['ins']),
        "bmi": float(request.form['bmi']),
        "dpf": float(request.form['dpf']),
        "age": float(request.form['age'])
    }
    
    # ML Prediction logic [cite: 1, 4, 5]
    features = np.array([[data['preg'], data['glu'], data['bp'], data['skin'], data['ins'], data['bmi'], data['dpf'], data['age']]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    result_text = "Higher Diabetic Risk" if prediction == 1 else "Lower Diabetic Risk"
    session['report_data'] = {**data, "result": result_text}
    
    return render_template('result.html', result=result_text, name=data['name'])

@app.route('/download_pdf')
def download_pdf():
    d = session.get('report_data')
    
    # Create PDF with better formatting
    class PDF(FPDF):
        def header(self):
            # Logo and title
            self.set_font('Arial', 'B', 20)
            self.set_text_color(37, 99, 235)  # Primary blue
            self.cell(0, 15, 'Diabetes Risk Assessment Report', 0, 1, 'C')
            self.set_font('Arial', 'I', 12)
            self.set_text_color(100, 116, 139)  # Gray 600
            self.cell(0, 5, 'Comprehensive Health Analysis Report', 0, 1, 'C')
            self.ln(5)
            self.line(10, 30, 200, 30)  # Divider line
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 116, 139)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Set default font
    pdf.set_font('Arial', '', 12)
    
    # Report metadata
    current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    # Patient Information Section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(15, 23, 42)  # Dark gray
    pdf.cell(0, 10, '1. Patient Information', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    # Patient info table
    col_width = 95
    row_height = 10
    
    # Row 1
    pdf.cell(col_width, row_height, f"Patient Name: {d['name']}", 0, 0, 'L')
    pdf.cell(col_width, row_height, f"Date of Assessment: {current_date}", 0, 1, 'L')
    
    # Risk Assessment Section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '2. Risk Assessment', 0, 1, 'L')
    
    # Risk level box
    pdf.set_fill_color(239, 246, 255)  # Light blue background
    pdf.set_draw_color(147, 197, 253)  # Border color
    pdf.set_line_width(0.5)
    pdf.rect(10, pdf.get_y(), 190, 30, 'DF')
    
    pdf.set_xy(20, pdf.get_y() + 5)
    pdf.set_font('Arial', 'B', 16)
    if "Higher" in d['result']:
        pdf.set_text_color(220, 38, 38)  # Red for high risk
        risk_level = "High"
        risk_text = "Our analysis indicates an elevated risk of diabetes based on the provided health indicators."
    else:
        pdf.set_text_color(5, 150, 105)  # Green for low risk
        risk_level = "Low"
        risk_text = "Your current health indicators suggest a lower risk of diabetes at this time."
    
    pdf.cell(0, 10, f"Diabetes Risk Level: {risk_level}", 0, 1, 'L')
    
    pdf.set_xy(20, pdf.get_y() + 5)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(170, 8, risk_text, 0, 'L')
    
    # Key Findings Section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '3. Key Findings', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    if "Higher" in d['result']:
        findings = [
            "• Elevated glucose levels were noted in your assessment.",
            "• Blood pressure readings may indicate potential risk factors.",
            "• Body Mass Index (BMI) suggests increased diabetes risk.",
            "• Other metabolic indicators show values that warrant attention."
        ]
    else:
        findings = [
            "• Glucose levels are within normal ranges.",
            "• Blood pressure readings are at healthy levels.",
            "• Body Mass Index (BMI) falls within recommended guidelines.",
            "• Overall metabolic indicators suggest good health."
        ]
    
    for item in findings:
        pdf.cell(10)  # Indent
        pdf.multi_cell(0, 8, item, 0, 'L')
    
    # Recommendations Section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '4. Recommendations', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    if "Higher" in d['result']:
        recommendations = [
            ("Medical Consultation", "Schedule an appointment with a healthcare provider for a comprehensive evaluation and possible diagnostic testing."),
            ("Nutrition", "Adopt a balanced diet rich in whole grains, lean proteins, and vegetables. Limit processed foods and added sugars."),
            ("Physical Activity", "Aim for at least 150 minutes of moderate-intensity exercise per week, such as brisk walking or cycling."),
            ("Regular Monitoring", "Schedule regular check-ups and monitor your blood sugar levels as recommended by your healthcare provider.")
        ]
    else:
        recommendations = [
            ("Maintain Healthy Habits", "Continue with your current balanced diet and regular physical activity."),
            ("Nutrition", "Focus on whole foods, plenty of vegetables, and limit processed foods and added sugars."),
            ("Physical Activity", "Aim for at least 150 minutes of moderate exercise weekly to maintain good health."),
            ("Regular Check-ups", "Continue with routine health screenings to monitor your health status.")
        ]
    
    for i, (title, desc) in enumerate(recommendations, 1):
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(10)  # Indent
        pdf.cell(0, 10, f"{i}. {title}", 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.cell(20)  # Additional indent for description
        pdf.multi_cell(0, 8, desc, 0, 'L')
        pdf.ln(2)
    
    # Disclaimer Section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, '5. Important Disclaimer', 0, 1, 'L')
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 116, 139)
    
    disclaimer_text = """
    This assessment is based on the information provided and is generated using an AI model. 
    It is not a medical diagnosis and should not be used as a substitute for professional 
    medical advice, diagnosis, or treatment. Always seek the advice of your physician or 
    other qualified health provider with any questions you may have regarding a medical condition.
    
    This report was generated by the Diabetes Risk Assessment System and is intended for 
    informational purposes only. The accuracy of the assessment depends on the accuracy and 
    completeness of the information provided.
    
    Report generated on: {}
    
    {} - Diabetes Risk Assessment System. All rights reserved.
    """.format(current_date, datetime.now().year)
    
    pdf.multi_cell(0, 8, disclaimer_text, 0, 'L')
    
    # Save to bytes buffer
    output = io.BytesIO()
    pdf_str = pdf.output(dest='S').encode('latin-1')
    output.write(pdf_str)
    output.seek(0)
    
    # Generate a filename with patient name and date
    safe_name = "".join([c if c.isalnum() else "_" for c in d['name']])
    filename = f"Diabetes_Risk_Report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf',
        as_attachment_kwargs={
            'mimetype': 'application/pdf',
            'as_attachment': True,
            'download_name': filename
        }
    )

if __name__ == '__main__':
    app.run(debug=True)