import os
import time
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import requests
from dotenv import load_dotenv


app = Flask(__name__)

# Folder for storing uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



load_dotenv()
API_KEY = os.getenv("API_KEY")


# IBM API key


# Watsonx API URL
WATSONX_API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

# Global variables for token management
ACCESS_TOKEN = None
TOKEN_EXPIRATION = 0  # UNIX timestamp for when the token expires


@app.route('/')
def welcome():
    return render_template('welcome.html')
# Function to generate a new access token
@app.route('/home')
def home():
    return render_template('index.html')
def generate_access_token():
    global ACCESS_TOKEN, TOKEN_EXPIRATION

    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"apikey": API_KEY, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"}

    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code == 200:
        token_data = response.json()
        ACCESS_TOKEN = token_data['access_token']
        TOKEN_EXPIRATION = time.time() + token_data['expires_in']
        print(f"New token generated, expires in {token_data['expires_in']} seconds.")
    else:
        raise Exception(f"Failed to generate token: {response.text}")

# Function to get the access token
def get_access_token():
    if ACCESS_TOKEN is None or time.time() >= TOKEN_EXPIRATION:
        generate_access_token()
    return ACCESS_TOKEN

# Function to extract text using PyMuPDF (fitz)
def extract_text_from_pdf(filepath):
    print(f"Extracting text from PDF: {filepath}")
    
    text = ""
    # Open the PDF using PyMuPDF (fitz)
    with fitz.open(filepath) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  
            text += page.get_text("text")  

    print(f"Extracted text from PDF (text): {text[:500]}...")  # Limit the output to first 500 characters for brevity
    return text

# Function to extract images from PDF and use pytesseract to perform OCR
def extract_text_from_pdf_images(filepath):
    print(f"Extracting text from images in PDF: {filepath}")
    
    image_text = ""
    
    # Open the PDF using PyMuPDF (fitz)
    with fitz.open(filepath) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load each page
            images = page.get_images(full=True)  # Get all images on the page
            
            for img_index, img in enumerate(images):
                xref = img[0]  # XREF index of the image
                base_image = doc.extract_image(xref)  # Extract the image
                image_bytes = base_image["image"]  # Get image bytes
                image_ext = base_image["ext"]  # Get the image file extension (png, jpeg, etc.)
                
                # Open image from bytes
                image = Image.open(io.BytesIO(image_bytes))
                
                # Perform OCR on the image
                text_from_image = pytesseract.image_to_string(image)
                image_text += f"\n\nText from image {img_index+1} on page {page_num+1}:\n{text_from_image}"

    print(f"Extracted text from PDF (images): {image_text[:500]}...")  # Limit the output to first 500 characters for brevity
    return image_text

# Function to send extracted text to Watsonx API for specific law
def detect_compliance_with_watsonx(extracted_text, law_type):
    # Get a valid access token (generate if necessary)
    access_token = get_access_token()

    # Define the prompt based on the selected law
    if law_type == "hipaa":
        law_prompt = """You are tasked with analyzing the following document to determine whether it is compliant with the Health Insurance Portability and Accountability Act (HIPAA). 
            Generate a professional summary that includes the following:
            - Compliance status (whether the document complies with HIPAA or not).
            - Compliance score (out of 100).
            - Identified gaps in compliance.
            - Recommended actions and improvements to meet compliance standards.
            - Specific suggestions on closing the gaps.

            The output should be a professional report written by a senior cybersecurity professional. Please analyze the following document:
            ---
            {extracted_text}"""
    elif law_type == "gdpr":
        law_prompt = """You are tasked with analyzing the following document to determine whether it is compliant with the General Data Protection Regulation (GDPR). 
                Generate a professional summary that includes the following:
                - Compliance status (whether the document complies with GDPR or not).
                - Compliance score (out of 100).
                - Identified gaps in compliance.
                - Recommended actions and improvements to meet compliance standards.
                - Specific suggestions on closing the gaps.

                The output should be a professional report written by a senior cybersecurity professional. Please analyze the following document:
                ---
                {extracted_text}"""
    elif law_type == "dpdp":
        law_prompt = """You are tasked with analyzing the following document to determine whether it is compliant with India's Digital Personal Data Protection Bill 2022 (DPDP). Generate a professional summary that includes the following:
        - Compliance status (whether the document complies with DPDP or not).
        - Compliance score (out of 100).- Identified gaps in compliance.
        - Recommended actions and improvements to meet compliance standards.
        - Specific suggestions on closing the gaps.

        The output should be a professional report written by a senior cybersecurity professional. Please analyze the following document:
        ---
        {extracted_text}"""    
    else:
        law_prompt = """You are tasked with detecting Personally Identifiable Information (PII) in the following document. Categorize any found PII into the following categories: Names, Addresses, Phone Numbers, Email Addresses, Social Security Numbers, Dates of Birth, and Others. Provide a summary of the PII detected and any areas of concern."""
    
    # Prepare the request to Watsonx API
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    body = {
        "input": f"""<|system|>
        {law_prompt}
        ---
        {extracted_text}
        <|assistant|>
        """,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 400,
            "repetition_penalty": 1
        },
        "model_id": "ibm/granite-13b-chat-v2",
        "project_id": "21160f40-c1a7-4283-925c-143d08947b39"
    }

    print(f"Sending {law_type.upper()} request to Watsonx API...")
    response = requests.post(WATSONX_API_URL, headers=headers, json=body)

    if response.status_code == 200:
        result = response.json()
        print(f"Response from Watsonx API ({law_type.upper()}) received successfully.")
        print(result)  # Debug output of full response
        compliance_output = result['results'][0]['generated_text']
    else:
        compliance_output = f"Error: {response.status_code} - {response.text}"
        print(f"Error from Watsonx API: {compliance_output}")

    return compliance_output

# Route for file upload and compliance detection
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            print("No file part in the request.")
            return "No file part"
        file = request.files['file']
        law_type = request.form.get('law_type', 'pii')  # Get the selected law type from form
        if file.filename == '':
            print("No selected file.")
            return "No selected file"
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File uploaded successfully: {filename}")

            # Extract text based on file type
            extracted_text = ""
            if filename.lower().endswith('.pdf'):
                # First extract normal text from the PDF
                text_from_pdf = extract_text_from_pdf(filepath)

                # Then extract text from images in the PDF (if any)
                text_from_images = extract_text_from_pdf_images(filepath)

                # Combine both results (text + image-based text)
                extracted_text = text_from_pdf + text_from_images

            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                extracted_text = extract_text_from_image(filepath)
            else:
                print("Unsupported file format.")
                return "Unsupported file format"

            # Detect compliance under selected law using Watsonx AI
            compliance_output = detect_compliance_with_watsonx(extracted_text, law_type)

            print(f"Compliance Summary ({law_type.upper()}): {compliance_output}")

            # Pass the extracted text, compliance output, and filename to the result page
            return render_template('result.html', extracted_text=extracted_text, compliance_output=compliance_output, filename=filename, law_type=law_type)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
