# MedSimplify AI 

MedSimplify is a web app that takes complex medical reports as text or images and converts them into simple explanations in English and Hindi using an Ollama LLM backend

The goal is to help patients and families understand medical information in a clear, supportive way, without needing deep medical knowledge

## Features

**Upload medical reports**
  - Supportd formats .txt, .png, .jpg, .jpeg, .gif
  - Images are processed with OCR to extract text
**Paste report text directly**
  - Use the textbox to analyze any text based medical report
**AI-powered explanation (via Ollama)**
  - Generates structured explanation in English
  - Automatically generates the same explanation in Hindi
**Bilingual toggle**
  - Frontend has a simple English / Hindi switch
**Clear structure**
  - Sections:
    - What’s Going On  
    - Symptoms & Problems  
    - Treatment & Medications  
    - Next Steps & Monitoring
**Backend health check**
  - `/status` endpoint + UI indicator to show if the Ollama backend is online

## Tools used 

- **Backend:** Flask Python
- **Frontend:** HTML + Tailwind CSS + Vanilla JavaScript
- **AI:** Ollama model API
- **OCR:** Tesseract 
- **HTTP client:** `requests`

## Project Structure

```text
project-root/
├── main.py                  # Flask backend
├── requirements.txt
├── README.md
├── uploads/                # Temporary upload folder
└── templates/
    └── index.html          # Frontend UI
```

## Setup & Installation


**Follow these steps to get MedSimplify running locally**

1. Extract the project

2. Project structure should look like:
```text
project-root/
├── main.py                  # Flask backend
├── requirements.txt
├── README.md
├── uploads/                # Temporary upload folder
└── templates/
    └── index.html          # Frontend UI
```

3. Install Python 3.9+
   python --version

4. Create and activate a virtual environment:
   python -m venv venv
   Windows: venv\\Scripts\\activate
   macOS/Linux: source venv/bin/activate

5. Install dependencies:
   In terminal paste this - pip install -r requirements.txt

6. Install Tesseract OCR:
   Ubuntu/Debian:
     sudo apt update && sudo apt install tesseract-ocr
   macOS:
     brew install tesseract
   Windows:
     Install Tesseract and add it to PATH.

7. Configure environment variables (added my own for this submission and can be changed as per requirement):
   - FLASK_SECRET_KEY
   - UPLOAD_FOLDER
   - OLLAMA_BASE_URL
   - OLLAMA_MODEL
9. Run the application:
   In the terminal run - python main.py
   Go to http://localhost:5000

10. Test features:
   - Demo report
   - Upload
   - Paste text
   - Ollama status check

11. Thank you :)
