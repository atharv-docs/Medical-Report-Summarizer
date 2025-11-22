from flask import Flask, render_template, request, jsonify
import os
import re
import uuid
import requests
from PIL import Image
import pytesseract

app = Flask(__name__)
app.secret_key = 'medsimplify-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif'}


class OllamaMedicalAnalyzer:
    """
    Uses an Ollama-compatible API for medical report simplification.
    Falls back to a rule-based explainer if Ollama is unavailable.
    """

    def __init__(self):
        # Base URL for your Ollama endpoint (I have used my own hosted instance)
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "https://ollama.thearogyamfoundation.info"
        )
        # Default model to use
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
        self.ollama_available = True  # changes to False if calls keep failing

    def analyze_medical_report(self, medical_text):
        """
        Try to use Ollama for summarization.
        If anything fails, fall back to rule-based summary.

        Returns: (summary_html: str, used_ollama: bool)
        """
        if not self.ollama_available:
            print("[OLLAMA] Marked unavailable earlier. Using fallback summarizer.")
            return self.get_fallback_summary(medical_text), False

        # Structured prompt
        prompt = f"""You are a medical expert and patient educator. Your job is to read the medical report below and convert it into a clear, accurate, and supportive explanation that an 8th-grade student can understand.

MEDICAL REPORT:
{medical_text}

Your output MUST follow this exact structure with these section names:

### What's Going On
- Summarize the recommended follow-up actions and you may add your opinion if appropriate
- State the main diagnosis or health condition in simple, everyday language
- Explain what this condition means for the patient's body
- Include how serious or urgent the situation is, if mentioned

### Symptoms & Problems
- Summarize the recommended follow-up actions and you may add your opinion if appropriate
- List the important symptoms or issues the report mentions
- Explain what each symptom means in normal words
- Highlight anything that may need close attention

### Treatment & Medications (These are just suggestions consider medical assistance before purchase)
- Summarize the recommended follow-up actions and you may add your opinion if appropriate
- List all treatments, medications, or procedures from the report
- Explain what each one is meant to do for the patient
- Include any lifestyle or care advice if provided

### Next Steps & Monitoring
- Summarize the recommended follow-up actions and you may add your opinion if appropriate
- What follow-up tests or visits are recommended
- What the patient should keep an eye on
- Clear warning signs that should be treated as urgent
- Any immediate actions required

INSTRUCTIONS:
- MOST IMPORTANT: Use simple, non-technical language throughout
- Break down complex medical terms into easy explanations
- Maintain spaces in between the sentences for readability even though the data is without spaces
- Use bullet points in every section
- Do NOT use medical jargon unless absolutely necessary, and explain it when you do
- Develop an abstractive text summarization system that transforms a long medical note into a summary written at a Grade 8 reading level.
- Keep the tone calm, supportive, and easy to understand
- Preserve all medically important information from the original report
- If the report is incomplete or unclear, make the best safe interpretation and mention it gently
- End with a note encouraging the patient to discuss the summary with their healthcare provider for personalized advice
- Reduce the number of points if the information is limited, but keep all key details
- Strictly do not repeat information across sections
- DO NOT USE EMOJIS IN YOUR RESPONSE
"""

        try:
            url = self.base_url.rstrip("/") + "/api/chat"

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            }

            print("[OLLAMA] Sending request to:", url)
            response = requests.post(
                url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60,
            )
            print("[OLLAMA] Status code:", response.status_code)

            if response.status_code != 200:
                print("[OLLAMA] Non-200 response, using fallback. Body:", response.text[:500])
                if response.status_code in (404, 500):
                    self.ollama_available = False
                return self.get_fallback_summary(medical_text), False

            try:
                result = response.json()
            except Exception as e:
                print("[OLLAMA] Failed to parse JSON:", e)
                self.ollama_available = False
                return self.get_fallback_summary(medical_text), False

            # Expected Ollama chat response formats
            try:
                if "choices" in result:
                    summary = result["choices"][0]["message"]["content"]
                elif "message" in result:
                    summary = result["message"]["content"]
                else:
                    raise KeyError("No 'choices' or 'message' in Ollama response")
            except (KeyError, IndexError, TypeError) as e:
                print("[OLLAMA] Unexpected response structure:", e)
                self.ollama_available = False
                return self.get_fallback_summary(medical_text), False

            cleaned = self.clean_and_format_summary(summary)
            print("[OLLAMA] Successfully generated summary using Ollama.")
            return cleaned, True

        except Exception as e:
            print("[OLLAMA] Exception while calling Ollama:", e)
            self.ollama_available = False
            return self.get_fallback_summary(medical_text), False

    def clean_and_format_summary(self, summary):
        """
        Take the LLM's markdown-ish text and convert it into clean HTML:
        - Detect sections: What's Going On, Symptoms & Problems, etc.
        - Support inline headings like: '... urgent. ### Symptoms & Problems'
        - Split section text into bullet sentences.
        - Use .medical-section blocks for each section.
        """

        # 1) Normalize line endings
        summary = summary.replace("\r\n", "\n").replace("\r", "\n").strip()

        # 2) Put any ' ### Heading' on a new line
        # e.g. "... urgent. ### Symptoms & Problems" -> "... urgent.\n### Symptoms & Problems"
        summary = re.sub(r"\s*###\s+", r"\n### ", summary)

        # 3) Drop leading 'Summary' label if present
        if summary.lower().startswith("summary"):
            # remove just the first line "Summary"
            parts = summary.split("\n", 1)
            summary = parts[1].strip() if len(parts) > 1 else ""

        # 4) Convert markdown bold **text** -> <strong>text</strong>
        summary = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", summary)

        # 5) Emoji remover (just in case)
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE,
        )
        summary = emoji_pattern.sub("", summary)

        # 6) Split into lines for heading detection
        lines = summary.split("\n")

        sections = []
        current_section = None
        buffer = []

        def flush_section():
            """Move buffered text into bullets for the current section."""
            nonlocal buffer, current_section, sections
            if current_section is None:
                buffer = []
                return
            text = " ".join(buffer).strip()
            buffer = []
            if not text:
                return

            # remove stray asterisks from note lines like *Please remember...*
            text = text.replace("*", "")

            # Split into sentences
            sentences = re.split(
                r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<=[.!?])\s+",
                text
            )

            for s in sentences:
                s = s.strip()
                if s:
                    current_section["items"].append(s)

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Heading pattern: ### Something
            m = re.match(r"^#{1,6}\s*(.+)$", line)
            if m:
                # New section starts
                flush_section()
                heading_text = m.group(1).strip()

                # Normalize known headings
                if "what's going on" in heading_text.lower():
                    heading = "What's Going On"
                elif "symptoms" in heading_text.lower():
                    heading = "Symptoms & Problems"
                elif "treatment" in heading_text.lower():
                    heading = "Treatment & Medications (These are just suggestions consider medical assistance)"
                elif "next steps" in heading_text.lower():
                    heading = "Next Steps & Monitoring"
                else:
                    heading = heading_text

                current_section = {"heading": heading, "items": []}
                sections.append(current_section)
            else:
                # Normal content line → buffer into current section
                if current_section is None:
                    # if somehow text appears before any heading, throw it into a 'Summary' section
                    current_section = {"heading": "Summary", "items": []}
                    sections.append(current_section)
                buffer.append(line)

        # Flush last section
        flush_section()

        # 7) Build HTML using your medical-section layout
        html_parts = []
        for sec in sections:
            html_parts.append(
                '<div class="medical-section rounded-lg p-4">'
                f'<h4 class="font-semibold text-lg mb-2">{sec["heading"]}</h4>'
                '<ul class="list-disc list-inside space-y-1">'
            )
            for item in sec["items"]:
                html_parts.append(f"<li>{item}</li>")
            html_parts.append("</ul></div>")

        return "\n".join(html_parts)

    def get_fallback_summary(self, medical_text):
        """Generate a basic summary when Ollama is unavailable"""
        text_lower = medical_text.lower()
        sections = []

        # Extract potential conditions
        conditions = []
        condition_keywords = ["diagnosis", "diagnosed", "finding", "condition", "disease"]
        for keyword in condition_keywords:
            if keyword in text_lower:
                sentences = re.findall(
                    r"[^.]*?" + keyword + r"[^.]*\.", medical_text, re.IGNORECASE
                )
                conditions.extend(sentences[:2])

        if conditions:
            sections.append(
                "<strong>What's Going On</strong><br>• "
                + "<br>• ".join(conditions[:3])
            )
        else:
            sections.append(
                "<strong>What's Going On</strong><br>• The report describes some medical findings that may indicate a health condition. Please discuss them with your doctor for a full explanation."
            )

        # Extract symptoms
        symptoms = []
        symptom_keywords = [
            "symptom",
            "pain",
            "fever",
            "cough",
            "headache",
            "nausea",
            "vomiting",
            "dizziness",
            "shortness of breath",
            "chest discomfort",
            "chest pain",
        ]
        for keyword in symptom_keywords:
            if keyword in text_lower:
                sentences = re.findall(
                    r"[^.]*?" + keyword + r"[^.]*\.", medical_text, re.IGNORECASE
                )
                symptoms.extend(sentences[:2])

        if symptoms:
            sections.append(
                "<strong>Symptoms & Problems</strong><br>• "
                + "<br>• ".join(symptoms[:5])
            )
        else:
            sections.append(
                "<strong>Symptoms & Problems</strong><br>• The report mentions clinical findings or measurements, but specific symptoms are not clearly listed."
            )

        # Extract medications / treatments
        medications = []
        med_keywords = ["prescribed", "medication", "treatment", "therapy", "dose", "mg", "tablet", "statin", "beta-blocker"]
        for keyword in med_keywords:
            if keyword in text_lower:
                sentences = re.findall(
                    r"[^.]*?" + keyword + r"[^.]*\.", medical_text, re.IGNORECASE
                )
                medications.extend(sentences[:3])

        if medications:
            sections.append(
                "<strong>Treatment & Medications (These are just suggestions consider medical assistance before purchase)</strong><br>• "
                + "<br>• ".join(medications[:5])
            )
        else:
            sections.append(
                "<strong>Treatment & Medications</strong><br>• The report does not clearly list medications or treatments, or they are implied rather than stated."
            )

        # Extract recommendations / follow-up
        recommendations = []
        rec_keywords = ["follow up", "monitor", "recommend", "advise", "suggest", "review", "control", "lifestyle"]
        for keyword in rec_keywords:
            if keyword in text_lower:
                sentences = re.findall(
                    r"[^.]*?" + keyword + r"[^.]*\.", medical_text, re.IGNORECASE
                )
                recommendations.extend(sentences[:4])

        if recommendations:
            sections.append(
                "<strong>Next Steps & Monitoring</strong><br>• "
                + "<br>• ".join(recommendations[:6])
            )
        else:
            sections.append(
                "<strong>Next Steps & Monitoring</strong><br>• Regular follow-up with your healthcare provider is important to understand these results and plan treatment.<br>• Watch for any worsening symptoms and seek urgent medical help if you feel very unwell, have severe pain, difficulty breathing, or sudden changes in your condition."
            )

        sections.append(
            "<br><em>Note: This is a simplified explanation based only on the text provided. "
            "For a complete and personalized interpretation, please discuss this report with your doctor.</em>"
        )

        return "<br><br>".join(sections)

    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        try:
            text = pytesseract.image_to_string(Image.open(image_path))
            return (
                text
                if text.strip()
                else "Could not read text from image. Please try a clearer image."
            )
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def allowed_file(self, filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize analyzer
medical_ai = OllamaMedicalAnalyzer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file selected"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and medical_ai.allowed_file(file.filename):
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = file.filename.rsplit(".", 1)[1].lower()
            filename = f"{file_id}.{file_extension}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Extract text based on file type
            if file_extension in ["png", "jpg", "jpeg", "gif"]:
                extracted_text = medical_ai.extract_text_from_image(filepath)
            elif file_extension == "txt":
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            else:
                extracted_text = "Please use text or image files for this demo."

            # Clean up uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                print("[UPLOAD] Failed to delete temp file:", e)

            # Generate summary
            if extracted_text and len(extracted_text.strip()) > 30:
                summary, used_ollama = medical_ai.analyze_medical_report(extracted_text)

                return jsonify(
                    {
                        "success": True,
                        "summary": summary,
                        "original_text_sample": (
                            extracted_text[:200] + "..."
                            if len(extracted_text) > 200
                            else extracted_text
                        ),
                        "used_ollama": used_ollama,
                    }
                )
            else:
                return (
                    jsonify(
                        {
                            "error": "Could not extract enough text from the file. Please try with a clearer image or text file with more content."
                        }
                    ),
                    400,
                )

        else:
            return (
                jsonify(
                    {
                        "error": "File type not allowed. Please upload TXT, PNG, JPG, or JPEG files."
                    }
                ),
                400,
            )

    except Exception as e:
        print("[UPLOAD] Exception:", e)
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/text", methods=["POST"])
def process_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        text = data.get("text", "")

        if not text or len(text.strip()) < 30:
            return (
                jsonify(
                    {
                        "error": "Please enter a longer medical report (at least 30 characters)."
                    }
                ),
                400,
            )

        # Generate summary
        summary, used_ollama = medical_ai.analyze_medical_report(text)

        return jsonify(
            {
                "success": True,
                "summary": summary,
                "original_text_sample": (
                    text[:200] + "..." if len(text) > 200 else text
                ),
                "used_ollama": used_ollama,
            }
        )

    except Exception as e:
        print("[TEXT] Exception:", e)
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/demo", methods=["GET"])
def get_demo_report():
    """Provide a sample medical report for testing"""
    demo_report = """
PATIENT: John Smith, 58-year-old male
CHIEF COMPLAINT: Chest discomfort and shortness of breath
HISTORY OF PRESENT ILLNESS: Patient presents with substernal chest pain radiating to the left arm, associated with diaphoresis and nausea. Symptoms began approximately 3 hours prior to admission. Patient has a history of hypertension and hyperlipidemia.
PHYSICAL EXAMINATION: BP 158/92, HR 108 bpm, RR 24, O2 saturation 94% on room air. Cardiac exam reveals regular rhythm without murmurs. Lungs clear to auscultation.
DIAGNOSTIC RESULTS: ECG shows ST-segment elevation in anterior leads. Cardiac enzymes elevated: Troponin I 12.5 ng/mL, CK-MB 45 ng/mL. Echocardiogram shows anterior wall hypokinesis with estimated EF 45%.
ASSESSMENT: Acute Anterior Wall Myocardial Infarction. Hypertension. Hyperlipidemia.
PLAN: 
1. Admit to Cardiac Care Unit
2. Start dual antiplatelet therapy: Aspirin 325 mg, Clopidogrel 75 mg daily
3. Statin therapy: Atorvastatin 40 mg daily
4. Beta-blocker: Metoprolol 25 mg twice daily
5. Schedule cardiac catheterization for possible PCI
6. Follow up with Cardiology in 2 weeks
"""
    return jsonify({"demo_report": demo_report})


if __name__ == "__main__":
    print("Starting MedSimplify...")
    print("Ollama base URL:", os.getenv("OLLAMA_BASE_URL", "https://ollama.thearogyamfoundation.info"))
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
