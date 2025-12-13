from flask import Flask, render_template, request, jsonify
import os
import re
import uuid
import requests
from PIL import Image
import pytesseract

"""
MedSimplify Backend:

Key features:
- Accepts medical reports as text or image files.
- Uses an Ollama LLM endpoint to generate bilingual (English + Hindi) explanations.
- Structured output with sections for better readability.
- Returns clean HTML sections for frontend rendering.
- Provides a /status endpoint to show AI connection health.
"""

app = Flask(__name__)

# Use an environment variable for security in production.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medsimplify-secret-key")

# Folder to temporarily store uploaded files (images / text).
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")

# Max file size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {"txt", "png", "jpg", "jpeg", "gif"}


class OllamaMedicalAnalyzer:
    """
    Encapsulates integration with an Ollama API for medical report simplification.

    Design choices:
    - No fallback summarizer: if the AI call fails, the request fails.
    - Generates bilingual output:
      1) English
      2) Hindi
    - Frontend toggles between these languages without extra API calls.
    """

    def __init__(self):
        # Base URL of Ollama backend (can be self-hosted).
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL", "https://ollama.thearogyamfoundation.info"
        )
        # Default model name for the Ollama backend.
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")

    def analyze_medical_report(self, medical_text):
        """
        Call the Ollama backend to generate summaries.

        The model is instructed to:
        - First output the explanation in English/Hindi.
        - Use specific section headings.
        - Keep language simple and supportive.
        """

        prompt = f"""You are a medical expert and patient educator. Your job is to read the medical report below and convert it into a clear, accurate, and supportive explanation that an 8th-grade student can understand.

MEDICAL REPORT:
{medical_text}

You MUST produce the explanation TWICE in this order:

1) First in ENGLISH
2) Then a line with exactly three dashes: ---
3) Then the SAME explanation in HINDI (simple, conversational Hindi)

For EACH LANGUAGE VERSION:

- Use these exact section headings in markdown format:
### What's Going On
### Symptoms & Problems
### Treatment & Medications (These are just suggestions consider medical assistance before purchase)
### Next Steps & Monitoring

- Under each heading, write short sentences or bullet-style lines.
- Use simple, non-technical language.
- Do NOT use emojis.
- Do NOT repeat the same information across sections.
- Keep tone calm and supportive.
- If the report is incomplete or unclear, mention it gently.
- End with a short note encouraging the patient to discuss this with their healthcare provider.

Important:
- ENGLISH part first.
- Then a line containing only: ---
- Then the HINDI part with the exact same structure and sections.
"""

        url = self.base_url.rstrip("/") + "/api/chat"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,  # Synchronous response
        }

        try:
            response = requests.post(
                url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            result = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse Ollama JSON: {e}")

        # Handle expected Ollama response formats (similar to OpenAI-like APIs)
        if "choices" in result:
            summary_text = result["choices"][0]["message"]["content"]
        elif "message" in result:
            summary_text = result["message"]["content"]
        else:
            raise RuntimeError("Unexpected Ollama response structure")

        # Normalize line endings and split EN/HI using a line that contains only '---'
        normalized = (
            summary_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        )
        parts = re.split(r"^---\s*$", normalized, flags=re.MULTILINE)

        # If the model fails to respect the separator, we still return something.
        if len(parts) == 1:
            eng_raw = parts[0].strip()
            hi_raw = parts[0].strip()
        else:
            eng_raw = parts[0].strip()
            hi_raw = parts[1].strip()

        html_en = self.clean_and_format_summary(eng_raw)
        html_hi = self.clean_and_format_summary(hi_raw)

        return {"en": html_en, "hi": html_hi}, True

    def clean_and_format_summary(self, summary):
        """
        Convert markdown-like text from the LLM into structured HTML.

        Steps:
        - Normalize line endings.
        - Ensure headings appear on their own lines.
        - Remove leading "Summary" labels if present.
        - Convert markdown bold (**text**) to <strong>.
        - Remove emojis for a clean medical UI.
        - Detect section headings (### ...).
        - Group text under headings and split into short bullet sentences.
        - Render each section using the `.medical-section` layout.
        """

        # Normalize line endings
        summary = summary.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Ensure headings like " ... ### What's Going On" are on new lines:
        # e.g., "... urgent. ### Symptoms & Problems" -> "... urgent.\n### Symptoms & Problems"
        summary = re.sub(r"\s*###\s+", r"\n### ", summary)

        # Drop a leading "Summary" line if present
        lower = summary.lower()
        if lower.startswith("summary"):
            parts = summary.split("\n", 1)
            summary = parts[1].strip() if len(parts) > 1 else ""

        # Convert markdown bold **text** into <strong>text</strong>
        summary = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", summary)

        # Remove emojis (safety + consistent styling)
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

        # Split into lines for further parsing
        lines = summary.split("\n")

        sections = []
        current_section = None
        buffer = []

        def flush_section():
            """
            Helper to move buffered text lines into bullet items for the current section.
            Splits into short sentences for better readability.
            """
            nonlocal buffer, current_section, sections
            if current_section is None:
                buffer = []
                return

            text = " ".join(buffer).strip()
            buffer = []

            if not text:
                return

            # Remove stray asterisks
            text = text.replace("*", "")

            # Split into sentences using a simple regex
            # Taking care not to split on common abbreviations (Dr., Mr., etc.).
            sentences = re.split(
                r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<=[.!?])\s+",
                text,
            )

            for s in sentences:
                s = s.strip()
                if s:
                    current_section["items"].append(s)

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Detect markdown headings: ### Heading
            m = re.match(r"^#{1,6}\s*(.+)$", line)
            if m:
                # Start a new section
                flush_section()
                heading_text = m.group(1).strip()

                # Normalize known headings to keep UI consistent
                heading_lower = heading_text.lower()
                if "what's going on" in heading_lower:
                    heading = "What's Going On"
                elif "symptoms" in heading_lower:
                    heading = "Symptoms & Problems"
                elif "treatment" in heading_lower:
                    heading = (
                        "Treatment & Medications "
                        "(These are just suggestions consider medical assistance before purchase)"
                    )
                elif "next steps" in heading_lower or "monitoring" in heading_lower:
                    heading = "Next Steps & Monitoring"
                else:
                    heading = heading_text

                current_section = {"heading": heading, "items": []}
                sections.append(current_section)
            else:
                # Regular content lines go into the buffer for the current section
                if current_section is None:
                    # If content appears before any heading, group it under a generic "Summary" heading
                    current_section = {"heading": "Summary", "items": []}
                    sections.append(current_section)
                buffer.append(line)

        # Flush the last section
        flush_section()

        # Build final HTML using .medical-section containers
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

    def extract_text_from_image(self, image_path):       #Extract text from an image using OCR.

        try:
            text = pytesseract.image_to_string(Image.open(image_path))
            return (
                text
                if text.strip()
                else "Could not read text from image. Please try a clearer image."
            )
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def allowed_file(self, filename):   #Check if the uploaded file extension is in the allowed list.
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        )

    def check_status(self):
        """
        Perform a lightweight health check on the Ollama backend.

        Uses the /api/tags endpoint to confirm connectivity.

        Returns:
            dict with keys:
            - online (bool)
            - model (str)
            - error (optional, str)
        """
        try:
            url = self.base_url.rstrip("/") + "/api/tags"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return {"online": True, "model": self.model}
            return {
                "online": False,
                "model": self.model,
                "error": f"Status code {r.status_code}",
            }
        except Exception as e:
            return {"online": False, "model": self.model, "error": str(e)}


# Initialize the analyzer once for reuse across requests
medical_ai = OllamaMedicalAnalyzer()


@app.route("/")
def index():    
    #Render the main frontend page.   
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    """
    Endpoint used by the frontend to show AI connection status.

    Returns:
        JSON structure like:
        {
          "success": true,
          "ollama": {
            "online": true/false,
            "model": "...",
            "error": "optional"
          }
        }
    """
    s = medical_ai.check_status()
    return jsonify({"success": True, "ollama": s})


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle file upload for medical reports (image or text).

    Steps:
    - Validate the file.
    - Save temporarily.
    - Run OCR for images or read text for .txt files.
    - Call the AI summarizer.
    - Clean up the temporary file.
    - Return bilingual HTML summaries.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file selected"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and medical_ai.allowed_file(file.filename):
            # Generate a unique filename to avoid collisions
            file_id = str(uuid.uuid4())
            file_extension = file.filename.rsplit(".", 1)[1].lower()
            filename = f"{file_id}.{file_extension}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Extract text based on file type
            if file_extension in ["png", "jpg", "jpeg", "gif"]:
                extracted_text = medical_ai.extract_text_from_image(filepath)
            elif file_extension == "txt":
                with open(
                    filepath, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    extracted_text = f.read()
            else:
                extracted_text = (
                    "Please use text or image files for this demo."
                )

            # Clean up the uploaded file after processing
            try:
                os.remove(filepath)
            except Exception as e:
                print("[UPLOAD] Failed to delete temp file:", e)

            # Ensure we have enough text to analyze
            if extracted_text and len(extracted_text.strip()) > 30:
                try:
                    summaries, used_ollama = medical_ai.analyze_medical_report(
                        extracted_text
                    )
                except RuntimeError as e:
                    # AI-specific failures surfaced clearly to the frontend
                    return jsonify({"error": str(e)}), 500

                return jsonify(
                    {
                        "success": True,
                        "summary_en": summaries["en"],
                        "summary_hi": summaries["hi"],
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
                            "error": "Could not extract enough text from the file. Please try with a clearer image or a text file with more content."
                        }
                    ),
                    400,
                )

        else:
            return (
                jsonify(
                    {
                        "error": "File type not allowed. Please upload TXT, PNG, JPG, JPEG, or GIF files."
                    }
                ),
                400,
            )

    except Exception as e:
        print("[UPLOAD] Exception:", e)
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/text", methods=["POST"])
def process_text():
    """
    Handle raw text submission of medical reports.

    Steps:
    - Validate incoming JSON.
    - Ensure minimum text length.
    - Call the AI summarizer.
    - Return bilingual HTML summaries.
    """
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

        # Generate summary using Ollama
        try:
            summaries, used_ollama = medical_ai.analyze_medical_report(text)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        return jsonify(
            {
                "success": True,
                "summary_en": summaries["en"],
                "summary_hi": summaries["hi"],
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
    """
    Provide a sample medical report for quick testing/demo.

    This is used by the "Try with Sample Medical Report" button on the frontend.
    """
    demo_report = """PATIENT: John Smith, 58-year-old male
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
    print(
        "Ollama base URL:",
        os.getenv("OLLAMA_BASE_URL", "https://ollama.thearogyamfoundation.info"),
    )
    print("Access the application at: http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)