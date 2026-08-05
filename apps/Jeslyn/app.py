from datetime import datetime, timedelta
from pathlib import Path
import os
import re

from flask import Flask, redirect, render_template, request, session, url_for
from dotenv import load_dotenv
import joblib
import pandas as pd
from pypdf import PdfReader

try:
    import google.generativeai as genai
except Exception:
    genai = None


# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")


class PrefixMiddleware:
    """Makes url_for() emit paths prefixed with X-Forwarded-Prefix when this
    app is proxied behind the gateway under a sub-path (e.g. /stroke). A
    no-op when the header is absent, so standalone runs are unaffected."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        prefix = environ.get("HTTP_X_FORWARDED_PREFIX", "")
        if prefix:
            environ["SCRIPT_NAME"] = prefix
        return self.wsgi_app(environ, start_response)


app.wsgi_app = PrefixMiddleware(app.wsgi_app)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def _resolve_rag_source_folder() -> Path:
    """Find the rag_sources folder using common project paths."""
    candidates = [
        BASE_DIR / "data" / "rag_sources",
        BASE_DIR / "Data" / "rag_sources",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


# Load ML artifacts once when app starts.
model = joblib.load(MODELS_DIR / "stroke_logistic_model.pkl")
scaler = joblib.load(MODELS_DIR / "stroke_scaler.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
num_cols = joblib.load(MODELS_DIR / "num_cols.pkl")
risk_threshold = float(joblib.load(MODELS_DIR / "risk_threshold.pkl"))


RAG_PDF_SOURCES = {
    "stroke_signs": {
        "file_name": "stroke_signs_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Recognising The Signs Of A Stroke",
        "topic": "Stroke signs, F.A.S.T response, stroke risk factors",
    },
    "blood_pressure": {
        "file_name": "high_blood_pressure_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "High Blood Pressure: Healthy Eating Guide",
        "topic": "High blood pressure, hypertension, healthy eating",
    },
    "cholesterol": {
        "file_name": "cholesterol_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Cholesterol and Heart Disease",
        "topic": "Cholesterol and heart disease",
    },
    "physical_activity": {
        "file_name": "physical_activity_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Health Benefits of Exercise and Physical Activity",
        "topic": "Exercise and physical activity",
    },
    "diabetes_eating": {
        "file_name": "diabetes_eating_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Diabetes Healthy Eating Guidance",
        "topic": "Diabetes, glucose, low GI food, healthier eating",
    },
    "stroke_prevention": {
        "file_name": "stroke_prevention_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Stroke Prevention: Medications and Lifestyle Changes to Help Reduce Stroke Risk",
        "topic": "Stroke prevention and lifestyle changes",
    },
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages in one PDF."""
    reader = PdfReader(str(pdf_path))
    extracted_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            extracted_text += f"\n--- Page {page_number} ---\n"
            extracted_text += page_text

    return extracted_text


def load_rag_documents() -> list:
    """Load all configured RAG PDFs into memory."""
    rag_documents = []
    rag_folder = _resolve_rag_source_folder()

    for source_key, source_info in RAG_PDF_SOURCES.items():
        pdf_path = rag_folder / source_info["file_name"]

        if not pdf_path.exists():
            continue

        pdf_text = extract_text_from_pdf(pdf_path)

        rag_documents.append(
            {
                "source_key": source_key,
                "source_name": source_info["source_name"],
                "source_title": source_info["source_title"],
                "topic": source_info["topic"],
                "source_file": source_info["file_name"],
                "text": pdf_text,
            }
        )

    return rag_documents


# Cache RAG docs at app startup for simplicity and speed.
rag_documents = load_rag_documents()


def retrieve_relevant_pdf_guidance(patient_data: dict) -> list:
    """Select relevant PDF guidance based on patient risk factors."""
    selected_source_keys = ["stroke_signs", "stroke_prevention"]

    if (
        patient_data["Blood_Pressure_Systolic"] >= 140
        or patient_data["Blood_Pressure_Diastolic"] >= 90
    ):
        selected_source_keys.append("blood_pressure")

    if patient_data["Cholesterol"] >= 200:
        selected_source_keys.append("cholesterol")

    if patient_data["Physical_Activity"] == 0:
        selected_source_keys.append("physical_activity")

    if patient_data["Glucose_Level"] >= 140 or patient_data["Diabetes"] == 1:
        selected_source_keys.append("diabetes_eating")

    if patient_data["BMI"] >= 25 and "diabetes_eating" not in selected_source_keys:
        selected_source_keys.append("diabetes_eating")

    selected_source_keys = list(dict.fromkeys(selected_source_keys))

    retrieved_guidance = []

    for document in rag_documents:
        if document["source_key"] in selected_source_keys:
            retrieved_guidance.append(
                {
                    "source_key": document["source_key"],
                    "source_name": document["source_name"],
                    "source_title": document["source_title"],
                    "source_file": document["source_file"],
                    "topic": document["topic"],
                    "retrieved_text": document["text"][:2500],
                }
            )

    return retrieved_guidance


def clean_generated_care_plan(text: str) -> str:
    """
    Clean Gemini output so markdown symbols do not appear on the website.
    """

    if not text:
        return ""

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        # Remove markdown headings such as ### Title
        line = re.sub(r"^#{1,6}\s*", "", line)

        # Remove markdown bold, italic, and code formatting
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = line.replace("`", "")

        # Remove horizontal rules like ***, ---, or ___
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Convert markdown bullet stars to simple dash bullets
        line = re.sub(r"^\*\s+", "- ", line)

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Remove too many blank lines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def build_rag_prompt(patient_data: dict, prediction_result: dict, retrieved_guidance: list) -> str:
    """Build Gemini prompt with patient profile, prediction output, and retrieved context."""
    retrieved_context_sections = []

    for item in retrieved_guidance:
        retrieved_context_sections.append(
            f"""
Source Title: {item['source_title']}
Source Name: {item['source_name']}
Source File: {item['source_file']}
Topic: {item['topic']}

Retrieved Text:
{item['retrieved_text']}
"""
        )

    prompt = f"""
You are a healthcare education assistant for a student AI project.

Your task:
Generate a personalized stroke risk care plan using the machine learning result and retrieved healthcare source context below.

Important safety rules:
- Do not diagnose stroke.
- Do not prescribe medication or dosage.
- Advise consulting a healthcare professional.
- Mention emergency stroke warning signs clearly.
- End with an educational disclaimer.
- Use simple patient-friendly language.
- Use plain text only.
- Do not use markdown symbols such as #, *, **, ---, or backticks.
- Use clear numbered section titles.
- Use simple dash bullet points only.

Binary values:
- 1 means Yes
- 0 means No

Machine Learning Model Output:
- Model used: Logistic Regression
- Predicted risk category: {prediction_result['risk_category']}
- Model-estimated risk probability: {prediction_result['risk_probability_percent']}%

Patient Health Profile:
- Age: {patient_data['Age']}
- Gender: {patient_data['Gender']}
- BMI: {patient_data['BMI']}
- Systolic Blood Pressure: {patient_data['Blood_Pressure_Systolic']}
- Diastolic Blood Pressure: {patient_data['Blood_Pressure_Diastolic']}
- Cholesterol: {patient_data['Cholesterol']}
- Glucose Level: {patient_data['Glucose_Level']}
- Smoking: {patient_data['Smoking']}
- Alcohol Intake: {patient_data['Alcohol_Intake']}
- Physical Activity: {patient_data['Physical_Activity']}
- Family History of Stroke: {patient_data['Family_History']}
- Heart Disease: {patient_data['Heart_Disease']}
- Diabetes: {patient_data['Diabetes']}

Retrieved PDF Source Context:
{chr(10).join(retrieved_context_sections)}

Generate the care plan with this exact structure:

1. Risk Summary
2. Sources Retrieved
3. Key Health Concerns
4. Recommended Daily Goals
5. Lifestyle Guidance
6. Monitoring Advice
7. When to Seek Medical Help
8. Disclaimer

Emergency warning signs to mention:
face drooping, arm weakness, speech difficulty, sudden confusion, sudden vision problems, sudden severe headache.
"""

    return prompt


def generate_care_plan_with_gemini(prompt: str) -> str:
    """Generate care plan using Gemini model and API key from .env."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return (
            "Gemini API key is not configured. Please add GEMINI_API_KEY in .env.\n\n"
            "Educational disclaimer: This output is for educational support only and does not replace advice from a qualified healthcare professional."
        )

    if genai is None:
        return (
            "google-generativeai package is not available. Install dependencies from requirements.txt.\n\n"
            "Educational disclaimer: This output is for educational support only and does not replace advice from a qualified healthcare professional."
        )

    try:
        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = genai_model.generate_content(prompt)
        return clean_generated_care_plan(response.text)

    except Exception as exc:
        return (
            "Care plan generation is currently unavailable.\n"
            f"Technical note: {exc}\n\n"
            "Educational disclaimer: This output is for educational support only and does not replace advice from a qualified healthcare professional."
        )


def build_daily_goals(patient_data: dict) -> list:
    """Create rule-based daily goals from patient risk factors."""
    goals = []

    if (
        patient_data["Blood_Pressure_Systolic"] >= 140
        or patient_data["Blood_Pressure_Diastolic"] >= 90
    ):
        goals.append("Check blood pressure and choose lower-salt meals.")

    if patient_data["Glucose_Level"] >= 140 or patient_data["Diabetes"] == 1:
        goals.append("Follow glucose-friendly meals and reduce high-sugar foods.")

    if patient_data["Cholesterol"] >= 200:
        goals.append("Choose heart-healthier meals with less fried or fatty food.")

    if patient_data["Smoking"] == 1:
        goals.append("Reduce smoking triggers and seek support to quit.")

    if patient_data["Alcohol_Intake"] == 1:
        goals.append("Limit alcohol intake and increase water intake.")

    if patient_data["Physical_Activity"] == 0:
        goals.append("Do light walking or gentle movement for 10 to 20 minutes.")

    if patient_data["BMI"] >= 25:
        goals.append("Practice portion control and include regular gentle movement.")

    if patient_data["Heart_Disease"] == 1 or patient_data["Family_History"] == 1:
        goals.append("Plan or maintain regular medical follow-up.")

    if not goals:
        goals.append("Maintain healthy habits and continue regular monitoring.")

    return goals


def build_7_day_care_calendar(patient_data: dict) -> list:
    """Build a simple 7-day care calendar table for display."""
    goals = build_daily_goals(patient_data)
    today = datetime.today()
    calendar_rows = []

    for i in range(7):
        current_date = today + timedelta(days=i)
        main_goal = goals[i % len(goals)]

        calendar_rows.append(
            {
                "Date": current_date.strftime("%Y-%m-%d"),
                "Day": current_date.strftime("%A"),
                "Main Focus": main_goal,
                "Morning Action": "Review today's health goal.",
                "Afternoon Action": main_goal,
                "Evening Action": "Reflect on progress and prepare for tomorrow.",
            }
        )

    return calendar_rows


def parse_patient_form(form_data) -> dict:
    """Convert submitted form values into a typed patient_data dictionary."""
    return {
        "Age": int(form_data["Age"]),
        "Gender": form_data["Gender"],
        "BMI": float(form_data["BMI"]),
        "Blood_Pressure_Systolic": float(form_data["Blood_Pressure_Systolic"]),
        "Blood_Pressure_Diastolic": float(form_data["Blood_Pressure_Diastolic"]),
        "Cholesterol": float(form_data["Cholesterol"]),
        "Glucose_Level": float(form_data["Glucose_Level"]),
        "Smoking": int(form_data["Smoking"]),
        "Alcohol_Intake": int(form_data["Alcohol_Intake"]),
        "Physical_Activity": int(form_data["Physical_Activity"]),
        "Family_History": int(form_data["Family_History"]),
        "Heart_Disease": int(form_data["Heart_Disease"]),
        "Diabetes": int(form_data["Diabetes"]),
    }


@app.route("/")
def home():
    return render_template("home.html", page_title="Home")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    prediction_result = None
    patient_data = session.get("patient_data")
    error_message = None

    if request.method == "POST":
        try:
            patient_data = parse_patient_form(request.form)

            # Keep model input in the exact trained feature order.
            model_input_df = pd.DataFrame([patient_data])[feature_cols].copy()

            # Encode Gender for model prediction.
            model_input_df["Gender"] = model_input_df["Gender"].map(
                {"Male": 1, "Female": 0}
            )

            # Scale only numeric columns.
            model_input_df[num_cols] = scaler.transform(model_input_df[num_cols])
            model_input_df = model_input_df.apply(pd.to_numeric, errors="coerce")

            # Predict stroke probability using Logistic Regression.
            risk_probability = float(model.predict_proba(model_input_df)[:, 1][0])
            risk_category = "High risk" if risk_probability >= risk_threshold else "Low risk"

            prediction_result = {
                "risk_category": risk_category,
                "risk_probability_percent": round(risk_probability * 100, 2),
                "threshold_used": float(risk_threshold),
            }

            # Store in session for care-plan route.
            session["patient_data"] = patient_data
            session["prediction_result"] = prediction_result

        except Exception as exc:
            error_message = f"Could not process prediction: {exc}"

    # On GET, show latest session result if available.
    if prediction_result is None:
        prediction_result = session.get("prediction_result")

    return render_template(
        "prediction.html",
        page_title="Prediction",
        prediction_result=prediction_result,
        patient_data=patient_data,
        error_message=error_message,
    )


@app.route("/care-plan")
def care_plan():
    patient_data = session.get("patient_data")
    prediction_result = session.get("prediction_result")

    if not patient_data or not prediction_result:
        return redirect(url_for("prediction"))

    retrieved_guidance = retrieve_relevant_pdf_guidance(patient_data)
    retrieved_source_titles = [item["source_title"] for item in retrieved_guidance]

    rag_prompt = build_rag_prompt(patient_data, prediction_result, retrieved_guidance)
    generated_care_plan = generate_care_plan_with_gemini(rag_prompt)

    care_calendar = build_7_day_care_calendar(patient_data)

    return render_template(
        "care_plan.html",
        page_title="Care Plan",
        prediction_result=prediction_result,
        patient_data=patient_data,
        retrieved_source_titles=retrieved_source_titles,
        generated_care_plan=generated_care_plan,
        care_calendar=care_calendar,
    )


if __name__ == "__main__":
    app.run(debug=True)