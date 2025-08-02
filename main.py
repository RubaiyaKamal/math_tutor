import streamlit as st
import google.generativeai as genai
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re

# ---------- Setup ----------

# Path to Tesseract (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Gemini
genai.configure(api_key="AIzaSyCeVJTQondc1QP1rOXCGXLeRQa5mlhLkRI")  # Replace with your actual key
model = genai.GenerativeModel("gemini-2.0-flash")  # or "gemini-1.5-pro" if available

# ---------- OCR + AI Functions ----------

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

def extract_text_from_image(image: Image.Image) -> str:
    processed_img = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    return text.strip()

def is_math_question(line: str) -> bool:
    return bool(re.search(r'\d.*[+\-×x*/=]', line))

def solve_question_with_gemini(question_text: str) -> str:
    prompt = f"""
You are a helpful AI math tutor specialized in GCSE-level (AQA/Edexcel) exams.
Solve the following question step by step with clear reasoning.

Question: {question_text}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error from Gemini API: {str(e)}"

# ---------- Streamlit UI ----------

st.set_page_config(page_title="MathMind – AI GCSE Solver", page_icon="📘")
st.title("📘 MathMind (Edexcel & AQA)")

st.markdown("**📖 Instantly solve GCSE math questions using AI. Enter text or upload a photo!**")

input_method = st.radio("Choose input type", ("Text Input", "Image Upload"))

# ---------- Text Input Mode ----------

if input_method == "Text Input":
    question = st.text_area("✍️ Enter your math question below:")
    if st.button("💡 Solve"):
        if question.strip():
            with st.spinner("Solving your question using Gemini..."):
                solution = solve_question_with_gemini(question)
            st.success("✅ Solution:")
            st.markdown(solution)
        else:
            st.warning("⚠️ Please enter a math question.")

# ---------- Image Upload Mode ----------

else:
    uploaded_file = st.file_uploader("📷 Upload an image with math questions", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Extract & Solve"):
            with st.spinner("Extracting text using OCR..."):
                extracted_text = extract_text_from_image(image)
            
            if not extracted_text:
                st.warning("⚠️ No text detected. Try another image or improve clarity.")
            else:
                st.subheader("📝 Extracted Text")
                st.code(extracted_text)

                lines = extracted_text.split("\n")
                questions = [line.strip()[:100] for line in lines if is_math_question(line.strip())]

                if questions:
                    st.success(f"✅ Found {len(questions)} potential question(s). Showing top 3.")
                    st.subheader("📘 AI-Powered Solutions")
                    for i, q in enumerate(questions[:3]):
                        with st.expander(f"Q{i+1}: {q}"):
                            solution = solve_question_with_gemini(q)
                            st.markdown(solution)
                else:
                    st.warning("⚠️ No math questions found. Try a clearer or more math-focused image.")
