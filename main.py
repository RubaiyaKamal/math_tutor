import streamlit as st
import google.generativeai as genai
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Gemini API config
genai.configure(api_key="AIzaSyCeVJTQondc1QP1rOXCGXLeRQa5mlhLkRI")


model = genai.GenerativeModel("gemini-2.0-flash")

def solve_question_with_gemini(question_text):
    prompt = f"Solve this GCSE-level math problem step by step: {question_text}"
    response = model.generate_content(prompt)
    return response.text

def extract_text_from_image(image: Image.Image) -> str:
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pytesseract.pytesseract.tesseract_cmd

    text = pytesseract.image_to_string(gray)
    return text.strip()

st.set_page_config(page_title="GCSE Math Solver", page_icon="📐")

st.title("📐 AI Math Solver for GCSE (Edexcel & AQA)")

input_method = st.radio("Choose input type", ("Text Input", "Image Upload"))

if input_method == "Text Input":
    question = st.text_area("Enter your math question here")
    if st.button("Solve"):
        if question.strip():
            with st.spinner("Solving..."):
                solution = solve_question_with_gemini(question)
            st.success("Solution:")
            st.markdown(solution)
        else:
            st.warning("Please enter a question.")

else:
    uploaded_file = st.file_uploader("Upload an image with math questions", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Extract & Solve"):
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)
                questions = [q.strip() for q in extracted_text.split("\n") if q.strip()]
                st.info(f"Detected {len(questions)} possible questions")
                
                st.subheader("Solutions")
                for i, question in enumerate(questions[:3]):  # Limit to 2–3 questions
                    st.markdown(f"**Q{i+1}:** {question}")
                    solution = solve_question_with_gemini(question)
                    st.markdown(f"**Answer:**\n{solution}")
