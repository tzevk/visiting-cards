import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import re
import google.generativeai as genai
import json
import cv2
import numpy as np
from datetime import datetime

# ===== CONFIG =====
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"  # Change if needed
GEMINI_API_KEY = "AIzaSyDdNtupejXi_F4rdueKb0I37uVC8vvuLkU"

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Streamlit UI
st.set_page_config(page_title="AI Visiting Card Scanner", layout="centered")
st.title("📇 AI Visiting Card Scanner (Phone Camera)")

# ===== CAMERA INPUT =====
photo = st.camera_input("📸 Scan visiting card using phone camera")

# ===== IMAGE PREPROCESSING =====
def preprocess_image(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

# ===== OCR FUNCTION =====
def extract_text(img):
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img, config=config)

# ===== GEMINI EXTRACTION =====
def extract_fields_with_gemini(text):
    prompt = f"""
    Extract name, phone, email, company from this visiting card text.
    Return ONLY valid JSON.

    Format:
    {{
      "name": "",
      "phone": "",
      "email": "",
      "company": ""
    }}

    Text:
    {text}
    """

    response = model.generate_content(prompt)
    return json.loads(response.text)

# ===== SAVE TO EXCEL =====
def save_to_excel(data):
    file = "cards.xlsx"
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        df = pd.read_excel(file)
    except:
        df = pd.DataFrame(columns=["name", "phone", "email", "company", "timestamp"])

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_excel(file, index=False)

# ===== MAIN LOGIC =====
if photo:
    img = Image.open(photo)
    st.image(img, caption="Captured Card")

    # Preprocess
    clean_img = preprocess_image(img)
    st.image(clean_img, caption="Processed Image")

    # OCR
    text = extract_text(clean_img)
    st.subheader("📜 OCR Output")
    st.text(text)

    # Gemini AI extraction
    if st.button("🤖 Extract Using AI"):
        with st.spinner("Gemini is analyzing..."):
            try:
                data = extract_fields_with_gemini(text)
            except Exception as e:
                st.error("Gemini failed. Using regex fallback.")
                phone = re.findall(r'\+?\d[\d\s\-]{9,}', text)
                email = re.findall(r'\S+@\S+', text)
                data = {
                    "name": text.split("\n")[0],
                    "phone": phone[0] if phone else "",
                    "email": email[0] if email else "",
                    "company": ""
                }

        # Editable fields
        st.subheader("✏️ Edit Extracted Data")
        name = st.text_input("Name", data.get("name", ""))
        phone = st.text_input("Phone", data.get("phone", ""))
        email = st.text_input("Email", data.get("email", ""))
        company = st.text_input("Company", data.get("company", ""))

        final_data = {
            "name": name,
            "phone": phone,
            "email": email,
            "company": company
        }

        if st.button("💾 Save to Excel"):
            save_to_excel(final_data)
            st.success("Saved to cards.xlsx!")

# ===== DOWNLOAD BUTTON =====
st.subheader("⬇️ Download Excel File")
try:
    with open("cards.xlsx", "rb") as f:
        st.download_button("Download cards.xlsx", f, file_name="cards.xlsx")
except:
    st.info("No Excel file yet.")