import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
import cv2
import re
from datetime import datetime
from paddleocr import PaddleOCR

# ================= CONFIG =================
st.set_page_config(page_title="Visiting Card Scanner", layout="centered")

@st.cache_resource
def get_ocr():
    return PaddleOCR(use_textline_orientation=True, lang='en')

ocr = get_ocr()


# ================= 1. IMAGE PREPROCESSING =================
def preprocess_image(img):
    """
    Preprocess image for better OCR accuracy.
    - Convert to grayscale
    - Denoise
    - Increase contrast
    - Binarize (adaptive threshold)
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Resize if too small
    height, width = img_cv.shape[:2]
    if width < 1000:
        scale = 1000 / width
        img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # Adaptive threshold for binarization
    binary = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to RGB for OCR
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return processed, img_cv


# ================= 2. OCR ENGINE =================
def run_ocr(img_array):
    """
    Run PaddleOCR on preprocessed image.
    Returns list of text lines with position info.
    """
    result = ocr.predict(img_array)
    
    lines = []
    if result and len(result) > 0:
        for item in result:
            if 'rec_texts' in item and 'dt_polys' in item:
                texts = item['rec_texts']
                polys = item['dt_polys']
                scores = item.get('rec_scores', [1.0] * len(texts))
                
                for text, poly, score in zip(texts, polys, scores):
                    if text and text.strip() and score > 0.5:
                        poly = np.array(poly)
                        y_pos = (np.min(poly[:, 1]) + np.max(poly[:, 1])) / 2
                        lines.append({
                            'text': text.strip(),
                            'y_pos': y_pos,
                            'confidence': score
                        })
    
    # Sort by vertical position (top to bottom)
    lines.sort(key=lambda x: x['y_pos'])
    
    return lines


# ================= 3. TEXT PARSING =================
def parse_text(ocr_lines):
    """
    Parse OCR text using regex patterns to extract structured data.
    """
    # Combine all text
    all_text = ' '.join([item['text'] for item in ocr_lines])
    all_text_no_space = ''.join([item['text'] for item in ocr_lines])
    
    # Initialize fields
    fields = {
        'name': '',
        'phone': '',
        'email': '',
        'company': '',
        'address': ''
    }
    
    # ===== EMAIL PATTERN =====
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, all_text) or re.search(email_pattern, all_text_no_space)
    if email_match:
        fields['email'] = email_match.group().lower()
    
    # ===== PHONE PATTERN =====
    # First, look for phone in individual lines (more reliable)
    for item in ocr_lines:
        line = item['text']
        digits = re.sub(r'\D', '', line)
        
        # Check if line has 10+ digits (phone number)
        if len(digits) >= 10:
            # Format the phone number
            if digits.startswith('91') and len(digits) >= 12:
                fields['phone'] = '+91 ' + digits[2:12]
            else:
                fields['phone'] = '+91 ' + digits[:10]
            break
    
    # Fallback: search in combined text
    if not fields['phone']:
        phone_patterns = [
            r'\+91[\s-]?\d{5}[\s-]?\d{5}',  # +91 format
            r'\+91[\s-]?\d{10}',             # +91 continuous
            r'(?<!\d)\d{10}(?!\d)',          # 10 digits
            r'(?<!\d)\d{5}[\s-]\d{5}(?!\d)', # 5-5 format
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, all_text_no_space)
            if phone_match:
                phone = re.sub(r'\D', '', phone_match.group())
                if len(phone) >= 10:
                    if phone.startswith('91') and len(phone) >= 12:
                        fields['phone'] = '+91 ' + phone[2:12]
                    else:
                        fields['phone'] = '+91 ' + phone[:10]
                    break
    
    # ===== COMPANY FROM EMAIL DOMAIN =====
    company_from_email = ''
    if fields['email']:
        domain_match = re.search(r'@([^.]+)', fields['email'])
        if domain_match:
            domain = domain_match.group(1)
            # Expand concatenated words: "sterlingandwilson" -> "Sterling And Wilson"
            domain = re.sub(r'(and|&)', r' \1 ', domain, flags=re.IGNORECASE)
            domain = re.sub(r'[-_]', ' ', domain)
            domain = ' '.join(domain.split())
            company_from_email = domain.title()
    
    # ===== COMPANY INDICATORS =====
    company_keywords = [
        'pvt', 'ltd', 'llc', 'inc', 'corp', 'limited', 'private',
        'technologies', 'solutions', 'services', 'enterprises', 
        'group', 'industries', 'consulting', 'software'
    ]
    
    # ===== TITLE/DESIGNATION KEYWORDS =====
    title_keywords = [
        'ceo', 'cto', 'cfo', 'coo', 'manager', 'director', 'engineer',
        'developer', 'analyst', 'consultant', 'founder', 'president',
        'chairman', 'partner', 'head', 'lead', 'senior', 'junior',
        'executive', 'officer', 'associate', 'coordinator', 'specialist'
    ]
    
    # ===== ADDRESS KEYWORDS =====
    address_keywords = [
        'road', 'street', 'lane', 'floor', 'block', 'sector', 'plot',
        'building', 'tower', 'office', 'nagar', 'colony', 'apartment',
        'mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad', 'pune',
        'kolkata', 'india', 'maharashtra', 'karnataka', 'pincode', 'pin'
    ]
    
    # ===== CATEGORIZE EACH LINE =====
    used_lines = set()
    company_lines = []
    name_candidates = []
    address_lines = []
    
    for idx, item in enumerate(ocr_lines):
        text = item['text']
        text_lower = text.lower()
        y_pos = item['y_pos']
        
        # Skip if it's email or phone
        if '@' in text or re.search(email_pattern, text):
            used_lines.add(idx)
            continue
        
        digits = re.sub(r'\D', '', text)
        if len(digits) >= 10:
            used_lines.add(idx)
            continue
        
        # Check for company
        if any(kw in text_lower for kw in company_keywords):
            company_lines.append((idx, text, y_pos))
            used_lines.add(idx)
            continue
        
        # Check for address
        if any(kw in text_lower for kw in address_keywords):
            address_lines.append((idx, text))
            used_lines.add(idx)
            continue
        
        # Check for title (skip for name)
        if any(kw in text_lower for kw in title_keywords):
            used_lines.add(idx)
            continue
        
        # Potential name: mostly letters, 1-4 words, top half of card
        words = text.split()
        if 1 <= len(words) <= 4:
            letter_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
            if letter_ratio > 0.8 and not re.search(r'\d', text):
                name_candidates.append((idx, text, y_pos))
    
    # ===== EXTRACT COMPANY =====
    if company_lines:
        # Sort by position, get first company line
        company_lines.sort(key=lambda x: x[2])
        main_idx, main_text, _ = company_lines[0]
        
        # Look for adjacent lines that might be part of company name
        company_parts = []
        
        # Check line before
        if main_idx > 0 and (main_idx - 1) not in used_lines:
            prev_text = ocr_lines[main_idx - 1]['text']
            prev_lower = prev_text.lower()
            # Include if it matches email domain or has "and/&"
            if company_from_email:
                domain_words = company_from_email.lower().split()
                if any(dw in prev_lower for dw in domain_words if len(dw) > 2):
                    company_parts.append(prev_text)
            elif ' and ' in prev_lower or ' & ' in prev_lower:
                company_parts.append(prev_text)
        
        company_parts.append(main_text)
        
        # Check line after
        if main_idx < len(ocr_lines) - 1:
            next_idx = main_idx + 1
            next_text = ocr_lines[next_idx]['text']
            next_lower = next_text.lower()
            if any(kw in next_lower for kw in ['pvt', 'ltd', 'limited', 'private']):
                company_parts.append(next_text)
        
        fields['company'] = ' '.join(company_parts)
    
    # If no company found but have email domain
    if not fields['company'] and company_from_email:
        # Look for lines matching email domain
        for idx, item in enumerate(ocr_lines):
            if idx in used_lines:
                continue
            text_lower = item['text'].lower()
            domain_words = company_from_email.lower().split()
            if any(dw in text_lower for dw in domain_words if len(dw) > 2 and dw != 'and'):
                fields['company'] = item['text']
                break
        
        # Fallback to email domain
        if not fields['company']:
            fields['company'] = company_from_email
    
    # ===== EXTRACT NAME =====
    if name_candidates:
        # Prefer candidates from top of card
        name_candidates.sort(key=lambda x: x[2])  # Sort by y_pos
        
        # Filter out company name if found
        if fields['company']:
            name_candidates = [
                (i, t, y) for (i, t, y) in name_candidates 
                if t.lower() not in fields['company'].lower()
            ]
        
        if name_candidates:
            fields['name'] = name_candidates[0][1].title()
    
    # ===== EXTRACT ADDRESS =====
    # First use lines with address keywords
    if address_lines:
        address_lines.sort(key=lambda x: x[0])
        address_texts = [text for (_, text) in address_lines]
        fields['address'] = ', '.join(address_texts)
    
    # If no address found, look at bottom lines of the card
    if not fields['address']:
        # Get lines from bottom half that aren't phone/email/name/company
        bottom_lines = []
        for idx, item in enumerate(ocr_lines):
            text = item['text']
            text_lower = text.lower()
            
            # Skip if already used or is email/phone
            if idx in used_lines:
                continue
            if '@' in text:
                continue
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 10:
                continue
            
            # Skip if it's the name or company
            if fields['name'] and text.lower() in fields['name'].lower():
                continue
            if fields['company'] and text.lower() in fields['company'].lower():
                continue
            
            # Check if line looks like address (has numbers or commas, not too short)
            has_numbers = bool(re.search(r'\d', text))
            has_comma = ',' in text
            is_long = len(text) > 10
            
            # Address indicators (expanded list)
            addr_indicators = [
                'road', 'street', 'lane', 'floor', 'block', 'sector', 'plot',
                'building', 'tower', 'office', 'nagar', 'colony', 'apartment',
                'mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad', 'pune',
                'kolkata', 'india', 'maharashtra', 'karnataka', 'pincode', 'pin',
                'no.', 'near', 'opp', 'complex', 'park', 'garden', 'phase',
                'city', 'town', 'dist', 'state', 'cross', 'main', 'avenue'
            ]
            has_addr_word = any(aw in text_lower for aw in addr_indicators)
            
            if has_addr_word or has_numbers or has_comma or is_long:
                bottom_lines.append(text)
        
        if bottom_lines:
            fields['address'] = ', '.join(bottom_lines)
    
    return fields


# ================= 4. EXCEL STORAGE =================
def save_to_excel(data):
    """Save extracted data to Excel file."""
    file = "cards.xlsx"
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        df = pd.read_excel(file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["name", "phone", "email", "company", "address", "timestamp"])

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_excel(file, index=False)


# ================= STREAMLIT UI =================
st.title("Visiting Card Scanner")
st.caption("Scan business cards and save contact information")

# Input options
tab1, tab2 = st.tabs(["Camera", "Upload File"])

with tab1:
    photo = st.camera_input("Capture card image")

with tab2:
    uploaded = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

# Get image from either source
img = None
if photo:
    img = Image.open(photo)
elif uploaded:
    img = Image.open(uploaded)

if img:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original", use_container_width=True)
    
    if st.button("Extract Information", type="primary"):
        with st.spinner("Processing image..."):
            try:
                # Step 1: Preprocess
                processed, original_cv = preprocess_image(img)
                
                with col2:
                    st.image(processed, caption="Processed", use_container_width=True)
                
                # Step 2: OCR
                ocr_lines = run_ocr(original_cv)
                
                if not ocr_lines:
                    ocr_lines = run_ocr(processed)
                
                if ocr_lines:
                    # Show raw OCR
                    with st.expander("View Raw OCR Text"):
                        for item in ocr_lines:
                            st.text(f"{item['text']} ({item['confidence']:.2f})")
                    
                    # Step 3: Parse
                    data = parse_text(ocr_lines)
                    
                    st.session_state['extracted_data'] = data
                    st.success("Information extracted successfully")
                else:
                    st.error("No text detected. Please try with a clearer image.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Editable form
if 'extracted_data' in st.session_state:
    st.divider()
    st.subheader("Review & Edit")
    
    data = st.session_state['extracted_data']
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", data.get("name", ""))
        email = st.text_input("Email", data.get("email", ""))
    with col2:
        phone = st.text_input("Phone", data.get("phone", ""))
        company = st.text_input("Company", data.get("company", ""))
    
    address = st.text_area("Address", data.get("address", ""), height=80)

    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if st.button("Save Contact", type="primary", use_container_width=True):
            final_data = {
                "name": name, "phone": phone, "email": email,
                "company": company, "address": address
            }
            save_to_excel(final_data)
            st.success("Contact saved!")
            del st.session_state['extracted_data']
            st.rerun()
    with col_clear:
        if st.button("Clear", use_container_width=True):
            del st.session_state['extracted_data']
            st.rerun()

# Download section
st.divider()
st.subheader("Saved Contacts")

try:
    df = pd.read_excel("cards.xlsx")
    st.caption(f"Total: {len(df)} contacts")
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    with open("cards.xlsx", "rb") as f:
        st.download_button(
            "Download Excel", f,
            file_name="visiting_cards.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
except FileNotFoundError:
    st.caption("No contacts saved yet. Scan a card to get started.")
