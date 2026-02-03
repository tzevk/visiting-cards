import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
import cv2
import re
import os
from datetime import datetime

# Bypass PaddleOCR connectivity check for faster startup
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR

# ================= CONFIG =================
st.set_page_config(page_title="Business Card Scanner", layout="centered")

@st.cache_resource
def get_ocr():
    # Use lightweight settings for faster inference
    return PaddleOCR(
        lang='en',
        use_angle_cls=False  # Disable angle classification for speed
    )

ocr = get_ocr()


# ================= 1. IMAGE PREPROCESSING =================
def preprocess_image(img):
    """
    Fast preprocessing for OCR.
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Resize to optimal size (smaller = faster)
    height, width = img_cv.shape[:2]    
    target_width = 1200
    if width > target_width:
        scale = target_width / width
        img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Simple contrast enhancement
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return img_cv, processed


def preprocess_for_difficult_cards(img_cv):
    """
    Fallback for difficult cards - only used if first pass fails.
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)
    return cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(best, cv2.COLOR_GRAY2BGR)


# ================= 2. OCR ENGINE =================
def clean_ocr_text(text):
    """
    Clean common OCR errors and normalize text.
    """
    if not text:
        return text
    
    # Common OCR substitutions
    replacements = {
        '|': 'l',
        '0': 'O',  # Will be context-dependent
        '1': 'l',  # Will be context-dependent  
        '\\': '',
        '  ': ' ',
    }
    
    cleaned = text.strip()
    
    # Remove stray special characters at start/end
    cleaned = re.sub(r'^[^a-zA-Z0-9@+]+', '', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9.]+$', '', cleaned)
    
    return cleaned


def run_ocr(img_array):
    """
    Run PaddleOCR on preprocessed image.
    Returns list of text lines with position info.
    """
    # Ensure we pass a single numpy array (not a tuple or list)
    if isinstance(img_array, tuple):
        img_array = img_array[0]
    
    result = ocr.predict(img_array)
    
    lines = []
    if result and len(result) > 0:
        for item in result:
            if 'rec_texts' in item and 'dt_polys' in item:
                texts = item['rec_texts']
                polys = item['dt_polys']
                scores = item.get('rec_scores', [1.0] * len(texts))
                
                for text, poly, score in zip(texts, polys, scores):
                    cleaned_text = clean_ocr_text(text)
                    if cleaned_text and len(cleaned_text) > 1 and score > 0.35:
                        poly = np.array(poly)
                        y_pos = (np.min(poly[:, 1]) + np.max(poly[:, 1])) / 2
                        x_pos = (np.min(poly[:, 0]) + np.max(poly[:, 0])) / 2
                        text_height = np.max(poly[:, 1]) - np.min(poly[:, 1])
                        text_width = np.max(poly[:, 0]) - np.min(poly[:, 0])
                        lines.append({
                            'text': cleaned_text,
                            'y_pos': y_pos,
                            'x_pos': x_pos,
                            'height': text_height,
                            'width': text_width,
                            'confidence': score
                        })
    
    # Sort by vertical position (top to bottom)
    lines.sort(key=lambda x: x['y_pos'])
    
    return lines


def merge_ocr_results(lines1, lines2):
    """
    Merge OCR results from multiple preprocessing methods.
    Prefer higher confidence results and longer text.
    """
    merged = {}
    
    for line in lines1 + lines2:
        text_lower = line['text'].lower().strip()
        # Skip very short text
        if len(text_lower) < 2:
            continue
        
        # Prefer longer text or higher confidence
        if text_lower not in merged:
            merged[text_lower] = line
        else:
            existing = merged[text_lower]
            # Prefer higher confidence, or longer original text if similar confidence
            if line['confidence'] > existing['confidence'] + 0.1:
                merged[text_lower] = line
            elif len(line['text']) > len(existing['text']):
                merged[text_lower] = line
    
    result = list(merged.values())
    result.sort(key=lambda x: x['y_pos'])
    return result


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
        'designation': '',
        'phone': '',
        'email': '',
        'company': '',
        'address': '',
        'notes': '',
        'card_width': 0,
        'card_height': 0
    }
    
    # ===== EMAIL PATTERN =====
    # Clean common OCR mistakes for email
    all_text_email = all_text.replace(' @ ', '@').replace('@ ', '@').replace(' @', '@')
    all_text_email = all_text_email.replace(' . ', '.').replace('. ', '.').replace(' .', '.')
    all_text_no_space_email = all_text_no_space.replace(' ', '')
    
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, all_text_email) or re.search(email_pattern, all_text_no_space_email)
    if email_match:
        fields['email'] = email_match.group().lower()
    else:
        # Try to find email in individual lines with cleaning
        for item in ocr_lines:
            line = item['text'].replace(' ', '')
            email_match = re.search(email_pattern, line)
            if email_match:
                fields['email'] = email_match.group().lower()
                break
    
    # ===== PHONE PATTERN =====
    # First, look for phone in individual lines (more reliable)
    phone_candidates = []
    for item in ocr_lines:
        line = item['text']
        # Clean common OCR mistakes in phone numbers
        line_cleaned = line.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
        digits = re.sub(r'\D', '', line_cleaned)
        
        # Check if line has 10+ digits (phone number)
        if len(digits) >= 10:
            # Format the phone number
            if digits.startswith('91') and len(digits) >= 12:
                phone = '+91 ' + digits[2:12]
            elif digits.startswith('0') and len(digits) >= 11:
                phone = '+91 ' + digits[1:11]
            else:
                phone = '+91 ' + digits[:10]
            phone_candidates.append((phone, item.get('confidence', 0.5)))
    
    # Pick highest confidence phone
    if phone_candidates:
        phone_candidates.sort(key=lambda x: x[1], reverse=True)
        fields['phone'] = phone_candidates[0][0]
    
    # Fallback: search in combined text
    if not fields['phone']:
        # Clean OCR mistakes in combined text
        all_text_cleaned = all_text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
        phone_patterns = [
            r'\+91[\s-]?\d{5}[\s-]?\d{5}',  # +91 format
            r'\+91[\s-]?\d{10}',             # +91 continuous
            r'(?<!\d)\d{10}(?!\d)',          # 10 digits
            r'(?<!\d)\d{5}[\s-]\d{5}(?!\d)', # 5-5 format
            r'(?<!\d)\d{3}[\s-]\d{3}[\s-]\d{4}(?!\d)',  # 3-3-4 format
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, all_text_cleaned)
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
        'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'vp', 'avp', 'svp', 'evp',
        'manager', 'director', 'engineer', 'developer', 'analyst', 'consultant',
        'founder', 'co-founder', 'cofounder', 'president', 'chairman', 'partner',
        'head', 'lead', 'senior', 'junior', 'executive', 'officer', 'associate',
        'coordinator', 'specialist', 'architect', 'designer', 'administrator',
        'supervisor', 'team lead', 'tech lead', 'project', 'product', 'sales',
        'marketing', 'hr', 'human resource', 'finance', 'accounting', 'legal',
        'operations', 'business', 'strategy', 'chief', 'general', 'managing',
        'assistant', 'intern', 'trainee', 'advisor', 'member', 'representative'
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
    designation_lines = []
    
    # Calculate average y position for top/bottom detection
    if ocr_lines:
        all_y_positions = [item['y_pos'] for item in ocr_lines]
        mid_y = (min(all_y_positions) + max(all_y_positions)) / 2
    else:
        mid_y = 0
    
    for idx, item in enumerate(ocr_lines):
        text = item['text']
        text_lower = text.lower()
        y_pos = item['y_pos']
        text_height = item.get('height', 0)
        
        # Skip if it's email or phone
        if '@' in text or re.search(email_pattern, text):
            used_lines.add(idx)
            continue
        
        digits = re.sub(r'\D', '', text)
        if len(digits) >= 8:
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
        
        # Check for title/designation
        is_designation = any(kw in text_lower for kw in title_keywords)
        if is_designation:
            designation_lines.append((idx, text, y_pos))
            # Don't add to used_lines yet - might need for name detection
        
        # Almost everything else could be a name candidate
        # Be very permissive here
        words = text.split()
        if len(text) >= 2 and len(words) <= 6:
            has_url = 'www' in text_lower or 'http' in text_lower or '.com' in text_lower
            
            if not has_url:
                is_top_half = y_pos <= mid_y
                name_candidates.append((idx, text, y_pos, text_height, is_top_half, is_designation))
    
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
    
    # ===== EXTRACT NAME (Simplified approach) =====
    # Strategy: Name is usually the LARGEST text or the TOPMOST non-email/phone text
    
    # Filter out obvious non-names from candidates
    filtered_candidates = []
    for (idx, text, y_pos, height, is_top, is_desig) in name_candidates:
        text_lower = text.lower()
        
        # Skip if it's the company
        if fields['company'] and (text_lower in fields['company'].lower() or fields['company'].lower() in text_lower):
            continue
        
        # Skip designations for primary name detection
        if is_desig:
            continue
            
        # Skip if mostly digits
        digit_count = sum(c.isdigit() for c in text)
        if digit_count > len(text) * 0.3:
            continue
        
        filtered_candidates.append((idx, text, y_pos, height, is_top))
    
    if filtered_candidates:
        # Method 1: Find the largest text in top half
        top_half = [c for c in filtered_candidates if c[4]]
        if top_half:
            # Sort by height (largest first), then by position (topmost)
            top_half.sort(key=lambda x: (-x[3], x[2]))
            fields['name'] = top_half[0][1].title()
        else:
            # No top-half candidates, use largest text overall
            filtered_candidates.sort(key=lambda x: (-x[3], x[2]))
            fields['name'] = filtered_candidates[0][1].title()
    
    # If still no name, look near designation
    if not fields['name'] and designation_lines:
        designation_lines.sort(key=lambda x: x[2])
        designation_y = designation_lines[0][2]
        designation_idx = designation_lines[0][0]
        
        # Find text closest to (preferably above) designation
        best_candidate = None
        best_distance = float('inf')
        
        for (idx, text, y_pos, height, is_top, is_desig) in name_candidates:
            if is_desig:  # Skip the designation itself
                continue
            if fields['company'] and text.lower() in fields['company'].lower():
                continue
                
            distance = abs(y_pos - designation_y)
            is_above = y_pos < designation_y
            
            # Prefer text above designation
            if is_above and distance < best_distance:
                best_distance = distance
                best_candidate = text
            elif not best_candidate and distance < best_distance:
                best_distance = distance
                best_candidate = text
        
        if best_candidate:
            fields['name'] = best_candidate.title()
    
    # Final fallback: just pick the first line that's not email/phone/company/address
    if not fields['name']:
        for item in ocr_lines:
            text = item['text']
            text_lower = text.lower()
            
            if '@' in text or len(re.sub(r'\D', '', text)) >= 8:
                continue
            if any(kw in text_lower for kw in company_keywords + address_keywords):
                continue
            if fields['company'] and text_lower in fields['company'].lower():
                continue
            if len(text) >= 2:
                fields['name'] = text.title()
                break
    
    # ===== EXTRACT DESIGNATION =====
    if designation_lines:
        # Sort by y position - designation is often right below name
        designation_lines.sort(key=lambda x: x[2])
        # Combine all designation parts (sometimes split across lines)
        designation_texts = [text for (_, text, _) in designation_lines]
        fields['designation'] = ' | '.join(designation_texts) if len(designation_texts) > 1 else designation_texts[0]
    
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
        df = pd.DataFrame(columns=["name", "designation", "phone", "email", "company", "address", "notes", "card_width", "card_height", "timestamp"])

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_excel(file, index=False)


# ================= STREAMLIT UI =================
st.title("📇 Business Card Scanner")
st.markdown("**Easily digitize your business cards** — Take a photo or upload an image to extract contact details automatically.")

st.divider()

# Input options with friendly descriptions
st.subheader("Step 1: Add a Business Card")

tab1, tab2 = st.tabs(["📷 Take Photo", "📁 Upload Image"])

with tab1:
    st.markdown("*Position the card flat with good lighting for best results*")
    photo = st.camera_input("Take a picture of the business card", label_visibility="collapsed")

with tab2:
    st.markdown("*Supported formats: PNG, JPG, JPEG*")
    uploaded = st.file_uploader("Select an image file", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

# Get image from either source
img = None
if photo:
    img = Image.open(photo)
elif uploaded:
    img = Image.open(uploaded)

if img:
    st.divider()
    st.subheader("Step 2: Extract Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Your Business Card", use_container_width=True)
    
    if st.button("🔍 Scan Card", type="primary", use_container_width=True, help="Click to automatically extract contact information from the card"):
        with st.spinner("Reading the card..."):
            try:
                # Preprocess image
                img_cv, processed = preprocess_image(img)
                
                # Single OCR pass on enhanced image (fast)
                ocr_lines = run_ocr(processed)
                
                # Only try fallback if we got very little text
                if len(ocr_lines) < 2:
                    enhanced = preprocess_for_difficult_cards(img_cv)
                    ocr_lines = run_ocr(enhanced)
                
                if ocr_lines:
                    # Parse extracted text
                    data = parse_text(ocr_lines)
                    
                    # Capture card dimensions
                    height, width = img_cv.shape[:2]
                    data['card_width'] = width
                    data['card_height'] = height
                    
                    st.session_state['extracted_data'] = data
                    st.success("✅ Contact details extracted! Please review below.")
                    
                    # Hidden technical details for advanced users
                    with st.expander("🔧 Technical Details (Advanced)"):
                        st.caption("Raw text detected from the card:")
                        for item in ocr_lines:
                            confidence_pct = int(item['confidence'] * 100)
                            st.text(f"• {item['text']} ({confidence_pct}% confidence)")
                else:
                    st.error("😕 Couldn't read the card. Please try again with:")
                    st.markdown("""
                    - Better lighting
                    - Card placed flat on a plain background
                    - A clearer, higher resolution image
                    """)
                    
            except Exception as e:
                st.error("😕 Something went wrong. Please try with a different image.")
                with st.expander("🔧 Error Details (Advanced)"):
                    import traceback
                    st.code(traceback.format_exc())

# Editable form
if 'extracted_data' in st.session_state:
    st.divider()
    st.subheader("Step 3: Review & Save")
    st.markdown("*Check the details below and correct any mistakes before saving*")
    
    data = st.session_state['extracted_data']
    
    # Store dimensions but don't show them prominently
    card_width = data.get("card_width", 0)
    card_height = data.get("card_height", 0)
    
    # Main contact info
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Name", data.get("name", ""), placeholder="e.g., John Smith")
        designation = st.text_input("💼 Job Title", data.get("designation", ""), placeholder="e.g., Marketing Manager")
        email = st.text_input("📧 Email", data.get("email", ""), placeholder="e.g., john@company.com")
    with col2:
        phone = st.text_input("📱 Phone", data.get("phone", ""), placeholder="e.g., +91 98765 43210")
        company = st.text_input("🏢 Company", data.get("company", ""), placeholder="e.g., Acme Corp")
    
    address = st.text_area("📍 Address", data.get("address", ""), height=80, placeholder="Full office address...")
    notes = st.text_area("📝 Notes", data.get("notes", ""), height=80, placeholder="Where did you meet? Any follow-up reminders?")

    st.markdown("")  # Add some spacing
    
    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if st.button("💾 Save Contact", type="primary", use_container_width=True):
            if not name.strip():
                st.warning("Please enter a name before saving.")
            else:
                final_data = {
                    "name": name, "designation": designation, "phone": phone, "email": email,
                    "company": company, "address": address, "notes": notes,
                    "card_width": card_width, "card_height": card_height
                }
                save_to_excel(final_data)
                st.success("🎉 Contact saved successfully!")
                st.balloons()
                del st.session_state['extracted_data']
                st.rerun()
    with col_clear:
        if st.button("🗑️ Discard", use_container_width=True, help="Clear the form without saving"):
            del st.session_state['extracted_data']
            st.rerun()

# Download section
st.divider()
st.subheader("📋 Your Contacts")

try:
    df = pd.read_excel("cards.xlsx")
    
    # Show count with friendly message
    contact_count = len(df)
    if contact_count == 1:
        st.success(f"You have **{contact_count} contact** saved")
    else:
        st.success(f"You have **{contact_count} contacts** saved")
    
    # Display table with user-friendly column names
    display_df = df.copy()
    display_columns = {
        'name': 'Name',
        'designation': 'Job Title', 
        'phone': 'Phone',
        'email': 'Email',
        'company': 'Company',
        'address': 'Address',
        'notes': 'Notes',
        'timestamp': 'Added On'
    }
    # Only rename columns that exist
    display_df = display_df.rename(columns={k: v for k, v in display_columns.items() if k in display_df.columns})
    
    # Hide technical columns from display
    columns_to_hide = ['card_width', 'card_height']
    for col in columns_to_hide:
        if col in display_df.columns:
            display_df = display_df.drop(columns=[col])
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button with friendly label
    st.markdown("")
    with open("cards.xlsx", "rb") as f:
        st.download_button(
            "📥 Download All Contacts (Excel)",
            f,
            file_name="my_contacts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Download all your saved contacts as an Excel file"
        )
except FileNotFoundError:
    st.info("👋 No contacts yet! Scan your first business card above to get started.")

# Footer
st.divider()
st.caption("💡 **Tip:** For best results, take photos in good lighting with the card flat on a plain background.")
