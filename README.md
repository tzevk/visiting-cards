# Visiting Card Scanner 📇

A Streamlit web app that uses your phone camera to scan visiting cards and extract contact information automatically.

## Features

- 📸 **Camera Capture**: Use your mobile phone camera to capture visiting cards
- 🔍 **OCR Extraction**: Extract text from images using Tesseract OCR
- 🤖 **AI Processing**: Use Google Gemini AI to identify and structure contact information
- ✏️ **Editable Fields**: Review and edit extracted information before saving
- 💾 **Excel Export**: Save contacts to an Excel file (cards.xlsx)
- 📥 **Download**: Download the Excel file with all saved contacts

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Tesseract OCR** installed:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Google Gemini API Key**: Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd visiting-cards
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the URL shown (usually http://localhost:8501)

3. For mobile access on the same network:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
   Then access via your computer's IP address on your phone

## Usage

1. **Enter API Key**: Add your Google Gemini API key in the sidebar
2. **Capture Image**: Use the camera input to take a photo of a visiting card
3. **Extract**: Click "Extract Information" to process the card
4. **Review**: Check and edit the extracted fields if needed
5. **Save**: Click "Save to Excel" to store the contact
6. **Download**: Use the download button to get your Excel file

## Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `cards.xlsx` - Generated Excel file with saved contacts (created after first save)

## Troubleshooting

### Tesseract not found
If you get a "tesseract not found" error, ensure Tesseract is installed and in your PATH. On macOS, you may need to set:
```python
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
```

### Camera not working
- Ensure you're accessing the app via HTTPS or localhost
- Grant camera permissions in your browser
- For mobile testing, you may need to run with `--server.address 0.0.0.0`

## License

MIT License
