# ANUVAAD Backend API v2.1 (Security Enhanced)
# Contributors: Yashwant Kumar Upadhyay, Vikrant Kumar, Medhabrata Konwar,
#               Debashis Bhuyan, Bhargab Jyoti Bhuyan, Yajant Kumar
# Guides: Anil Kumar Gupta (CDAC), Dr. Nabajyoti Medhi (Tezpur University)

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import logging
from functools import wraps
import os

# --- 1. Configuration & Security ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# In a real deployment, store this in an environment variable
# For the demo, we can set it here
API_KEY = os.environ.get("SARALVARTA_API_KEY", "your-super-secret-key-123")

# Authentication Decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for X-API-KEY header
        provided_key = request.headers.get('X-API-KEY')
        if provided_key and provided_key == API_KEY:
            return f(*args, **kwargs)
        else:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 401
    return decorated_function

# Device selection: GPU is highly recommended for IndicTrans2
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- 2. Model Manager Class ---
class AIModelManager:
    def __init__(self):
        # 1. Language Detection (IndicLID)
        logger.info("Initializing IndicLID (Detection)...")
        self.lid_model_name = "ai4bharat/IndicLID"
        self.lid_tokenizer = AutoTokenizer.from_pretrained(self.lid_model_name)
        self.lid_model = AutoModelForSequenceClassification.from_pretrained(self.lid_model_name).to(device)

        # 2. Translation (IndicTrans2)
        logger.info("Initializing IndicTrans2 (Translation)...")

        self.en_indic_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        self.indic_indic_name = "ai4bharat/indictrans2-indic-indic-dist-320M"

        self.en_indic_tokenizer = AutoTokenizer.from_pretrained(self.en_indic_name, trust_remote_code=True)
        self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(self.en_indic_name, trust_remote_code=True).to(device)

        self.indic_indic_tokenizer = AutoTokenizer.from_pretrained(self.indic_indic_name, trust_remote_code=True)
        self.indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(self.indic_indic_name, trust_remote_code=True).to(device)

        self.lang_map = {
            "as": "asm_Beng", "bn": "ben_Beng", "gu": "guj_Gujr",
            "hi": "hin_Deva", "kn": "kan_Knda", "ml": "mal_Mlym",
            "mr": "mar_Deva", "or": "ory_Orya", "pa": "pan_Guru",
            "ta": "tam_Taml", "te": "tel_Telu", "en": "eng_Latn"
        }

    def detect_lang(self, text):
        inputs = self.lid_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.lid_model(**inputs).logits

        pred_id = logits.argmax().item()
        label = self.lid_model.config.id2label[pred_id]
        code = label.split('_')[0]

        # Correction for Assamese
        if code == 'bn' and any(c in text for c in ['ৰ', 'ৱ']):
            return 'as'
        return code

    def translate(self, text, src_code, tgt_code):
        if src_code == "en":
            model = self.en_indic_model
            tokenizer = self.en_indic_tokenizer
        else:
            model = self.indic_indic_model
            tokenizer = self.indic_indic_tokenizer

        tgt_token = self.lang_map.get(tgt_code, "hin_Deva")
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_token] if hasattr(tokenizer, 'lang_code_to_id') else None,
                max_length=256
            )

        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Initialize Manager
ai_manager = AIModelManager()

# --- 3. API Endpoints ---

@app.route('/detect_language', methods=['POST'])
@require_api_key
def handle_detection():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        lang_code = ai_manager.detect_lang(data['text'])
        return jsonify({"language_code": lang_code, "status": "success"})
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
@require_api_key
def handle_translation():
    data = request.get_json()
    if not all(k in data for k in ("text", "source_lang", "target_lang")):
        return jsonify({"error": "Missing fields"}), 400

    try:
        result = ai_manager.translate(
            data['text'],
            data['source_lang'],
            data['target_lang']
        )
        return jsonify({"translated_text": result, "status": "success"})
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
