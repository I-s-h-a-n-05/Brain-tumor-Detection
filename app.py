import os
import io
import json
import requests
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from groq import Groq

from gradcam import generate_gradcam
from report import generate_report
from db import init_db, save_scan, get_scans, delete_scan, clear_all_scans, get_stats

app = Flask(__name__)

# ── DATABASE ──────────────────────────────────────────────────────────────────
init_db()

# ── MODEL DOWNLOAD (runs once on Render if model not present) ─────────────────
MODEL_PATH    = 'brain_tumor_model_v3.h5'
GDRIVE_FILE_ID = '1DKX80GGmB8vt5N12WDSu3X-Jjh4u56Pj'

def download_model_from_gdrive(file_id: str, dest: str):
    """Download a large file from Google Drive, handling the virus-scan warning."""
    print(f"Downloading model from Google Drive → {dest} ...")
    session = requests.Session()
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = session.get(url, stream=True)

    # Google redirects large files through a confirmation page
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm={value}'
            response = session.get(url, stream=True)
            break

    # Also handle newer Google Drive confirmation via HTML token
    if 'text/html' in response.headers.get('Content-Type', ''):
        import re
        token_match = re.search(r'confirm=([0-9A-Za-z_\-]+)', response.text)
        if token_match:
            confirm = token_match.group(1)
            url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm}'
            response = session.get(url, stream=True)

    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f'\r  {pct:.1f}% ({downloaded/1e6:.1f} MB / {total/1e6:.1f} MB)',
                          end='', flush=True)
    print(f'\nModel downloaded successfully → {dest}')

if not os.path.exists(MODEL_PATH):
    download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)

# ── MODEL LOAD ────────────────────────────────────────────────────────────────
for model_path in [
    MODEL_PATH,
    'brain_tumor_model_finetuned.h5',
    'brain_tumor_model.keras',
]:
    if os.path.exists(model_path):
        print(f"Loading model: {model_path} ...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded: {model_path}\n")
        LOADED_MODEL = model_path
        break
else:
    raise FileNotFoundError("No model file found even after download attempt.")

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ── GROQ ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are NeuroScan AI Assistant, an expert in neuro-oncology and brain tumor medicine. You have deep, comprehensive knowledge of:

- Brain tumor types: Glioma (GBM, astrocytoma, oligodendroglioma), Meningioma, Pituitary adenomas, Medulloblastoma, and all other CNS tumors
- WHO classification of CNS tumors (2021 edition) and tumor grading (Grade I-IV)
- MRI imaging: T1, T2, FLAIR, contrast enhancement patterns for each tumor type
- Symptoms, clinical presentation, and diagnosis workflows
- Treatment: surgery (craniotomy, stereotactic biopsy, awake surgery), radiotherapy (WBRT, SRS, Gamma Knife), chemotherapy (Temozolomide, Bevacizumab), immunotherapy, targeted therapy
- Prognosis and survival statistics for each tumor type
- Molecular markers: IDH1/2 mutation, MGMT methylation, EGFR amplification, 1p/19q codeletion, TERT promoter mutation
- The NeuroScan model: EfficientNetB0 transfer learning, 5712 MRI scans, 97.06% validation accuracy, 4 classes

Your communication style:
- Speak like a knowledgeable, warm doctor explaining to a patient or curious student
- Be clear, genuinely detailed, and helpful
- Use medical terms but always explain them
- Be honest but compassionate about prognosis topics
- Keep responses conversational — typically 3-6 sentences unless more detail is needed
- Always end advice about specific symptoms with a reminder to consult a qualified neurosurgeon or oncologist
- Never refuse brain tumor topics — this is your speciality
- If the question is unrelated to brain tumor or neuroscience, politely explain that you specialise in brain tumors"""


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model_info')
def model_info():
    return jsonify({'model': LOADED_MODEL})


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        file_bytes = request.files['file'].read()
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        return jsonify({
            'class':          CLASS_NAMES[class_idx],
            'confidence':     confidence,
            'low_confidence': confidence < 0.70,
            'probabilities':  {CLASS_NAMES[i]: float(preds[i]) for i in range(4)},
            'model_used':     LOADED_MODEL,
        })
    except Exception as e:
        print("PREDICT ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/gradcam', methods=['POST'])
def gradcam():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        class_index = request.form.get('class_index', None)
        file_bytes  = request.files['file'].read()
        pil_image   = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        if class_index is None:
            arr = np.array(pil_image.resize((224, 224)), dtype=np.float32)
            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr)[0]
            class_index = int(np.argmax(preds))
        else:
            class_index = int(class_index)
        heatmap_b64 = generate_gradcam(model, pil_image, class_index)
        return jsonify({
            'heatmap':     heatmap_b64,
            'class_index': class_index,
            'class_name':  CLASS_NAMES[class_index]
        })
    except Exception as e:
        print("GRADCAM ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/report', methods=['POST'])
def report():
    if 'file' not in request.files:
        return jsonify({'error': 'No scan image provided'}), 400
    try:
        file_bytes      = request.files['file'].read()
        pil_img         = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        result          = json.loads(request.form.get('result', '{}'))
        predicted_class = result.get('class', 'Unknown')
        confidence      = float(result.get('confidence', 0.0))
        probabilities   = result.get('probabilities', {c: 0.0 for c in CLASS_NAMES})
        gradcam_b64     = request.form.get('gradcam', None)

        pdf_bytes = generate_report(
            scan_image_pil=pil_img,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            gradcam_b64=gradcam_b64,
        )
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'NeuroScan_Report_{predicted_class.replace(" ", "_")}_{ts}.pdf'
        return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf',
                         as_attachment=True, download_name=filename)
    except Exception as e:
        print("REPORT ERROR:", str(e))
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── HISTORY ENDPOINTS ─────────────────────────────────────────────────────────
@app.route('/history/save', methods=['POST'])
def history_save():
    """
    Save a scan result to SQLite.
    Expects JSON: {label, class, cls, confidence, severity, reco,
                   probabilities, thumb, gradcam?}
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400
    try:
        scan_id = save_scan(
            label         = data.get('label', ''),
            cls_name      = data.get('class', ''),
            cls_style     = data.get('cls', ''),
            confidence    = float(data.get('confidence', 0)),
            severity      = data.get('severity', ''),
            reco          = data.get('reco', ''),
            probabilities = data.get('probabilities', {}),
            thumb_b64     = data.get('thumb', ''),
            gradcam_b64   = data.get('gradcam', None),
        )
        return jsonify({'id': scan_id, 'status': 'saved'})
    except Exception as e:
        print("HISTORY SAVE ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/history/get', methods=['GET'])
def history_get():
    """Return last 50 scans + stats."""
    try:
        scans = get_scans(limit=50)
        stats = get_stats()
        return jsonify({'scans': scans, 'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history/delete/<int:scan_id>', methods=['DELETE'])
def history_delete(scan_id):
    """Delete a single scan."""
    try:
        delete_scan(scan_id)
        return jsonify({'status': 'deleted', 'id': scan_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history/clear', methods=['DELETE'])
def history_clear():
    """Delete all scan history."""
    try:
        clear_all_scans()
        return jsonify({'status': 'cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history/stats', methods=['GET'])
def history_stats():
    try:
        return jsonify(get_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── CHAT ──────────────────────────────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({'error': 'No messages provided'}), 400
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + data['messages']
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
        )
        return jsonify({'reply': response.choices[0].message.content})
    except Exception as e:
        print("CHAT ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)