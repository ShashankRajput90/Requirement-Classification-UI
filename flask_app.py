# flask_app.py
"""
Simplified & cleaned Flask app (Option B)
- Robust JSON serialization (NpEncoder)
- Safe model/eval imports with fallbacks
- Endpoints: single classify, batch classify, compare, analytics, downloads, clear history
- Use this file as a drop-in replacement for your previous Flask file
"""

from flask import Flask, render_template, request, session, send_file, Response, jsonify
import json
import io
import os
import time
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from collections import Counter
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.utils
import logging

# ----------------------
# Basic app config
# ----------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flask_app")

# ----------------------
# JSON encoder for numpy / pandas types
# ----------------------
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # numpy scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # pandas types
        try:
            import pandas as _pd
            if isinstance(obj, _pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, _pd.Timedelta):
                return str(obj)
        except Exception:
            pass
        return super().default(obj)

def json_response(payload, status=200):
    """Return a flask Response with JSON serialized using NpEncoder."""
    return Response(json.dumps(payload, cls=NpEncoder), status=status, mimetype="application/json")


# ----------------------
# Try importing real model & eval functions (from your project); fallback to stubs
# ----------------------
try:
    from code_integration import (
        classify_with_groq_deepseek,
        classify_with_groq,
        classify_with_gemini,
        classify_with_cohere,
        classify_with_claude,
        classify_with_mistral_local
    )
    from evaluation import extract_is_nfr, extract_nfr_type_multilabel
    logger.info("Imported real model functions and evaluation helpers.")
except Exception as e:
    logger.warning(f"Could not import model/eval modules: {e}. Using stub implementations.")

    # Stubs for model functions: they accept (story, technique) and return a string response.
    def _stub_response_for(story, technique):
        # Simple heuristic stub: if contains words like 'respond', 'encrypt', treat as NFR
        story_low = str(story).lower()
        if any(w in story_low for w in ["encrypt", "secure", "latency", "respond", "throughput", "performance", "availability", "scalability", "accessibility"]):
            return "Is NFR: Yes\nNFR Type: Security\nReason: Mentions security/performance attribute."
        else:
            return "Is NFR: No\nNFR Type: None\nReason: This is a functional requirement describing behavior."

    classify_with_groq_deepseek = lambda s, t: _stub_response_for(s, t)
    classify_with_groq = lambda s, t: _stub_response_for(s, t)
    classify_with_gemini = lambda s, t: _stub_response_for(s, t)
    classify_with_cohere = lambda s, t: _stub_response_for(s, t)
    classify_with_claude = lambda s, t: _stub_response_for(s, t)
    classify_with_mistral_local = lambda s, t: _stub_response_for(s, t)

    # Stubs for evaluation parsing
    def extract_is_nfr(response_text):
        # Try robustly to extract Yes/No; fall back to 'PARSE_ERROR'
        if not response_text:
            return "PARSE_ERROR"
        if isinstance(response_text, dict):
            # already parsed
            return response_text.get("Is_NFR") or response_text.get("is_nfr") or response_text.get("isNfr") or "PARSE_ERROR"
        rt = str(response_text)
        if "is nfr" in rt.lower() or "is_nfr" in rt.lower() or "isnfr" in rt.lower():
            m = None
            import re
            m = re.search(r"(is[_\s]?nfr)[:\s]*\s*(yes|no)", rt, re.I)
            if m:
                return "Yes" if m.group(2).lower().startswith("y") else "No"
        # fallback heuristics
        if any(w in rt.lower() for w in ["security", "performance", "accessibility", "usability", "availability", "scalability", "maintain"]):
            return "Yes"
        return "No"

    def extract_nfr_type_multilabel(response_text):
        # Try to pull "NFR Type: <...>"
        rt = str(response_text)
        import re
        m = re.search(r"NFR Type:\s*([^\n\r]+)", rt, re.I)
        if m:
            val = m.group(1).strip()
            # allow comma-separated
            if "," in val:
                return [v.strip() for v in val.split(",") if v.strip()]
            return val
        # fallback label
        for label in ["Security", "Performance", "Usability", "Reliability", "Scalability", "Maintainability", "Compatibility", "Accessibility"]:
            if label.lower() in rt.lower():
                return label
        return "Other"


# ----------------------
# Utility helpers
# ----------------------
def ensure_py(value):
    """Convert numpy / pandas types to native Python types recursively."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray, list, tuple, set)):
        return [ensure_py(v) for v in value]
    if isinstance(value, dict):
        return {k: ensure_py(v) for k, v in value.items()}
    try:
        import pandas as pd
        if isinstance(value, pd.Series):
            return [ensure_py(v) for v in value.tolist()]
        if isinstance(value, pd.DataFrame):
            return json.loads(value.to_json(orient="records"))
    except Exception:
        pass
    return value

def safe_cast_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def safe_cast_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ----------------------
# Classifier config (models/prompts/samples)
# ----------------------
class FlaskClassifier:
    def __init__(self):
        self.models = {
            'Gemini 2.5 Pro': classify_with_gemini,
            'Groq LLaMA 3.3': classify_with_groq,
            'Groq DeepSeek': classify_with_groq_deepseek,
            'Cohere Command R+': classify_with_cohere,
            'Claude 3 Haiku': classify_with_claude,
            'Mistral Local': classify_with_mistral_local
        }
        self.prompt_templates = {
            'Zero Shot': 'zeroshot',
            'Few Shot': 'fewshot',
            'Chain Of Thought': 'chainofthought'
        }
        self.sample_stories = [
            {'story': 'As a user, I want to log into the system using my email and password.', 'expected': 'FR', 'type': 'Authentication'},
            {'story': 'As a user, I want the system to respond within 2 seconds.', 'expected': 'NFR', 'type': 'Performance'},
            {'story': 'As a user, I want to search for products by name.', 'expected': 'FR', 'type': 'Search'},
            {'story': 'As a user, I want my data to be encrypted and secure.', 'expected': 'NFR', 'type': 'Security'}
        ]

classifier = FlaskClassifier()

# ----------------------
# Session init
# ----------------------
def init_session():
    if 'classification_history' not in session:
        session['classification_history'] = []
    if 'selected_model' not in session:
        session['selected_model'] = 'Gemini 2.5 Pro'
    if 'selected_prompt' not in session:
        session['selected_prompt'] = 'Few Shot'

@app.before_request
def _before_request():
    init_session()

# ----------------------
# Routes (render templates assumed to exist)
# ----------------------
@app.route('/')
def index():
    return render_template('index.html', models=list(classifier.models.keys()), prompts=list(classifier.prompt_templates.keys()),
                           selected_model=session.get('selected_model'), selected_prompt=session.get('selected_prompt'))

@app.route('/single_classification')
def single_classification():
    return render_template('single_classification.html', models=list(classifier.models.keys()),
                           prompts=list(classifier.prompt_templates.keys()),
                           selected_model=session.get('selected_model'), selected_prompt=session.get('selected_prompt'),
                           sample_stories=classifier.sample_stories)

@app.route('/batch_processing')
def batch_processing():
    return render_template('batch_processing.html', models=list(classifier.models.keys()),
                           prompts=list(classifier.prompt_templates.keys()),
                           selected_model=session.get('selected_model'), selected_prompt=session.get('selected_prompt'))

@app.route('/model_comparison')
def model_comparison():
    return render_template('model_comparison.html', models=list(classifier.models.keys()),
                           prompts=list(classifier.prompt_templates.keys()))

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', history=session.get('classification_history', []),
                           total_classifications=len(session.get('classification_history', [])))


# ----------------------
# API endpoints
# ----------------------
@app.route('/api/classify', methods=['POST'])
def api_classify():
    try:
        data = request.get_json(force=True)
        user_story = str(data.get('story', '')).strip()
        model_name = data.get('model', session.get('selected_model'))
        prompt_type = data.get('prompt', session.get('selected_prompt'))

        if not user_story:
            return json_response({'error': 'User story is required'}, status=400)

        session['selected_model'] = model_name
        session['selected_prompt'] = prompt_type
        session.modified = True

        model_fn = classifier.models.get(model_name)
        technique = classifier.prompt_templates.get(prompt_type, 'zeroshot')
        if not model_fn:
            return json_response({'error': f'Model {model_name} not available'}, status=400)

        start = time.time()
        raw = model_fn(user_story, technique)
        elapsed = round(time.time() - start, 3)

        binary_pred = extract_is_nfr(raw)
        type_pred = extract_nfr_type_multilabel(raw)
        # normalize outputs
        binary_pred = str(binary_pred)
        type_pred_out = type_pred if isinstance(type_pred, str) else ', '.join(type_pred) if isinstance(type_pred, (list, tuple)) else str(type_pred)

        confidence = 'High' if len(str(raw)) > 20 and 'Error' not in str(raw) else 'Medium'
        if 'PARSE_ERROR' in str(binary_pred):
            confidence = 'Low'

        result = {
            'classification': binary_pred,
            'nfr_type': type_pred_out,
            'processing_time': float(elapsed),
            'confidence': confidence,
            'raw_response': str(raw),
            'model': model_name,
            'prompt': prompt_type
        }

        # Add to session history
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'story': user_story,
            'story_preview': (user_story[:100] + '...') if len(user_story) > 100 else user_story,
            'model': model_name,
            'prompt': prompt_type,
            'classification': binary_pred,
            'nfr_type': type_pred_out,
            'confidence': confidence,
            'processing_time': elapsed,
            'raw_response': str(raw)
        }
        # Ensure JSON-serializable plain python types
        entry = ensure_py(entry)
        history = session.get('classification_history', [])
        history.append(entry)
        session['classification_history'] = history
        session.modified = True

        return json_response({'success': True, 'result': result})
    except Exception as e:
        logger.exception("api_classify failure")
        return json_response({'error': f'Classification failed: {str(e)}'}, status=500)


@app.route('/api/batch_classify', methods=['POST'])
def api_batch_classify():
    try:
        if 'file' not in request.files:
            return json_response({'error': 'No file uploaded'}, status=400)
        file = request.files['file']
        if file.filename == '':
            return json_response({'error': 'No file selected'}, status=400)

        model_name = request.form.get('model', session.get('selected_model'))
        prompt_type = request.form.get('prompt', session.get('selected_prompt'))
        try:
            max_stories = int(request.form.get('max_stories', 25))
        except Exception:
            max_stories = 25

        df = pd.read_csv(file)

        if 'user_story' not in df.columns and 'story' not in df.columns:
            return json_response({'error': 'CSV must contain "user_story" or "story" column'}, status=400)
        if 'story' in df.columns:
            df = df.rename(columns={'story': 'user_story'})

        stories = df['user_story'].astype(str).head(max_stories).tolist()
        model_fn = classifier.models.get(model_name)
        technique = classifier.prompt_templates.get(prompt_type, 'zeroshot')
        if not model_fn:
            return json_response({'error': f'Model {model_name} not available'}, status=400)

        results = []
        for i, story in enumerate(stories, 1):
            try:
                s_time = time.time()
                raw = model_fn(story, technique)
                elapsed = round(time.time() - s_time, 3)
                binary_pred = extract_is_nfr(raw)
                type_pred = extract_nfr_type_multilabel(raw)
                if isinstance(type_pred, (list, tuple)):
                    type_pred_out = ', '.join([str(x) for x in type_pred])
                else:
                    type_pred_out = str(type_pred)
                results.append({
                    'user_story': str(story),
                    'classification': str(binary_pred),
                    'nfr_type': type_pred_out,
                    'processing_time': float(elapsed)
                })
                # small throttle
                time.sleep(0.05)
            except Exception as e:
                logger.exception(f"Error classifying story idx {i}")
                results.append({
                    'user_story': str(story),
                    'classification': 'ERROR',
                    'nfr_type': 'ERROR',
                    'processing_time': 0.0
                })

        # metrics (cast to python types)
        results_df = pd.DataFrame(results)
        total = int(len(results_df))
        fr_count = int((results_df['classification'] == 'No').sum()) if 'classification' in results_df else 0
        nfr_count = int((results_df['classification'] == 'Yes').sum()) if 'classification' in results_df else 0
        error_count = int((results_df['classification'] == 'ERROR').sum()) if 'classification' in results_df else 0
        avg_time = float(round(results_df['processing_time'].mean(), 3)) if not results_df['processing_time'].empty else 0.0

        metrics = {
            'total': total,
            'fr_count': fr_count,
            'nfr_count': nfr_count,
            'error_count': error_count,
            'avg_time': avg_time
        }

        # Save last batch results to session (ensure python types)
        session['last_batch_results'] = ensure_py(results)
        session.modified = True

        return json_response({'success': True, 'results': ensure_py(results), 'metrics': metrics, 'download_available': True})
    except Exception as e:
        logger.exception("api_batch_classify failure")
        return json_response({'error': f'Batch processing failed: {str(e)}'}, status=500)


@app.route('/api/compare_models', methods=['POST'])
def api_compare_models():
    try:
        data = request.get_json(force=True)
        selected_models = data.get('models', [])
        selected_prompts = data.get('prompts', [])
        test_stories = data.get('test_stories', classifier.sample_stories[:4])

        if not selected_models or not selected_prompts:
            return json_response({'error': 'Please select at least one model and one prompt'}, status=400)

        # Normalize test stories to text list
        if isinstance(test_stories, list) and test_stories and isinstance(test_stories[0], dict):
            stories = [it.get('story') for it in test_stories]
        else:
            stories = test_stories

        results = []
        for model_name in selected_models:
            for prompt_name in selected_prompts:
                model_fn = classifier.models.get(model_name)
                technique = classifier.prompt_templates.get(prompt_name, 'zeroshot')
                if not model_fn:
                    continue
                times = []
                preds = []
                for s in stories:
                    start = time.time()
                    try:
                        raw = model_fn(s, technique)
                        pred = extract_is_nfr(raw)
                    except Exception:
                        pred = 'ERROR'
                    times.append(time.time() - start)
                    preds.append(pred)
                    time.sleep(0.03)
                avg_time = float(round(sum(times) / len(times), 4)) if times else 0.0
                error_rate = float(round(preds.count('ERROR') / len(preds), 3)) if preds else 0.0
                results.append({'Model': model_name, 'Prompt': prompt_name, 'Avg_Time': avg_time, 'Error_Rate': error_rate})
        return json_response({'success': True, 'results': results})
    except Exception as e:
        logger.exception("api_compare_models failure")
        return json_response({'error': f'Model comparison failed: {str(e)}'}, status=500)


@app.route('/api/analytics_data', methods=['GET'])
def api_analytics_data():
    try:
        history = session.get('classification_history', [])
        if not history:
            return json_response({
                'success': True,
                'metrics': {
                    'total': 0,
                    'fr': 0,
                    'nfr': 0,
                    'avg_time': 0
                },
                'charts': {},
                'recent_activity': []
            })

        df = pd.DataFrame(history)

        # -------------------------
        # GLOBAL METRICS
        # -------------------------
        total = int(len(df))
        fr = int((df['classification'] == 'No').sum())
        nfr = int((df['classification'] == 'Yes').sum())
        avg_time = float(round(df['processing_time'].mean(), 3))

        metrics = {
            'total': total,
            'fr': fr,
            'nfr': nfr,
            'avg_time': avg_time
        }

        # -------------------------
        # CHARTS (UNCHANGED)
        # -------------------------
        charts = {}

        try:
            classification_counts = df['classification'].value_counts()
            fig_dist = px.pie(
                values=classification_counts.values,
                names=classification_counts.index,
                title='Classification Distribution'
            )
            charts['distribution'] = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception:
            charts['distribution'] = None

        try:
            model_counts = df['model'].value_counts()
            fig_models = px.bar(
                x=model_counts.index,
                y=model_counts.values,
                title='Model Usage'
            )
            charts['models'] = json.dumps(fig_models, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception:
            charts['models'] = None

        # -------------------------
        # GROUP BY MODEL (MOST RECENT)
        # -------------------------
        grouped = (
            df.sort_values('timestamp')
              .groupby('model', as_index=False)
              .agg(
                  total_runs=('model', 'count'),
                  fr_count=('classification', lambda x: (x == 'No').sum()),
                  nfr_count=('classification', lambda x: (x == 'Yes').sum()),
                  first_seen=('timestamp', 'min'),
                  last_seen=('timestamp', 'max'),
                  avg_time=('processing_time', 'mean')
              )
        )

        grouped['avg_time'] = grouped['avg_time'].astype(float).round(3)

        # -------------------------
        # ACCURACY (BEST-EFFORT)
        # -------------------------
        # If expected labels exist, compute accuracy
        if 'expected' in df.columns:
            acc_df = (
                df.assign(correct=lambda x: x['expected'] == x['classification'])
                  .groupby('model')['correct']
                  .mean()
                  .reset_index(name='accuracy')
            )
            grouped = grouped.merge(acc_df, on='model', how='left')
            grouped['accuracy'] = (grouped['accuracy'] * 100).round(2)
        else:
            grouped['accuracy'] = 'N/A'

        # -------------------------
        # MOST RECENT PER MODEL
        # -------------------------
        recent_activity = (
            grouped.sort_values(by='last_seen', ascending=False)
                   .to_dict('records')
        )

        return json_response({
            'success': True,
            'metrics': metrics,
            'charts': charts,
            'recent_activity': ensure_py(recent_activity)
        })

    except Exception as e:
        logger.exception("api_analytics_data failure")
        return json_response(
            {'error': f'Analytics data failed: {str(e)}'},
            status=500
        )

@app.route('/api/download_history', methods=['GET'])
def api_download_history():
    try:
        history = session.get('classification_history', [])
        if not history:
            return json_response({'error': 'No history available'}, status=400)
        df = pd.DataFrame(history)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        filename = f"classification_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=filename)
    except Exception as e:
        logger.exception("api_download_history failure")
        return json_response({'error': f'Download failed: {str(e)}'}, status=500)


@app.route('/api/download_batch_results', methods=['GET'])
def api_download_batch_results():
    try:
        results = session.get('last_batch_results', [])
        if not results:
            return json_response({'error': 'No batch results available'}, status=400)
        df = pd.DataFrame(results)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=filename)
    except Exception as e:
        logger.exception("api_download_batch_results failure")
        return json_response({'error': f'Download failed: {str(e)}'}, status=500)


@app.route('/api/clear_history', methods=['POST'])
def api_clear_history():
    try:
        session['classification_history'] = []
        session.modified = True
        return json_response({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        logger.exception("api_clear_history failure")
        return json_response({'error': f'Failed to clear history: {str(e)}'}, status=500)


@app.route('/api/sample_data', methods=['POST'])
def api_sample_data():
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', session.get('selected_model'))
        prompt_type = data.get('prompt', session.get('selected_prompt'))
        model_fn = classifier.models.get(model_name)
        technique = classifier.prompt_templates.get(prompt_type, 'zeroshot')
        if not model_fn:
            return json_response({'error': f'Model {model_name} not available'}, status=400)

        results = []
        for item in classifier.sample_stories[:3]:
            story = item['story']
            try:
                raw = model_fn(story, technique)
                pred = extract_is_nfr(raw)
                expected_binary = 'Yes' if item['expected'] == 'NFR' else 'No'
                match = True if expected_binary == pred else False
                results.append({
                    'Story': story[:80] + ('...' if len(story) > 80 else ''),
                    'Expected': item['expected'],
                    'Predicted': 'NFR' if pred == 'Yes' else 'FR',
                    'Match': match
                })
            except Exception:
                results.append({'Story': story[:80] + ('...' if len(story) > 80 else ''), 'Expected': item['expected'], 'Predicted': 'ERROR', 'Match': False})

        matches = sum(1 for r in results if r['Match'])
        accuracy = round(matches / len(results), 3) if results else 0.0
        return json_response({'success': True, 'results': results, 'accuracy': accuracy})
    except Exception as e:
        logger.exception("api_sample_data failure")
        return json_response({'error': f'Sample data test failed: {str(e)}'}, status=500)


# ----------------------
# Run (development)
# ----------------------
if __name__ == '__main__':
    # Use debug=True for dev only
    app.run(debug=True, host='0.0.0.0', port=5000)
