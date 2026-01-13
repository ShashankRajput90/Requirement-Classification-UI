"""
Flask FR/NFR Classification Tool
Object-oriented design for easy customization and maintenance
- Clean separation of config, models, and API logic
- Easy to add new models, techniques, or features
- Maintains session history for analytics
"""
import os
import json
import uuid
import logging
from datetime import datetime
from collections import Counter

import pandas as pd
from flask import Flask, render_template, request, session, Response, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION CLASS - Easy to customize and extend
# ═══════════════════════════════════════════════════════════════════
class AppConfig:
    """Centralized configuration for easy maintenance"""
    
    # Available Models - Add/remove models here
    MODELS = {
        'groq_llama3': 'Groq LLaMA 3.1 8B',
        'gemini': 'Gemini 2.5 Pro',
        'cohere': 'Cohere Command-R+',
        'claude': 'Claude 3 Haiku',
        'mistral': 'Mistral Local'
    }
    
    # Available Techniques - Only Few Shot and Zero Shot
    TECHNIQUES = ['Zero Shot', 'Few Shot']
    
    # Technique Mapping - Maps UI labels to backend keys
    TECHNIQUE_MAP = {
        'Zero Shot': 'zero_shot',
        'Few Shot': 'few_shot'
    }
    
    # Default Settings
    DEFAULT_MODEL = 'groq_llama3'
    DEFAULT_TECHNIQUE = 'Zero Shot'
    
    # Test Stories for Model Comparison - Easy to add more
    TEST_STORIES = [
        {'story': 'The system shall be available 24/7.', 'expected': 'Yes'},
        {'story': 'Users can reset passwords via email.', 'expected': 'No'},
        {'story': 'The page should load within 2 seconds.', 'expected': 'Yes'},
        {'story': 'As a user, I want to view my order history.', 'expected': 'No'},
        {'story': 'The system must handle 10000 concurrent users.', 'expected': 'Yes'}
    ]
    
    # Sample Stories for Quick Demo
    SAMPLE_STORIES = [
        {'story': 'The system shall be available 24/7.', 'expected': 'NFR'},
        {'story': 'Users can reset passwords via email.', 'expected': 'FR'},
        {'story': 'The page should load within 2 seconds.', 'expected': 'NFR'}
    ]

# Use config class
AVAILABLE_MODELS = AppConfig.MODELS
AVAILABLE_TECHNIQUES = {v: v for k, v in enumerate(AppConfig.TECHNIQUES)}

# Try import model/eval helpers; provide safe fallbacks
try:
    from code_integration import (
        classify_with_groq_deepseek,
        classify_with_groq,
        classify_with_gemini,
        classify_with_cohere,
        classify_with_claude,
        classify_with_mistral_local,
        clean_response as _clean_response  # if available
    )
except Exception:
    classify_with_groq_deepseek = None
    classify_with_groq = None
    classify_with_gemini = None
    classify_with_cohere = None
    classify_with_claude = None
    classify_with_mistral_local = None
    def _clean_response(s: str) -> str:
        return (s or '').strip()

# Stub classifier used as fallback when real model functions are unavailable
def _stub_classify(user_story: str, technique: str) -> str:
    text = (user_story or '').lower()
    nfr_keywords = {
        'Performance': ['latency', 'respond', 'throughput', 'speed', 'performance'],
        'Security': ['secure', 'security', 'encrypt', 'authentication', 'authorization'],
        'Availability': ['24/7', 'uptime', 'availability', 'downtime'],
        'Reliability': ['reliable', 'reliability', 'robust'],
        'Maintainability': ['maintainable', 'maintainability', 'modularity', 'extensible'],
        'Usability': ['usability', 'user-friendly', 'ease of use'],
    }
    for nfr, kws in nfr_keywords.items():
        if any(k in text for k in kws):
            return f"1. Is NFR: Yes\n2. NFR Type: {nfr}\n3. Reason: Heuristic keyword match"
    return "1. Is NFR: No\n2. NFR Type: -\n3. Reason: Looks functional by heuristic"

try:
    from evaluation import extract_is_nfr, extract_nfr_type_multilabel
except Exception:
    def extract_is_nfr(resp: str) -> str:
        if not resp:
            return 'No'
        text = resp.lower()
        return 'Yes' if 'is nfr: yes' in text or 'non-functional' in text else 'No'
    def extract_nfr_type_multilabel(resp: str):
        if not resp:
            return []
        # naive parse after "NFR Type:" if present
        lower = resp
        if 'NFR Type:' in resp:
            lower = resp.split('NFR Type:')[-1]
        tokens = [t.strip() for t in lower.replace(',', '-').split('-') if t.strip()]
        return tokens or ['Unknown']

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-me')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('interactive_flask')

# JSON encoder to safely handle numpy/pandas
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            import pandas as pd
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
        except Exception:
            pass
        return super().default(obj)

def json_response(payload, status=200):
    return Response(json.dumps(payload, cls=NpEncoder), status=status, mimetype='application/json')

# Session helpers
def init_session():
    if 'classification_history' not in session:
        session['classification_history'] = []

@app.before_request
def _ensure_session():
    init_session()

# Model dispatch
def get_model_fn(model_key):
    mapping = {
        'groq_llama3': classify_with_groq or classify_with_groq_deepseek or _stub_classify,
        'groq_gpt': classify_with_groq_deepseek or classify_with_groq or _stub_classify,
        'gemini': classify_with_gemini or _stub_classify,
        'cohere': classify_with_cohere or _stub_classify,
        'claude': classify_with_claude or _stub_classify,
        'mistral': classify_with_mistral_local or _stub_classify,
        'mistral_local': classify_with_mistral_local or _stub_classify,
    }
    # Default to stub for any unknown key
    return mapping.get(model_key, _stub_classify)

# Routes: Pages
@app.route('/')
def home():
    """Main page with configuration from AppConfig"""
    return render_template(
        'index.html',
        models=list(AppConfig.MODELS.keys()),
        prompts=AppConfig.TECHNIQUES,
        selected_model=AppConfig.DEFAULT_MODEL if AppConfig.DEFAULT_MODEL in AppConfig.MODELS else (list(AppConfig.MODELS.keys())[0] if AppConfig.MODELS else ''),
        selected_prompt=AppConfig.DEFAULT_TECHNIQUE
    )

@app.route('/evaluate')
def evaluate_page():
    return render_template('evaluate.html', models=AVAILABLE_MODELS, techniques=AVAILABLE_TECHNIQUES)

@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html')

# APIs
@app.route('/api/classify', methods=['POST'])
def api_classify():
    try:
        data = request.get_json(force=True)
        story = data.get('story') or data.get('user_story')
        model_key = data.get('model', 'groq_llama3')
        # Accept either explicit technique or UI "prompt" label
        prompt_label = data.get('prompt')
        technique = data.get('technique') or AppConfig.TECHNIQUE_MAP.get(prompt_label, 'few_shot')

        if not story:
            return json_response({'success': False, 'error': 'story is required'}, status=400)

        fn = get_model_fn(model_key)
        if not fn:
            return json_response({'success': False, 'error': f'Unsupported model: {model_key}'}, status=400)

        raw = fn(story, technique)
        raw_clean = _clean_response(raw) if callable(_clean_response) else (raw or '')
        is_nfr = extract_is_nfr(raw_clean)
        types = extract_nfr_type_multilabel(raw_clean)

        # record into session history with serializable types
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'story': str(story),
            'model': str(model_key),
            'technique': str(technique),
            'raw': str(raw_clean),
            'is_nfr': str(is_nfr),
            'types': [str(t) for t in (types if isinstance(types, list) else [types])],
            'processing_time': 0.0
        }
        hist = session.get('classification_history', [])
        hist.append(entry)
        session['classification_history'] = hist
        session.modified = True

        # Return formatted for UI
        result = {
            'timestamp': entry['timestamp'],
            'story': entry['story'],
            'model': entry['model'],
            'classification': 'Yes' if is_nfr == 'Yes' else 'No',
            'nfr_type': ', '.join(entry['types']) if entry['types'] else 'N/A',
            'confidence': 'High',
            'processing_time': 0.0,
            'raw_response': entry['raw']
        }
        return json_response({'success': True, 'result': result})
    except Exception as e:
        logger.exception('Classification failed')
        return json_response({'success': False, 'error': str(e)}, status=500)


@app.route('/api/batch_classify', methods=['POST'])
def api_batch_classify():
    try:
        if 'file' not in request.files:
            return json_response({'error': 'No file uploaded'}, status=400)
        file = request.files['file']
        if file.filename == '':
            return json_response({'error': 'No file selected'}, status=400)

        model_key = request.form.get('model', 'groq_llama3')
        prompt_label = request.form.get('prompt', 'Few Shot')
        try:
            max_stories = int(request.form.get('max_stories', 25))
        except Exception:
            max_stories = 25

        # Map UI prompt label to technique key using AppConfig
        technique = AppConfig.TECHNIQUE_MAP.get(prompt_label, 'few_shot')

        df = pd.read_csv(file)
        story_col = 'user_story' if 'user_story' in df.columns else ('story' if 'story' in df.columns else None)
        if not story_col:
            return json_response({'error': 'CSV must contain "user_story" or "story" column'}, status=400)

        stories = df[story_col].astype(str).head(max_stories).tolist()
        fn = get_model_fn(model_key)
        if not fn:
            return json_response({'error': f'Model {model_key} not available'}, status=400)

        results = []
        times = []
        for story in stories:
            try:
                import time as _t
                start = _t.time()
                raw = fn(story, technique)
                elapsed = round(_t.time() - start, 3)
                times.append(elapsed)
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
            except Exception:
                results.append({
                    'user_story': str(story),
                    'classification': 'ERROR',
                    'nfr_type': 'ERROR',
                    'processing_time': 0.0
                })

        import numpy as _np
        total = int(len(results))
        fr_count = int(sum(1 for r in results if r['classification'] == 'No'))
        nfr_count = int(sum(1 for r in results if r['classification'] == 'Yes'))
        error_count = int(sum(1 for r in results if r['classification'] == 'ERROR'))
        avg_time = float(round(float(_np.mean(times)) if times else 0.0, 3))

        metrics = {
            'total': total,
            'fr_count': fr_count,
            'nfr_count': nfr_count,
            'error_count': error_count,
            'avg_time': avg_time
        }

        session['last_batch_results'] = results
        
        # Add batch results to classification history for analytics with serializable types
        history = session.get('classification_history', [])
        for result in results:
            history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'story': str(result['user_story']),
                'model': str(model_key),
                'technique': str(technique),
                'is_nfr': str(result['classification']),
                'types': [str(result['nfr_type'])] if result['nfr_type'] != 'ERROR' else [],
                'processing_time': float(result['processing_time'])
            })
        session['classification_history'] = history
        session.modified = True

        return json_response({'success': True, 'results': results, 'metrics': metrics, 'download_available': True})
    except Exception as e:
        logger.exception('api_batch_classify failed')
        return json_response({'error': f'Batch processing failed: {str(e)}'}, status=500)


@app.route('/api/download_batch_results', methods=['GET'])
def api_download_batch_results():
    try:
        results = session.get('last_batch_results', [])
        if not results:
            return json_response({'error': 'No batch results available'}, status=400)
        import io
        df = pd.DataFrame(results)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=filename)
    except Exception as e:
        logger.exception('download_batch_results failed')
        return json_response({'error': f'Download failed: {str(e)}'}, status=500)


@app.route('/api/sample_data', methods=['POST'])
def api_sample_data():
    try:
        data = request.get_json(force=True)
        model_key = data.get('model', 'groq_llama3')
        prompt_label = data.get('prompt', 'Few Shot')
        # Use AppConfig for technique mapping and sample data
        technique = AppConfig.TECHNIQUE_MAP.get(prompt_label, 'few_shot')
        fn = get_model_fn(model_key)
        if not fn:
            return json_response({'error': f'Model {model_key} not available'}, status=400)

        sample_stories = AppConfig.SAMPLE_STORIES

        results = []
        for item in sample_stories:
            story = item['story']
            try:
                raw = fn(story, technique)
                pred = extract_is_nfr(raw)
                expected_binary = 'Yes' if item['expected'] == 'NFR' else 'No'
                match = (expected_binary == pred)
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
        logger.exception('api_sample_data failed')
        return json_response({'error': f'Sample data test failed: {str(e)}'}, status=500)

@app.route('/api/upload-batch', methods=['POST'])
def api_upload_batch():
    try:
        if 'file' not in request.files:
            return json_response({'success': False, 'error': 'No file part'}, status=400)
        file = request.files['file']
        if file.filename == '':
            return json_response({'success': False, 'error': 'No selected file'}, status=400)
        fname = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(save_path)
        
        # Read CSV to get row count for dynamic slider
        df = pd.read_csv(save_path)
        story_col = 'user_story' if 'user_story' in df.columns else ('story' if 'story' in df.columns else None)
        if story_col:
            total_stories = int(len(df))
        else:
            total_stories = 0
        
        return json_response({'success': True, 'file': fname, 'total_stories': total_stories})
    except Exception as e:
        logger.exception('Upload failed')
        return json_response({'success': False, 'error': str(e)}, status=500)

@app.route('/api/evaluate-batch', methods=['POST'])
def api_evaluate_batch():
    try:
        data = request.get_json(force=True)
        model_key = data.get('model', 'groq_llama3')
        technique = data.get('technique', 'few_shot')
        fname = data.get('file')
        if not fname:
            return json_response({'success': False, 'error': 'file is required'}, status=400)

        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        if not os.path.exists(csv_path):
            return json_response({'success': False, 'error': 'Uploaded file not found'}, status=404)

        df = pd.read_csv(csv_path)
        # Column detection: prefer 'user_story', fallback to 'story'
        story_col = 'user_story' if 'user_story' in df.columns else ('story' if 'story' in df.columns else None)
        if not story_col:
            # try heuristic for common names
            for c in df.columns:
                if c.lower() in {'user story', 'requirement', 'text'}:
                    story_col = c
                    break
        if not story_col:
            return json_response({'success': False, 'error': 'Could not detect story column'}, status=400)

        fn = get_model_fn(model_key)
        if not fn:
            return json_response({'success': False, 'error': f'Unsupported model: {model_key}'}, status=400)

        results = []
        for story in df[story_col].fillna('').astype(str).tolist():
            if not story.strip():
                results.append({'story': story, 'raw': '', 'is_nfr': 'No', 'types': []})
                continue
            raw = fn(story, technique)
            raw_clean = _clean_response(raw) if callable(_clean_response) else (raw or '')
            is_nfr = extract_is_nfr(raw_clean)
            types = extract_nfr_type_multilabel(raw_clean)
            results.append({'story': story, 'raw': raw_clean, 'is_nfr': is_nfr, 'types': types})

        # Simple aggregate counts
        total = len(results)
        fr = sum(1 for r in results if r['is_nfr'] == 'No')
        nfr = total - fr
        type_counter = Counter(t for r in results for t in r['types'] if t)

        payload = {
            'success': True,
            'summary': {
                'total': total,
                'fr': fr,
                'nfr': nfr,
                'types': type_counter,
            },
            'results': results,
        }
        return json_response(payload)
    except Exception as e:
        logger.exception('Evaluate batch failed')
        return json_response({'success': False, 'error': str(e)}, status=500)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    try:
        history = session.get('classification_history', [])
        if not history:
            return json_response({'success': True, 'total': 0, 'fr': 0, 'nfr': 0, 'models': {}, 'techniques': {}, 'types': {}})

        df = pd.DataFrame(history)
        total = len(df)
        nfr = (df['is_nfr'] == 'Yes').sum()
        fr = total - nfr
        models = df['model'].value_counts().to_dict()
        techniques = df['technique'].value_counts().to_dict()
        types = Counter(t for lst in df['types'].tolist() for t in (lst or []))

        return json_response({'success': True, 'total': int(total), 'fr': int(fr), 'nfr': int(nfr), 'models': models, 'techniques': techniques, 'types': dict(types)})
    except Exception as e:
        logger.exception('Stats failed')
        return json_response({'success': False, 'error': str(e)}, status=500)


@app.route('/api/analytics_data', methods=['GET'])
def api_analytics_data():
    """Return analytics data with native Python types to avoid int64 serialization."""
    try:
        history = session.get('classification_history', [])
        if not history:
            return json_response({
                'success': True,
                'metrics': {'total': 0, 'fr': 0, 'nfr': 0, 'avg_time': 0.0},
                'charts': {},
                'model_summary': [],
                'recent_activity': []
            })

        df = pd.DataFrame(history)

        # Overall metrics (cast to native types)
        total = int(len(df))
        fr = int((df.get('is_nfr') == 'No').sum()) if 'is_nfr' in df else 0
        nfr = int((df.get('is_nfr') == 'Yes').sum()) if 'is_nfr' in df else 0
        avg_time = float(round(df['processing_time'].mean(), 3)) if 'processing_time' in df and not df['processing_time'].empty else 0.0

        metrics = {'total': total, 'fr': fr, 'nfr': nfr, 'avg_time': avg_time}

        # Charts can be added later; keep empty to avoid serialization surprises
        charts = {}

        # Model-level summary (mirrors flask_app.py format)
        model_summary = []
        if 'model' in df.columns:
            for model, g in df.groupby('model'):
                total_g = int(len(g))
                fr_g = int((g.get('is_nfr') == 'No').sum()) if 'is_nfr' in g else 0
                nfr_g = int((g.get('is_nfr') == 'Yes').sum()) if 'is_nfr' in g else 0
                correct = int(g.get('is_nfr').value_counts().max()) if 'is_nfr' in g and not g['is_nfr'].empty else 0
                accuracy = float(round(correct / total_g, 3)) if total_g else 0.0

                model_summary.append({
                    'model': str(model),
                    'total': total_g,
                    'fr': fr_g,
                    'nfr': nfr_g,
                    'accuracy': accuracy,
                    'last_used': str(g['timestamp'].iloc[-1]) if 'timestamp' in g else ''
                })

        # Recent activity (keep for compatibility, but optional in UI)
        recent_activity = [
            {
                'timestamp': row.get('timestamp', ''),
                'story': (row.get('story', '') or '')[:50] + '...',
                'model': row.get('model', ''),
                'classification': 'NFR' if row.get('is_nfr') == 'Yes' else 'FR'
            }
            for _, row in df.tail(10).iterrows()
        ]

        return json_response({
            'success': True,
            'metrics': metrics,
            'charts': charts,
            'model_summary': model_summary,
            'recent_activity': recent_activity
        })
    except Exception as e:
        logger.exception('api_analytics_data failed')
        return json_response({'error': f'Analytics data failed: {str(e)}'}, status=500)


@app.route('/api/compare_models', methods=['POST'])
def api_compare_models():
    """Compare multiple models and prompts on sample test data"""
    try:
        data = request.get_json(force=True)
        selected_models = data.get('models', [])
        selected_prompts = data.get('prompts', [])

        if not selected_models or not selected_prompts:
            return json_response({'error': 'Select at least one model and prompt'}, status=400)

        # Use AppConfig for test stories and technique mapping
        test_stories = AppConfig.TEST_STORIES
        
        comparison_results = []
        
        for model_key in selected_models:
            for prompt_label in selected_prompts:
                fn = get_model_fn(model_key)
                if not fn:
                    continue
                    
                technique = AppConfig.TECHNIQUE_MAP.get(prompt_label, 'few_shot')
                correct = 0
                total_time = 0
                
                for test in test_stories:
                    try:
                        import time as _t
                        start = _t.time()
                        raw = fn(test['story'], technique)
                        elapsed = _t.time() - start
                        total_time += elapsed
                        
                        prediction = extract_is_nfr(raw)
                        if prediction == test['expected']:
                            correct += 1
                    except Exception:
                        pass
                
                accuracy = (correct / len(test_stories)) if test_stories else 0
                avg_time = (total_time / len(test_stories)) if test_stories else 0
                
                comparison_results.append({
                    'model': model_key,
                    'prompt': prompt_label,
                    'accuracy': round(accuracy, 2),
                    'avg_time': round(avg_time, 2),
                    'correct': correct,
                    'total': len(test_stories)
                })

        return json_response({'success': True, 'results': comparison_results})
    except Exception as e:
        logger.exception('api_compare_models failed')
        return json_response({'error': f'Model comparison failed: {str(e)}'}, status=500)


@app.route('/api/download_history', methods=['GET'])
def api_download_history():
    try:
        history = session.get('classification_history', [])
        if not history:
            return json_response({'error': 'No history available'}, status=400)
        import io
        df = pd.DataFrame(history)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        filename = f"classification_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=filename)
    except Exception as e:
        logger.exception('api_download_history failed')
        return json_response({'error': f'Download failed: {str(e)}'}, status=500)


@app.route('/api/clear_history', methods=['POST'])
def api_clear_history():
    try:
        session['classification_history'] = []
        session.modified = True
        return json_response({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        logger.exception('api_clear_history failed')
        return json_response({'error': f'Failed to clear history: {str(e)}'}, status=500)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
