import re
import json
import time
import os
import threading

# Thread-safe per-user batch progress store
# {user_id: {"total": int, "processed": int, "status": "running"|"done"|"idle", "run_id": int}}
_batch_progress: dict = {}
_batch_progress_lock = threading.Lock()

import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from authlib.integrations.flask_client import OAuth
from sqlalchemy import func

from code_integration import classify
from models import db, User, BatchRun, BatchResult, RequirementHistory

# Add to existing imports at top of app.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

app = Flask(__name__)

# =========================
# App Config
# =========================
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# =========================
# Extensions
# =========================
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."
login_manager.login_message_category = "error"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Create tables on startup
with app.app_context():
    db.create_all()

# =========================
# Module-level Constants
# =========================
MODEL_MAP = {
    "ChatGPT": "groq_gpt",
    "Gemini":  "gemini",
    "Claude":  "claude",
    "Groq":    "groq_llama3",
    "Cohere":  "cohere",
    "Mistral": "mistral"
}

STRATEGY_MAP = {
    "Zero-shot":        "zero_shot",
    "Few-shot":         "few_shot",
    "Chain-of-Thought": "chain_of_thought",
}

MODELS = ["groq_gpt", "groq_llama3", "gemini", "cohere", "claude", "mistral"]

CATEGORIES = [
    "Accuracy", "Usability", "Performance", "Efficiency", "Security",
    "Privacy", "Fairness & Bias", "Explainability", "Interpretability",
    "Transparency", "Accessibility", "Reliability", "Robustness",
    "Maintainability", "Scalability", "Interoperability",
    "Completeness & Consistency", "Trust", "Safety & Governance"
]

batch_results_storage = []

# =========================
# Helper: Parse Response
# =========================
def parse_backend_response(raw_response, model, strategy, latency=0):
    if not isinstance(raw_response, str):
        return {
            "classification": "Error",
            "category": None,
            "classification_full": "Error",
            "confidence": "50",
            "model": model,
            "strategy": strategy,
            "latency": latency,
            "reason": str(raw_response),
            "step_by_step": "",
            "raw_response": str(raw_response)
        }

    classification = "NFR"
    category = None
    reason = "No reason provided"
    confidence = 50
    step_by_step = ""

    think_match = re.search(r"<think>(.*?)</think>", raw_response, flags=re.DOTALL)
    if think_match:
        step_by_step = think_match.group(1).strip()
    else:
        # Fallback: Capture everything before "1. Is NFR:"
        lines_before = []
        for line in raw_response.split('\n'):
            if "1. Is NFR:" in line:
                break
            lines_before.append(line)
        step_by_step = "\n".join(lines_before).strip()

    lines = raw_response.split('\n')
    for line in lines:
        if "Is NFR:" in line:
            val = line.split(":", 1)[1].strip().lower()
            classification = "NFR" if "yes" in val else "FR"
        elif "NFR Type:" in line:
            category = line.split(":", 1)[1].strip()
            if category.lower() in ["-", "none", "n/a", "na", ""]:
                category = None
        elif "Reason:" in line:
            reason = line.split(":", 1)[1].strip()
        elif "Confidence:" in line:
            val = line.split(":", 1)[1].strip()
            digits = ''.join(filter(str.isdigit, val))
            confidence = int(digits) if digits else 50

    if classification == "FR":
        category = None

    return {
        "classification": classification,
        "category": category,
        "classification_full": f"{classification} - {category}" if category else classification,
        "confidence": confidence,
        "model": model,
        "strategy": strategy,
        "latency": latency,
        "reason": reason,
        "step_by_step": step_by_step,
        "raw_response": raw_response
    }

# =========================
# Auth Routes
# =========================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            return render_template('login.html', error="Invalid email or password.", page='login')

        login_user(user, remember=remember)
        next_page = request.args.get('next')
        return redirect(next_page or url_for("index"))

    return render_template('login.html', page='login')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not name or not email or len(password) < 8:
            return render_template('signup.html', error="All fields required. Password min 8 characters.", page='signup')

        if User.query.filter_by(email=email).first():
            return render_template('signup.html', error="An account with this email already exists.", page='signup')

        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return redirect(url_for("index"))

    return render_template('signup.html', page='signup')


@app.route('/logout', methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    token     = google.authorize_access_token()
    user_info = token.get('userinfo')

    if not user_info:
        return redirect(url_for("login"))

    email     = user_info['email'].lower()
    name      = user_info.get('name', email.split('@')[0])
    google_id = user_info['sub']
    avatar    = user_info.get('picture')

    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User.query.filter_by(email=email).first()
        if user:
            # Link Google to existing email account
            user.google_id  = google_id
            user.avatar_url = avatar
        else:
            # Brand new user via Google
            user = User(name=name, email=email, google_id=google_id, avatar_url=avatar)
            db.session.add(user)

    db.session.commit()
    login_user(user)
    return redirect(url_for("index"))


# =========================
# Main Routes (protected)
# =========================
@app.route('/')
@login_required
def index():
    return render_template('index.html', page='index')


@app.route('/single', methods=['GET', 'POST'])
@login_required
def single():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data     = request.json
        story    = data.get('story')
        model    = data.get('model', 'ChatGPT')
        strategy = data.get('strategy', 'Zero-shot')

        backend_model = MODEL_MAP.get(model, "groq_gpt")
        technique     = STRATEGY_MAP.get(strategy, "zero_shot")

        try:
            start_t = time.time()
            raw_response, status_code = classify(backend_model, story, technique)
            latency = round(time.time() - start_t, 2)

            if status_code != 200:
                return jsonify(raw_response), status_code

            result = parse_backend_response(raw_response, model, strategy, latency)
            return jsonify(result), 200

        except Exception as e:
            return jsonify({"error": str(e), "classification": "Error", "category": None}), 500

    return render_template('single.html', page='single')


# =========================
# Batch Status
# =========================
@app.route('/api/batch/status')
@login_required
def batch_status():
    with _batch_progress_lock:
        prog = _batch_progress.get(current_user.id, {"status": "idle"})
    return jsonify(prog)


# =========================
# Batch Route
# =========================
@app.route('/batch', methods=['GET', 'POST'])
@login_required
def batch():
    if request.method == 'POST':
        count    = int(request.form.get('count', 10))
        model    = request.form.get('model', 'ChatGPT')
        strategy = request.form.get('strategy', 'Zero-shot')

        backend_model = MODEL_MAP.get(model, "groq_gpt")
        technique     = STRATEGY_MAP.get(strategy, "zero_shot")

        # File upload handling
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                df = pd.read_csv(file)
                story_col = None
                for col in ['story', 'user_story', 'User Story', 'text', 'content']:
                    if col in df.columns:
                        story_col = col
                        break
                if not story_col:
                    return jsonify({"error": "CSV must contain story column."}), 400
                df = df.rename(columns={story_col: 'story'})
                sampled_df = df.head(min(count, len(df))).reset_index(drop=True)
            except Exception as e:
                return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400
        else:
            batch_dir = "batches"
            if not os.path.exists(batch_dir):
                return jsonify({"error": "No batch data found."}), 500
            batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.csv')]
            if not batch_files:
                return jsonify({"error": "No CSV files in batches folder."}), 500
            df = pd.read_csv(os.path.join(batch_dir, batch_files[0]))
            sampled_df = df.sample(n=min(count, len(df))).reset_index(drop=True)

        # Use Flask-Login's current_user instead of session
        user_id    = current_user.id
        total_rows = len(sampled_df)

        # ORM insert for batch run
        batch_run = BatchRun(
            user_id=user_id,
            model=model,
            prompting_technique=strategy,
            total_stories=total_rows
        )
        db.session.add(batch_run)
        db.session.commit()
        batch_run_id = batch_run.id

        def generate():
            fr_count = 0
            nfr_count = 0
            total_time = 0
            processed = 0
            category_counts = {c: 0 for c in CATEGORIES}

            # Mark as running in the global progress store
            with _batch_progress_lock:
                _batch_progress[user_id] = {
                    "status": "running",
                    "total": total_rows,
                    "processed": 0,
                    "run_id": batch_run_id,
                }

            for _, row in sampled_df.iterrows():
                story = row['story']
                try:
                    start_t = time.time()
                    raw_response, status_code = classify(backend_model, story, technique)
                    latency = round(time.time() - start_t, 2)

                    if status_code != 200:
                        res = {"classification": "Error", "category": None, "latency": latency, "error": raw_response}
                    else:
                        res = parse_backend_response(raw_response, model, strategy, latency)
                        batch_results_storage.append(res)

                        # ORM insert for batch result
                        result_row = BatchResult(
                            batch_run_id=batch_run_id,
                            user_id=user_id,
                            story=story,
                            model=model,
                            classification=res["classification"],
                            category=res.get("category"),
                            latency=res.get("latency")
                        )
                        db.session.add(result_row)
                        db.session.commit()
                        res["id"] = result_row.id  # expose DB id to the stream

                except Exception as e:
                    res = {"classification": "Error", "category": None, "latency": 0.0, "error": str(e)}

                if res.get("classification") == "FR":
                    fr_count += 1
                elif res.get("classification") == "NFR":
                    nfr_count += 1
                    cat = res.get("category")
                    if cat:
                        for c in CATEGORIES:
                            if c.lower() in cat.lower():
                                category_counts[c] += 1
                                break

                total_time += res.get("latency", 0)
                processed += 1

                # Update global progress store
                with _batch_progress_lock:
                    if user_id in _batch_progress:
                        _batch_progress[user_id]["processed"] = processed

                yield json.dumps({
                    "type": "result",
                    "story": story,
                    "result": res,
                    "current_stats": {
                        "total": processed,
                        "fr_count": fr_count,
                        "nfr_count": nfr_count,
                        "avg_time": round(total_time / processed, 2),
                        "category_counts": category_counts
                    }
                }) + "\n"

            yield json.dumps({
                "type": "summary",
                "summary": {
                    "total": processed,
                    "fr_count": fr_count,
                    "nfr_count": nfr_count,
                    "avg_time": round(total_time / processed, 2) if processed else 0,
                    "category_counts": category_counts
                }
            }) + "\n"

            # Mark done so global progress bar knows to stop
            with _batch_progress_lock:
                if user_id in _batch_progress:
                    _batch_progress[user_id]["status"] = "done"

        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

    return render_template('batch.html', page='batch')


# =========================
# Comparison Route
# =========================
@app.route('/comparison')
@login_required
def comparison():
    return render_template('comparison.html', page='comparison')


@app.route('/api/comparison-data')
@login_required
def comparison_data():
    import random
    results_file = "binary_results.csv"

    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            summary = df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].mean().reset_index()
            data = []
            for _, row in summary.iterrows():
                data.append({
                    "model":       row['Model'],
                    "accuracy":    round(row['Accuracy'], 2),
                    "precision":   round(row['Precision'], 2),
                    "recall":      round(row['Recall'], 2),
                    "f1":          round(row['F1'], 2),
                    "avg_latency": 0.0
                })
            data.sort(key=lambda x: x['f1'], reverse=True)
            return jsonify(data)
        except Exception as e:
            print(f"Error reading results: {e}")

    # Fallback mock
    import random
    data = []
    for model in MODELS:
        data.append({
            "model":       model,
            "accuracy":    round(random.uniform(0.80, 0.98), 2),
            "precision":   round(random.uniform(0.75, 0.95), 2),
            "recall":      round(random.uniform(0.75, 0.95), 2),
            "f1":          round(random.uniform(0.78, 0.96), 2),
            "avg_latency": round(random.uniform(0.2, 2.5), 2)
        })
    data.sort(key=lambda x: x['accuracy'], reverse=True)
    return jsonify(data)


# =========================
# Analytics Routes
# =========================
@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', page='analytics')


@app.route("/api/analytics_data")
@login_required
def analytics_data():
    batch_run_id = request.args.get("batch_run_id")

    query = db.session.query(BatchResult)
    if batch_run_id:
        query = query.filter(BatchResult.batch_run_id == int(batch_run_id))

    rows = query.all()

    if not rows:
        return jsonify({"total": 0, "fr": 0, "nfr": 0, "categories": {}, "latencies": []})

    total      = len(rows)
    fr         = sum(1 for r in rows if r.classification == "FR")
    nfr        = sum(1 for r in rows if r.classification == "NFR")
    categories = {}
    latencies  = []

    for r in rows:
        if r.classification == "NFR" and r.category:
            categories[r.category] = categories.get(r.category, 0) + 1
        if r.latency is not None:
            latencies.append(r.latency)

    return jsonify({"total": total, "fr": fr, "nfr": nfr, "categories": categories, "latencies": latencies})


@app.route("/api/reset_batch", methods=["POST"])
@login_required
def reset_batch():
    global batch_results_storage
    batch_results_storage = []
    return jsonify({"status": "cleared"})


@app.route("/api/compare_prompting")
@login_required
def compare_prompting():
    rows = (
        db.session.query(
            BatchRun.prompting_technique,
            func.count(BatchResult.id).label("total"),
            func.sum(db.case((BatchResult.classification == "FR", 1), else_=0)).label("fr"),
            func.sum(db.case((BatchResult.classification == "NFR", 1), else_=0)).label("nfr"),
            func.avg(BatchResult.latency).label("avg_latency")
        )
        .join(BatchResult, BatchRun.id == BatchResult.batch_run_id)
        .group_by(BatchRun.prompting_technique)
        .all()
    )

    data = [
        {
            "prompting_technique": r.prompting_technique,
            "total":               r.total,
            "fr":                  r.fr,
            "nfr":                 r.nfr,
            "avg_latency":         float(r.avg_latency) if r.avg_latency else 0.0
        }
        for r in rows
    ]
    return jsonify(data)


@app.route("/api/batch_runs")
@login_required
def get_batch_runs():
    runs = BatchRun.query.order_by(BatchRun.created_at.desc()).all()
    data = [
        {
            "id":                  r.id,
            "model":               r.model,
            "prompting_technique": r.prompting_technique,
            "created_at":          r.created_at.isoformat() if r.created_at else None
        }
        for r in runs
    ]
    return jsonify(data)


@app.route("/api/technique_comparison")
@login_required
def technique_comparison():
    batch_run_id = request.args.get("batch_run_id")

    query = (
        db.session.query(
            BatchRun.prompting_technique,
            func.count(BatchResult.id).label("total"),
            func.sum(db.case((BatchResult.classification == "FR", 1), else_=0)).label("fr"),
            func.sum(db.case((BatchResult.classification == "NFR", 1), else_=0)).label("nfr"),
            func.avg(BatchResult.latency).label("avg_latency")
        )
        .join(BatchResult, BatchRun.id == BatchResult.batch_run_id)
        .group_by(BatchRun.prompting_technique)
    )

    if batch_run_id:
        query = query.filter(BatchRun.id == int(batch_run_id))

    rows = query.all()
    data = [
        {
            "prompting_technique": r.prompting_technique,
            "total":               r.total,
            "fr":                  r.fr,
            "nfr":                 r.nfr,
            "avg_latency":         float(r.avg_latency) if r.avg_latency else 0.0
        }
        for r in rows
    ]
    return jsonify(data)


# =========================
# API Usage Dashboard Routes
# =========================
@app.route('/api_dashboard')
@login_required
def api_dashboard():
    return render_template('api_dashboard.html', page='api_dashboard')


@app.route('/api/usage_data')
@login_required
def usage_data():
    total_calls     = db.session.query(func.count(BatchResult.id)).scalar() or 0
    successful_calls = (
        db.session.query(func.count(BatchResult.id))
        .filter(BatchResult.classification != "Error")
        .scalar() or 0
    )

    success_rate = round((successful_calls / total_calls * 100)) if total_calls > 0 else 0

    total_tokens = total_calls * 250   # ~250 tokens per call average
    avg_cost     = 0.0015
    total_cost   = total_calls * avg_cost

    return jsonify({
        "total_cost":   total_cost,
        "total_tokens": total_tokens,
        "total_calls":  total_calls,
        "success_rate": success_rate,
        "avg_cost":     avg_cost
    })
    
# =========================
# Ground Truth Evaluation Route
# =========================
@app.route('/evaluate', methods=['GET', 'POST'])
@login_required
def evaluate():
    return render_template('evaluate.html', page='evaluate')


@app.route('/api/evaluate', methods=['POST'])
@login_required
def run_evaluation():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    file     = request.files['file']
    model    = request.form.get('model', 'ChatGPT')
    strategy = request.form.get('strategy', 'Zero-shot')

    model_map    = MODEL_MAP
    strategy_map = {**STRATEGY_MAP, "Role-Based": "role_based", "ReAct": "react"}
    backend_model = model_map.get(model, "groq_gpt")
    technique     = strategy_map.get(strategy, "zero_shot")

    # --- Parse uploaded CSV ---
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {str(e)}"}), 400

    # Flexible column name support
    story_col = next((c for c in ['user_story', 'story', 'text', 'User Story', 'content'] if c in df.columns), None)
    label_col = next((c for c in ['label', 'Label', 'is_nfr', 'type'] if c in df.columns), None)

    if not story_col:
        return jsonify({"error": "CSV must have a 'user_story' column"}), 400
    if not label_col:
        return jsonify({"error": "CSV must have a 'label' column with ground truth (FR/NFR or Yes/No)"}), 400

    df = df.rename(columns={story_col: 'user_story', label_col: 'label'})
    df = df.dropna(subset=['user_story', 'label']).head(50)  # cap at 50 for cost control

    if len(df) < 2:
        return jsonify({"error": "CSV must have at least 2 valid rows"}), 400

    # --- Normalize ground truth labels ---
    def normalize_label(val):
        v = str(val).strip().lower()
        if v in ['yes', 'nfr', '1', 'non-functional', 'nonfunctional']:
            return 'NFR'
        return 'FR'

    df['true_label'] = df['label'].apply(normalize_label)
    has_type_col     = 'nfr_type' in df.columns

    # --- Run classification for each story ---
    y_true, y_pred, details = [], [], []

    for _, row in df.iterrows():
        story      = str(row['user_story']).strip()
        true_label = row['true_label']
        true_type  = str(row.get('nfr_type', '')).strip() if has_type_col else ''

        try:
            start_t        = time.time()
            raw_resp, code = classify(backend_model, story, technique)
            latency        = round(time.time() - start_t, 2)

            if code != 200:
                pred_label  = 'Error'
                parsed      = {}
                confidence  = 0
                reason      = str(raw_resp)
                pred_type   = ''
            else:
                parsed      = parse_backend_response(raw_resp, model, strategy, latency)
                pred_label  = parsed.get('classification', 'FR')
                confidence  = parsed.get('confidence', 50)
                reason      = parsed.get('reason', '')
                pred_type   = parsed.get('category') or ''

        except Exception as e:
            pred_label  = 'Error'
            confidence  = 0
            reason      = str(e)
            pred_type   = ''
            latency     = 0

        correct = (true_label == pred_label)
        y_true.append(true_label)
        y_pred.append(pred_label)

        details.append({
            "story":      story,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct":    correct,
            "confidence": confidence,
            "reason":     reason,
            "true_type":  true_type,
            "pred_type":  pred_type,
            "latency":    latency
        })

    # --- Filter out errors for metric computation ---
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != 'Error']
    if len(valid_pairs) < 2:
        return jsonify({"error": "Too many classification errors to compute metrics"}), 500

    vt, vp = zip(*valid_pairs)

    # --- Binary metrics (FR vs NFR) ---
    binary_metrics = {
        "accuracy":  round(accuracy_score(vt, vp) * 100, 1),
        "precision": round(precision_score(vt, vp, pos_label='NFR', zero_division=0) * 100, 1),
        "recall":    round(recall_score(vt, vp, pos_label='NFR', zero_division=0) * 100, 1),
        "f1":        round(f1_score(vt, vp, pos_label='NFR', zero_division=0) * 100, 1),
    }

    # --- Confusion matrix values ---
    cm = confusion_matrix(vt, vp, labels=['FR', 'NFR'])
    confusion = {
        "TN": int(cm[0][0]),  # FR predicted as FR
        "FP": int(cm[0][1]),  # FR predicted as NFR
        "FN": int(cm[1][0]),  # NFR predicted as FR
        "TP": int(cm[1][1]),  # NFR predicted as NFR
    }

    # --- Error analysis: group misclassifications ---
    false_positives = [d for d in details if d['true_label'] == 'FR'  and d['pred_label'] == 'NFR']
    false_negatives = [d for d in details if d['true_label'] == 'NFR' and d['pred_label'] == 'FR']

    # --- NFR type accuracy (only where both sides are NFR) ---
    type_details = [d for d in details if d['true_label'] == 'NFR' and d['pred_label'] == 'NFR'
                    and d['true_type'] and d['pred_type']]
    type_accuracy = None
    if type_details:
        type_correct  = sum(1 for d in type_details if d['true_type'].lower() in d['pred_type'].lower())
        type_accuracy = round(type_correct / len(type_details) * 100, 1)

    # --- Confidence breakdown ---
    correct_confidences   = [d['confidence'] for d in details if d['correct']     and d['pred_label'] != 'Error']
    incorrect_confidences = [d['confidence'] for d in details if not d['correct'] and d['pred_label'] != 'Error']
    avg_conf_correct   = round(sum(correct_confidences)   / len(correct_confidences), 1)   if correct_confidences   else 0
    avg_conf_incorrect = round(sum(incorrect_confidences) / len(incorrect_confidences), 1) if incorrect_confidences else 0

    return jsonify({
        "model":          model,
        "strategy":       strategy,
        "total":          len(details),
        "valid":          len(valid_pairs),
        "errors":         len(details) - len(valid_pairs),
        "binary_metrics": binary_metrics,
        "confusion":      confusion,
        "type_accuracy":  type_accuracy,
        "details":        details,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_conf_correct":   avg_conf_correct,
        "avg_conf_incorrect": avg_conf_incorrect,
    })


# =========================
# History Page
# =========================
@app.route('/history')
@login_required
def history():
    return render_template('history.html', page='history')


@app.route('/api/requirements/all')
@login_required
def get_all_requirements():
    page      = request.args.get('page',     1,    type=int)
    per_page  = request.args.get('per_page', 25,   type=int)
    run_id    = request.args.get('run_id',   None, type=int)
    clf       = request.args.get('cls',      None)
    search    = request.args.get('q',        None)
    model     = request.args.get('model',    None)

    query = db.session.query(BatchResult)

    if run_id:
        query = query.filter(BatchResult.batch_run_id == run_id)
    if clf:
        query = query.filter(BatchResult.classification == clf)
    if model:
        query = query.filter(BatchResult.model == model)
    if search:
        query = query.filter(BatchResult.story.ilike(f"%{search}%"))

    total = query.count()

    sort = request.args.get('sort', 'newest')
    if sort == 'oldest':
        query = query.order_by(BatchResult.id.asc())
    elif sort == 'fr_first':
        query = query.order_by(BatchResult.classification.asc(), BatchResult.id.desc())
    elif sort == 'nfr_first':
        query = query.order_by(BatchResult.classification.desc(), BatchResult.id.desc())
    elif sort == 'most_edits':
        from models import RequirementHistory as _RH
        edit_subq = (
            db.session.query(
                _RH.batch_result_id,
                func.count(_RH.id).label("ec")
            )
            .group_by(_RH.batch_result_id)
            .subquery()
        )
        query = (
            query.outerjoin(edit_subq, BatchResult.id == edit_subq.c.batch_result_id)
            .order_by(db.func.coalesce(edit_subq.c.ec, 0).desc(), BatchResult.id.desc())
        )
    else:  # newest
        query = query.order_by(BatchResult.id.desc())

    results = query.offset((page - 1) * per_page).limit(per_page).all()

    from models import RequirementHistory as RH

    data = []
    for r in results:
        edit_count = db.session.query(func.count(RH.id)).filter(RH.batch_result_id == r.id).scalar() or 0
        run_ts = None
        if r.run and r.run.created_at:
            run_ts = r.run.created_at.strftime("%b %d, %Y %I:%M %p")
        data.append({
            "id":             r.id,
            "story":          r.story,
            "model":          r.model,
            "classification": r.classification,
            "category":       r.category,
            "latency":        r.latency,
            "batch_run_id":   r.batch_run_id,
            "edit_count":     edit_count,
            "created_at":     run_ts,
        })

    return jsonify({
        "results": data,
        "total":   total,
        "page":    page,
        "pages":   (total + per_page - 1) // per_page,
    })


# =========================
# Requirement Version Control
# =========================
@app.route("/api/requirements/<int:result_id>/edit", methods=["POST"])
@login_required
def edit_requirement(result_id):
    data      = request.get_json()
    new_story = (data or {}).get("new_story", "").strip()

    if not new_story:
        return jsonify({"error": "new_story is required"}), 400

    req = BatchResult.query.get_or_404(result_id)

    # Only save history when something actually changed
    if req.story != new_story:
        history = RequirementHistory(
            batch_result_id         = req.id,
            user_id                 = current_user.id,
            previous_story          = req.story,
            new_story               = new_story,
            previous_classification = req.classification,
            new_classification      = req.classification  # classification unchanged on text edit
        )
        db.session.add(history)
        req.story = new_story
        db.session.commit()

    return jsonify({"success": True, "message": "Requirement updated."})


@app.route("/api/requirements/<int:result_id>/history")
@login_required
def get_requirement_history(result_id):
    # Make sure the result exists
    BatchResult.query.get_or_404(result_id)

    entries = (
        RequirementHistory.query
        .filter_by(batch_result_id=result_id)
        .order_by(RequirementHistory.changed_at.desc())
        .all()
    )

    data = [
        {
            "id":                      e.id,
            "previous_story":          e.previous_story,
            "new_story":               e.new_story,
            "previous_classification": e.previous_classification,
            "new_classification":      e.new_classification,
            "changed_at":              e.changed_at.strftime("%b %d, %Y at %I:%M %p") if e.changed_at else "",
            "edited_by":               e.editor.name if e.editor else "Unknown"
        }
        for e in entries
    ]

    return jsonify(data)


# =========================
# Run
# =========================
if __name__ == '__main__':
    app.run(debug=True, port=5000)