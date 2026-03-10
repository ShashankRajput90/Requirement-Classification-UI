# from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
# import json
# from code_integration import classify
# import time
# import os
# import pandas as pd
# import random
# import psycopg2
# from psycopg2.extras import RealDictCursor
# from werkzeug.security import generate_password_hash, check_password_hash
# from authlib.integrations.flask_client import OAuth
# from functools import wraps
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, login_required, login_user, logout_user, current_user
# from authlib.integrations.flask_client import OAuth
# from models import db, User

# app = Flask(__name__)
# app.secret_key = 'supersecretkey'  # Needed for session management
# oauth = OAuth(app)
# google = oauth.register(
#     name='google',
#     client_id=os.getenv("GOOGLE_CLIENT_ID"),
#     client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
#     server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
#     client_kwargs={'scope': 'openid email profile'}
# )

# #PostgreSQL connection setup (update with your credentials)
# def get_db_connection():
#     return psycopg2.connect(
#         dbname="nfr_fr_db",
#         user="postgres",
#         password="123456",
#         host="localhost",
#         port="5432"
#     )

# # db logic for proper history tracking and analytics
# # single batch run (creates a id for a single processing)
# def create_batch_run(user_id, model, technique, total_stories):
#     conn = get_db_connection()
#     cur = conn.cursor()

#     cur.execute("""
#         INSERT INTO batch_runs (user_id, model, prompting_technique, total_stories)
#         VALUES (%s, %s, %s, %s)
#         RETURNING id
#     """, (user_id, model, technique, total_stories))

#     batch_run_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()

#     return batch_run_id

# #store batch result in db function 
# def insert_batch_result(batch_run_id, user_id, story, model, classification, category, latency):
#     conn = get_db_connection()
#     cur = conn.cursor()

#     cur.execute("""
#         INSERT INTO batch_results
#         (batch_run_id, user_id, story, model, classification, category, latency)
#         VALUES (%s, %s, %s, %s, %s, %s, %s)
#     """, (
#         batch_run_id,
#         user_id,
#         story,
#         model,
#         classification,
#         category,
#         latency
#     ))

#     conn.commit()
#     cur.close()
#     conn.close()



# # --- Mock Data & Constants ---

# MODELS = ["groq_gpt", "groq_llama3", "gemini", "cohere", "claude", "mistral"]

# CATEGORIES = [
#     "Accuracy", "Usability", "Performance", "Efficiency", "Security",
#     "Privacy", "Fairness & Bias", "Explainability", "Interpretability",
#     "Transparency", "Accessibility", "Reliability", "Robustness",
#     "Maintainability", "Scalability", "Interoperability",
#     "Completeness & Consistency", "Trust", "Safety & Governance"
# ]
# #global variable for storing batch results in-memory (for analytics)
# batch_results_storage = []

# # --- Helper Functions ---

# def parse_backend_response(raw_response, model, strategy, latency=0):

#     # 🔥 Safety: if backend returned dict (error case)
#     if not isinstance(raw_response, str):
#         return {
#             "classification": "Error",
#             "category": None,
#             "classification_full": "Error",
#             "confidence": "50",
#             "model": model,
#             "strategy": strategy,
#             "latency": latency,
#             "reason": str(raw_response),
#             "raw_response": str(raw_response)
#         }

#     classification = "NFR"
#     category = None
#     reason = "No reason provided"
#     confidence = 50      

#     lines = raw_response.split('\n')
#     for line in lines:
#         if "Is NFR:" in line:
#             val = line.split(":", 1)[1].strip().lower()
#             classification = "NFR" if "yes" in val else "FR"
#         elif "NFR Type:" in line:
#             category = line.split(":", 1)[1].strip()
#             if category.lower() in ["-", "none", "n/a", "na", ""]:
#                 category = None
#         elif "Reason:" in line:
#             reason = line.split(":", 1)[1].strip()
#         elif "Confidence:" in line:                          # ✅ ADD THIS
#             val = line.split(":", 1)[1].strip()
#             digits = ''.join(filter(str.isdigit, val))
#             confidence = int(digits) if digits else 50

#     if classification == "FR":
#         category = None

#     return {
#         "classification": classification,
#         "category": category,
#         "classification_full": f"{classification} - {category}" if category else classification,
#         "confidence": confidence,
#         "model": model,
#         "strategy": strategy,
#         "latency": latency,
#         "reason": reason,
#         "raw_response": raw_response
#     }

# # --- Routes ---

# @app.route('/')
# def index():
#     return render_template('index.html', page='index')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         return index() # Placeholder for actual auth logic
#     return render_template('login.html', page='login')

# @app.route('/single', methods=['GET', 'POST'])
# def single():
#     if request.method == 'POST':
#         if not request.is_json:
#             return jsonify({"error": "Request must be JSON"}), 400
#         data = request.json
#         story = data.get('story')
#         model = data.get('model', 'ChatGPT')
#         strategy = data.get('strategy', 'Zero-shot')

#         # Model mapping
#         model_map = {
#             "ChatGPT": "groq_gpt",
#             "Gemini": "gemini",
#             "Claude": "claude",
#             "Groq": "groq_llama3",
#             "Cohere": "cohere",
#             "Mistral": "mistral"
#         }
#         backend_model = model_map.get(model, "groq_gpt")

#         # Strategy mapping
#         strategy_map = {
#             "Zero-shot": "zero_shot",
#             "Few-shot": "few_shot",
#             "Chain-of-Thought": "chain_of_thought",
#         }

#         technique = strategy_map.get(strategy, "zero_shot")

#         try:
#             start_t = time.time()
#             raw_response, status_code = classify(backend_model, story, technique)

#             latency = round(time.time() - start_t, 2)

#             if status_code != 200:
#                 return jsonify(raw_response), status_code

#             result = parse_backend_response(raw_response, model, strategy, latency)
#             return jsonify(result), 200

#         except Exception as e:
#             return jsonify({
#                 "error": str(e),
#                 "classification": "Error",
#                 "category": None
#             }), 500

#     return render_template('single.html', page='single')


# # ===============================
# # BATCH ROUTE (Streaming NDJSON)
# # ===============================

# @app.route('/batch', methods=['GET', 'POST'])
# def batch():
#     if request.method == 'POST':

#         count = int(request.form.get('count', 10))
#         model = request.form.get('model', 'ChatGPT')
#         strategy = request.form.get('strategy', 'Zero-shot')

#         model_map = {
#             "ChatGPT": "groq_gpt",
#             "Gemini": "gemini",
#             "Claude": "claude",
#             "Groq": "groq_llama3",
#             "Cohere": "cohere",
#             "Mistral": "mistral"
#         }

#         backend_model = model_map.get(model, "groq_gpt")

#         strategy_map = {
#             "Zero-shot": "zero_shot",
#             "Few-shot": "few_shot",
#             "Chain-of-Thought": "chain_of_thought",
#         }

#         technique = strategy_map.get(strategy, "zero_shot")

#         # --- File Upload Handling ---
#         if 'file' in request.files and request.files['file'].filename != '':
#             file = request.files['file']
#             try:
#                 df = pd.read_csv(file)

#                 story_col = None
#                 for col in ['story', 'user_story', 'User Story', 'text', 'content']:
#                     if col in df.columns:
#                         story_col = col
#                         break

#                 if not story_col:
#                     return jsonify({"error": "CSV must contain story column."}), 400

#                 df = df.rename(columns={story_col: 'story'})
#                 sampled_df = df.head(min(count, len(df))).reset_index(drop=True)

#             except Exception as e:
#                 return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

#         else:
#             batch_dir = "batches"

#             if not os.path.exists(batch_dir):
#                 return jsonify({"error": "No batch data found."}), 500

#             batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.csv')]
#             if not batch_files:
#                 return jsonify({"error": "No CSV files in batches folder."}), 500

#             df = pd.read_csv(os.path.join(batch_dir, batch_files[0]))
#             sampled_df = df.sample(n=min(count, len(df))).reset_index(drop=True)
        
#         user_id = session.get("user_id", 1)
#         total_rows = len(sampled_df)

#         batch_run_id = create_batch_run(
#             user_id=user_id,
#             model=model,
#             technique=strategy,
#             total_stories=total_rows
#         )

#         def generate():
#             fr_count = 0
#             nfr_count = 0
#             total_time = 0
#             processed = 0
#             category_counts = {c: 0 for c in CATEGORIES}

#             for _, row in sampled_df.iterrows():
#                 story = row['story']

#                 try:
#                     start_t = time.time()
#                     raw_response, status_code = classify(backend_model, story, technique)
#                     latency = round(time.time() - start_t, 2)

#                     if status_code != 200:
#                         res = {
#                             "classification": "Error",
#                             "category": None,
#                             "latency": latency,
#                             "error": raw_response
#                         }
#                     else:
#                         res = parse_backend_response(raw_response, model, strategy, latency)
#                         # Store result for analytics
#                         batch_results_storage.append(res)
#                         insert_batch_result(
#                             batch_run_id=batch_run_id,
#                             user_id=user_id,
#                             story=story,
#                             model=model,
#                             classification=res["classification"],
#                             category=res.get("category"),
#                             latency=res.get("latency")
#                         )
                        

#                 except Exception as e:
#                     res = {
#                         "classification": "Error",
#                         "category": None,
#                         "latency": 0.0,
#                         "error": str(e)
#                     }

#                 if res.get("classification") == "FR":
#                     fr_count += 1
#                 elif res.get("classification") == "NFR":
#                     nfr_count += 1
#                     cat = res.get("category")
#                     if cat:
#                         for c in CATEGORIES:
#                             if c.lower() in cat.lower():
#                                 category_counts[c] += 1
#                                 break

#                 total_time += res.get("latency", 0)
#                 processed += 1

#                 yield json.dumps({
#                     "type": "result",
#                     "story": story,
#                     "result": res,
#                     "current_stats": {
#                         "total": processed,
#                         "fr_count": fr_count,
#                         "nfr_count": nfr_count,
#                         "avg_time": round(total_time / processed, 2),
#                         "category_counts": category_counts
#                     }
#                 }) + "\n"

#             yield json.dumps({
#                 "type": "summary",
#                 "summary": {
#                     "total": processed,
#                     "fr_count": fr_count,
#                     "nfr_count": nfr_count,
#                     "avg_time": round(total_time / processed, 2),
#                     "category_counts": category_counts
#                 }
#             }) + "\n"

#         return Response(stream_with_context(generate()),
#                         mimetype='application/x-ndjson')

#     return render_template('batch.html', page='batch')


# # ===============================
# # COMPARISON ROUTE
# # ===============================

# @app.route('/comparison')
# def comparison():
#     return render_template('comparison.html', page='comparison')


# @app.route('/api/comparison-data')
# def comparison_data():

#     results_file = "binary_results.csv"

#     if os.path.exists(results_file):
#         try:
#             df = pd.read_csv(results_file)
#             summary = df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].mean().reset_index()

#             data = []
#             for _, row in summary.iterrows():
#                 data.append({
#                     "model": row['Model'],
#                     "accuracy": round(row['Accuracy'], 2),
#                     "precision": round(row['Precision'], 2),
#                     "recall": round(row['Recall'], 2),
#                     "f1": round(row['F1'], 2),
#                     "avg_latency": 0.0
#                 })

#             data.sort(key=lambda x: x['f1'], reverse=True)
#             return jsonify(data)

#         except Exception as e:
#             print(f"Error reading results: {e}")

#     # Fallback mock
#     data = []
#     for model in MODELS:
#         data.append({
#             "model": model,
#             "accuracy": round(random.uniform(0.80, 0.98), 2),
#             "precision": round(random.uniform(0.75, 0.95), 2),
#             "recall": round(random.uniform(0.75, 0.95), 2),
#             "f1": round(random.uniform(0.78, 0.96), 2),
#             "avg_latency": round(random.uniform(0.2, 2.5), 2)
#         })

#     data.sort(key=lambda x: x['accuracy'], reverse=True)
#     return jsonify(data)


# @app.route('/analytics')
# def analytics():
#     return render_template('analytics.html', page='analytics')

# @app.route("/api/analytics_data")
# def analytics_data():

#     batch_run_id = request.args.get("batch_run_id")

#     conn = get_db_connection()
#     cur = conn.cursor(cursor_factory=RealDictCursor)

#     if batch_run_id:
#         cur.execute("""
#             SELECT classification, category, latency
#             FROM batch_results
#             WHERE batch_run_id = %s
#         """, (batch_run_id,))
#     else:
#         cur.execute("""
#             SELECT classification, category, latency
#             FROM batch_results
#         """)

#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     if not rows:
#         return jsonify({
#             "total": 0,
#             "fr": 0,
#             "nfr": 0,
#             "categories": {},
#             "latencies": []
#         })

#     total = len(rows)
#     fr = sum(1 for r in rows if r["classification"] == "FR")
#     nfr = sum(1 for r in rows if r["classification"] == "NFR")

#     categories = {}
#     latencies = []

#     for r in rows:
#         if r["classification"] == "NFR" and r["category"]:
#             categories[r["category"]] = categories.get(r["category"], 0) + 1

#         if r["latency"]:
#             latencies.append(r["latency"])

#     return jsonify({
#         "total": total,
#         "fr": fr,
#         "nfr": nfr,
#         "categories": categories,
#         "latencies": latencies
#     })

# @app.route("/api/reset_batch", methods=["POST"])
# def reset_batch():
#     global batch_results_storage
#     batch_results_storage = []
#     return jsonify({"status": "cleared"})

# @app.route("/api/compare_prompting")
# def compare_prompting():

#     conn = get_db_connection()
#     cur = conn.cursor(cursor_factory=RealDictCursor)

#     cur.execute("""
#         SELECT br.prompting_technique,
#                COUNT(*) AS total,
#                SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) AS fr,
#                SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) AS nfr,
#                AVG(r.latency) AS avg_latency
#         FROM batch_runs br
#         JOIN batch_results r ON br.id = r.batch_run_id
#         GROUP BY br.prompting_technique
#     """)

#     data = cur.fetchall()
#     cur.close()
#     conn.close()

#     return jsonify(data)

# @app.route("/api/batch_runs")
# def get_batch_runs():
#     conn = get_db_connection()
#     cur = conn.cursor(cursor_factory=RealDictCursor)

#     cur.execute("""
#         SELECT id, model, prompting_technique, created_at
#         FROM batch_runs
#         ORDER BY created_at DESC
#     """)

#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     return jsonify(rows)

# @app.route("/api/technique_comparison")
# def technique_comparison():

#     batch_run_id = request.args.get("batch_run_id")

#     conn = get_db_connection()
#     cur = conn.cursor(cursor_factory=RealDictCursor)

#     if batch_run_id:
#         cur.execute("""
#             SELECT br.prompting_technique,
#                    COUNT(*) as total,
#                    SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) as fr,
#                    SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) as nfr,
#                    AVG(r.latency) as avg_latency
#             FROM batch_runs br
#             JOIN batch_results r ON br.id = r.batch_run_id
#             WHERE br.id = %s
#             GROUP BY br.prompting_technique
#         """, (batch_run_id,))
#     else:
#         cur.execute("""
#             SELECT br.prompting_technique,
#                    COUNT(*) as total,
#                    SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) as fr,
#                    SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) as nfr,
#                    AVG(r.latency) as avg_latency
#             FROM batch_runs br
#             JOIN batch_results r ON br.id = r.batch_run_id
#             GROUP BY br.prompting_technique
#         """)

#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     return jsonify(rows)


# # ===============================
# # RUN
# # ===============================

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for
import json
from code_integration import classify
import time
import os
import pandas as pd
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from authlib.integrations.flask_client import OAuth
from models import db, User

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
# PostgreSQL connection (kept for analytics routes)
# =========================
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

# =========================
# Batch DB Helpers
# =========================
def create_batch_run(user_id, model, technique, total_stories):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO batch_runs (user_id, model, prompting_technique, total_stories)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (user_id, model, technique, total_stories))
    batch_run_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return batch_run_id

def insert_batch_result(batch_run_id, user_id, story, model, classification, category, latency):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO batch_results
        (batch_run_id, user_id, story, model, classification, category, latency)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (batch_run_id, user_id, story, model, classification, category, latency))
    conn.commit()
    cur.close()
    conn.close()

# =========================
# Constants
# =========================
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

    # Extract thinking block
    import re
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


# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for("login"))
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

        model_map = {
            "ChatGPT": "groq_gpt",
            "Gemini":  "gemini",
            "Claude":  "claude",
            "Groq":    "groq_llama3",
            "Cohere":  "cohere",
            "Mistral": "mistral"
        }
        backend_model = model_map.get(model, "groq_gpt")

        strategy_map = {
            "Zero-shot":       "zero_shot",
            "Few-shot":        "few_shot",
            "Chain-of-Thought": "chain_of_thought",
        }
        technique = strategy_map.get(strategy, "zero_shot")

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
# Batch Route
# =========================
@app.route('/batch', methods=['GET', 'POST'])
@login_required
def batch():
    if request.method == 'POST':
        count    = int(request.form.get('count', 10))
        model    = request.form.get('model', 'ChatGPT')
        strategy = request.form.get('strategy', 'Zero-shot')

        model_map = {
            "ChatGPT": "groq_gpt",
            "Gemini":  "gemini",
            "Claude":  "claude",
            "Groq":    "groq_llama3",
            "Cohere":  "cohere",
            "Mistral": "mistral"
        }
        backend_model = model_map.get(model, "groq_gpt")

        strategy_map = {
            "Zero-shot":       "zero_shot",
            "Few-shot":        "few_shot",
            "Chain-of-Thought": "chain_of_thought",
        }
        technique = strategy_map.get(strategy, "zero_shot")

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

        batch_run_id = create_batch_run(
            user_id=user_id,
            model=model,
            technique=strategy,
            total_stories=total_rows
        )

        def generate():
            fr_count = 0
            nfr_count = 0
            total_time = 0
            processed = 0
            category_counts = {c: 0 for c in CATEGORIES}

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
                        insert_batch_result(
                            batch_run_id=batch_run_id,
                            user_id=user_id,
                            story=story,
                            model=model,
                            classification=res["classification"],
                            category=res.get("category"),
                            latency=res.get("latency")
                        )

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
                    "avg_time": round(total_time / processed, 2),
                    "category_counts": category_counts
                }
            }) + "\n"

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
    conn = get_db_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)

    if batch_run_id:
        cur.execute("""
            SELECT classification, category, latency
            FROM batch_results
            WHERE batch_run_id = %s
        """, (batch_run_id,))
    else:
        cur.execute("SELECT classification, category, latency FROM batch_results")

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return jsonify({"total": 0, "fr": 0, "nfr": 0, "categories": {}, "latencies": []})

    total      = len(rows)
    fr         = sum(1 for r in rows if r["classification"] == "FR")
    nfr        = sum(1 for r in rows if r["classification"] == "NFR")
    categories = {}
    latencies  = []

    for r in rows:
        if r["classification"] == "NFR" and r["category"]:
            categories[r["category"]] = categories.get(r["category"], 0) + 1
        if r["latency"]:
            latencies.append(r["latency"])

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
    conn = get_db_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT br.prompting_technique,
               COUNT(*) AS total,
               SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) AS fr,
               SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) AS nfr,
               AVG(r.latency) AS avg_latency
        FROM batch_runs br
        JOIN batch_results r ON br.id = r.batch_run_id
        GROUP BY br.prompting_technique
    """)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(data)


@app.route("/api/batch_runs")
@login_required
def get_batch_runs():
    conn = get_db_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, model, prompting_technique, created_at
        FROM batch_runs
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)


@app.route("/api/technique_comparison")
@login_required
def technique_comparison():
    batch_run_id = request.args.get("batch_run_id")
    conn = get_db_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)

    if batch_run_id:
        cur.execute("""
            SELECT br.prompting_technique,
                   COUNT(*) as total,
                   SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) as fr,
                   SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) as nfr,
                   AVG(r.latency) as avg_latency
            FROM batch_runs br
            JOIN batch_results r ON br.id = r.batch_run_id
            WHERE br.id = %s
            GROUP BY br.prompting_technique
        """, (batch_run_id,))
    else:
        cur.execute("""
            SELECT br.prompting_technique,
                   COUNT(*) as total,
                   SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) as fr,
                   SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) as nfr,
                   AVG(r.latency) as avg_latency
            FROM batch_runs br
            JOIN batch_results r ON br.id = r.batch_run_id
            GROUP BY br.prompting_technique
        """)

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)


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
    # Currently returning mock data or simple aggregates based on batch_results if possible.
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # In a fully implemented version, this would join with an api_costs table.
    # For now, we return basic aggregated metrics.
    cur.execute("""
        SELECT COUNT(*) as total_calls,
               SUM(CASE WHEN classification != 'Error' THEN 1 ELSE 0 END) as successful_calls
        FROM batch_results
    """)
    stats = cur.fetchone()
    cur.close()
    conn.close()

    total_calls = stats['total_calls'] if stats['total_calls'] else 0
    successful_calls = stats['successful_calls'] if stats['successful_calls'] else 0
    
    success_rate = round((successful_calls / total_calls * 100)) if total_calls > 0 else 0

    # Mock cost calculation
    total_tokens = total_calls * 250  # Assuming 250 tokens per call on average
    avg_cost = 0.0015                 # Dummy avg cost
    total_cost = total_calls * avg_cost

    return jsonify({
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "total_calls": total_calls,
        "success_rate": success_rate,
        "avg_cost": avg_cost
    })


# =========================
# Run
# =========================
if __name__ == '__main__':
    app.run(debug=True, port=5000)