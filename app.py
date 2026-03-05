from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import json
import threading
from code_integration import classify
import time
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import random
import psycopg2
from psycopg2.extras import RealDictCursor
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Needed for session management

#PostgreSQL connection setup (update with your credentials)
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

# db logic for proper history tracking and analytics
# single batch run (creates a id for a single processing)
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

#store batch result in db function 
def insert_batch_result(batch_run_id, user_id, story, model, classification, category, latency, prompt_tokens, completion_tokens, cost):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO batch_results
        (batch_run_id, user_id, story, model, classification, category, latency, prompt_tokens, completion_tokens, cost)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        batch_run_id,
        user_id,
        story,
        model,
        classification,
        category,
        latency,
        prompt_tokens,
        completion_tokens,
        cost
    ))

    conn.commit()
    cur.close()
    conn.close()



# --- Mock Data & Constants ---

MODELS = ["groq_gpt", "groq_llama3", "gemini", "cohere", "claude", "mistral"]

MODEL_COSTS = {
    # Costs per 1M tokens in USD
    "groq_gpt": {"input": 0.0, "output": 0.0}, # Llama on Groq is typically free tier or very cheap, mock as 0
    "groq_llama3": {"input": 0.0, "output": 0.0},
    "gemini": {"input": 0.35, "output": 1.05}, # Gemini 1.5 Flash approx
    "cohere": {"input": 3.00, "output": 15.00}, # Command R+ approx
    "claude": {"input": 0.25, "output": 1.25}, # Haiku approx
    "mistral": {"input": 0.0, "output": 0.0} # Local is free
}

CATEGORIES = [
    "Accuracy", "Usability", "Performance", "Efficiency", "Security",
    "Privacy", "Fairness & Bias", "Explainability", "Interpretability",
    "Transparency", "Accessibility", "Reliability", "Robustness",
    "Maintainability", "Scalability", "Interoperability",
    "Completeness & Consistency", "Trust", "Safety & Governance"
]
#global variable for storing batch results in-memory (for analytics)
batch_results_storage = []

# --- Helper Functions ---

def parse_backend_response(raw_response, model, strategy, latency=0):

    # 🔥 Safety: if backend returned dict (error case)
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
            "raw_response": str(raw_response)
        }

    classification = "NFR"
    category = None
    reason = "No reason provided"
    confidence = 50      

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
        elif "Confidence:" in line:                          # ✅ ADD THIS
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
        "raw_response": raw_response
    }

# --- Auth Helper ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

@app.route('/')
@login_required
def index():
    return render_template('index.html', page='index')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT * FROM users WHERE name = %s", (username,))
        user = cur.fetchone()

        cur.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["name"]
            return redirect(url_for("analytics"))

        return render_template("login.html", error="Invalid credentials")

    return render_template('login.html', page='login')

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO users (name, email, password)
                VALUES (%s, %s, %s)
            """, (username, username + "@local.com", hashed_password))
            conn.commit()
        except Exception as e:
            conn.rollback()
            return render_template("login.html", error="User already exists")
        finally:
            cur.close()
            conn.close()

        return redirect(url_for("login"))

    return render_template("login.html", show_signup=True)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/single', methods=['GET', 'POST'])
@login_required
def single():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.json
        story = data.get('story')
        model = data.get('model', 'ChatGPT')
        strategy = data.get('strategy', 'Zero-shot')

        # Model mapping
        model_map = {
            "ChatGPT": "groq_gpt",
            "Gemini": "gemini",
            "Claude": "claude",
            "Groq": "groq_llama3",
            "Cohere": "cohere",
            "Mistral": "mistral"
        }
        backend_model = model_map.get(model, "groq_gpt")

        # Strategy mapping
        strategy_map = {
            "Zero-shot": "zero_shot",
            "Few-shot": "few_shot",
            "Chain-of-Thought": "chain_of_thought",
        }

        technique = strategy_map.get(strategy, "zero_shot")

        try:
            start_t = time.time()
            raw_response, status_code, usage = classify(backend_model, story, technique)

            latency = round(time.time() - start_t, 2)

            if status_code != 200:
                return jsonify(raw_response), status_code

            # Cost calc
            p_tokens = usage.get("prompt", 0)
            c_tokens = usage.get("completion", 0)
            rate = MODEL_COSTS.get(backend_model, {"input": 0, "output": 0})
            cost = (p_tokens * rate["input"] / 1000000) + (c_tokens * rate["output"] / 1000000)

            result = parse_backend_response(raw_response, model, strategy, latency)
            result["usage"] = usage
            result["cost"] = round(cost, 6)
            
            return jsonify(result), 200

        except Exception as e:
            return jsonify({
                "error": str(e),
                "classification": "Error",
                "category": None
            }), 500

    return render_template('single.html', page='single')


# ===============================
# BATCH ROUTE (Streaming NDJSON)
# ===============================

@app.route('/batch', methods=['GET', 'POST'])
@login_required
def batch():
    if request.method == 'POST':

        count = int(request.form.get('count', 10))
        model = request.form.get('model', 'ChatGPT')
        strategy = request.form.get('strategy', 'Zero-shot')

        model_map = {
            "ChatGPT": "groq_gpt",
            "Gemini": "gemini",
            "Claude": "claude",
            "Groq": "groq_llama3",
            "Cohere": "cohere",
            "Mistral": "mistral"
        }

        backend_model = model_map.get(model, "groq_gpt")

        strategy_map = {
            "Zero-shot": "zero_shot",
            "Few-shot": "few_shot",
            "Chain-of-Thought": "chain_of_thought",
        }

        technique = strategy_map.get(strategy, "zero_shot")

        # --- File Upload Handling ---
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
        
        user_id = session.get("user_id", 1)
        total_rows = len(sampled_df)

        batch_run_id = create_batch_run(
            user_id=user_id,
            model=model,
            technique=strategy,
            total_stories=total_rows
        )

        def process_batch_thread(sampled_df, backend_model, technique, user_id, batch_run_id, model_name, strat_name):
            for _, row in sampled_df.iterrows():
                story = row['story']
                try:
                    start_t = time.time()
                    raw_response, status_code, usage = classify(backend_model, story, technique)
                    latency = round(time.time() - start_t, 2)

                    if status_code == 200:
                        res = parse_backend_response(raw_response, model_name, strat_name, latency)
                        
                        p_tokens = usage.get("prompt", 0)
                        c_tokens = usage.get("completion", 0)
                        rate = MODEL_COSTS.get(backend_model, {"input": 0, "output": 0})
                        cost = (p_tokens * rate["input"] / 1000000) + (c_tokens * rate["output"] / 1000000)

                        batch_results_storage.append(res)
                        insert_batch_result(
                            batch_run_id=batch_run_id,
                            user_id=user_id,
                            story=story,
                            model=model_name,
                            classification=res["classification"],
                            category=res.get("category"),
                            latency=res.get("latency"),
                            prompt_tokens=p_tokens,
                            completion_tokens=c_tokens,
                            cost=cost
                        )
                except Exception as e:
                    pass

        thread = threading.Thread(target=process_batch_thread, args=(sampled_df, backend_model, technique, user_id, batch_run_id, model, strategy))
        thread.daemon = True
        thread.start()

        return jsonify({
            "status": "started",
            "batch_run_id": batch_run_id,
            "total": total_rows
        }), 200

    return render_template('batch.html', page='batch')


@app.route('/api/batch_progress')
def batch_progress():
    batch_run_id = request.args.get("batch_run_id")
    if not batch_run_id:
        return jsonify({"error": "Missing batch_run_id"}), 400

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT total_stories FROM batch_runs WHERE id = %s", (batch_run_id,))
    run = cur.fetchone()
    if not run:
        cur.close()
        conn.close()
        return jsonify({"error": "Batch run not found"}), 404

    total = run['total_stories']

    cur.execute("""
        SELECT classification, category, latency, story
        FROM batch_results
        WHERE batch_run_id = %s
        ORDER BY id ASC
    """, (batch_run_id,))
    rows = cur.fetchall()

    processed = len(rows)
    fr_count = sum(1 for r in rows if r["classification"] == "FR")
    nfr_count = sum(1 for r in rows if r["classification"] == "NFR")
    latency_sum = sum((r["latency"] or 0) for r in rows)
    avg_time = round(latency_sum / processed, 2) if processed > 0 else 0

    category_counts = {c: 0 for c in CATEGORIES}
    for r in rows:
        if r["classification"] == "NFR" and r["category"]:
            for c in CATEGORIES:
                if c.lower() in (r["category"] or "").lower():
                    category_counts[c] += 1
                    break

    cur.close()
    conn.close()

    return jsonify({
        "total": total,
        "processed": processed,
        "fr_count": fr_count,
        "nfr_count": nfr_count,
        "avg_time": avg_time,
        "category_counts": category_counts,
        "results": rows
    })


# ===============================
# COMPARISON ROUTE
# ===============================

@app.route('/comparison')
@login_required
def comparison():
    return render_template('comparison.html', page='comparison')


@app.route('/api/comparison-data')
def comparison_data():

    results_file = "binary_results.csv"

    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            summary = df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].mean().reset_index()

            data = []
            for _, row in summary.iterrows():
                data.append({
                    "model": row['Model'],
                    "accuracy": round(row['Accuracy'], 2),
                    "precision": round(row['Precision'], 2),
                    "recall": round(row['Recall'], 2),
                    "f1": round(row['F1'], 2),
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
            "model": model,
            "accuracy": round(random.uniform(0.80, 0.98), 2),
            "precision": round(random.uniform(0.75, 0.95), 2),
            "recall": round(random.uniform(0.75, 0.95), 2),
            "f1": round(random.uniform(0.78, 0.96), 2),
            "avg_latency": round(random.uniform(0.2, 2.5), 2)
        })

    data.sort(key=lambda x: x['accuracy'], reverse=True)
    return jsonify(data)


@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', page='analytics')

@app.route('/api_dashboard')
@login_required
def api_dashboard():
    return render_template('api_dashboard.html', page='api_dashboard')

@app.route('/history')
@login_required
def history():
    return render_template('history.html', page='history')

@app.route("/api/history_data")
@login_required
def history_data():
    user_id = session.get("user_id")
    days_filter = request.args.get("days_filter", "all")
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    query = """
        SELECT r.story, r.model, br.prompting_technique, r.classification, 
               r.category, r.latency, br.created_at, r.prompt_tokens, r.completion_tokens, r.cost
        FROM batch_results r
        JOIN batch_runs br ON r.batch_run_id = br.id
        WHERE r.user_id = %s
    """
    params = [user_id]

    if days_filter == 'today':
        query += " AND DATE(br.created_at) = CURRENT_DATE"
    elif days_filter == 'yesterday':
        query += " AND DATE(br.created_at) = CURRENT_DATE - INTERVAL '1 day'"
    elif days_filter == '7':
        query += " AND br.created_at >= NOW() - INTERVAL '7 days'"
    elif days_filter == '30':
        query += " AND br.created_at >= NOW() - INTERVAL '30 days'"
        
    query += " ORDER BY br.created_at DESC LIMIT 500" # Safety limit

    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    
    cur.close()
    conn.close()

    return jsonify(rows)

@app.route("/api/usage_data")
@login_required
def usage_data():
    user_id = session.get("user_id")
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT 
            COUNT(*) as total_calls,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            SUM(cost) as total_cost,
            SUM(CASE WHEN classification != 'Error' THEN 1 ELSE 0 END) as successful_calls
        FROM batch_results
        WHERE user_id = %s
    """, (user_id,))
    
    stats = cur.fetchone()
    cur.close()
    conn.close()

    total_calls = stats['total_calls'] or 0
    successful_calls = stats['successful_calls'] or 0
    total_tokens = (stats['total_prompt_tokens'] or 0) + (stats['total_completion_tokens'] or 0)
    total_cost = float(stats['total_cost'] or 0.0)

    success_rate = round((successful_calls / total_calls * 100), 2) if total_calls > 0 else 0.0
    avg_cost = round(total_cost / total_calls, 6) if total_calls > 0 else 0.0

    return jsonify({
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "success_rate": success_rate,
        "total_cost": total_cost,
        "avg_cost": avg_cost
    })

@app.route("/api/analytics_data")
@login_required
def analytics_data():

    batch_run_id = request.args.get("batch_run_id")
    days_filter = request.args.get("days_filter", "all")
    user_id = session.get("user_id")

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if batch_run_id:
        cur.execute("""
            SELECT classification, category, latency
            FROM batch_results
            WHERE batch_run_id = %s AND user_id = %s
        """, (batch_run_id, user_id))
    else:
        query = """
            SELECT r.classification, r.category, r.latency
            FROM batch_results r
            JOIN batch_runs br ON r.batch_run_id = br.id
            WHERE r.user_id = %s
        """
        params = [user_id]
        if days_filter == 'today':
            query += " AND DATE(br.created_at) = CURRENT_DATE"
        elif days_filter == 'yesterday':
            query += " AND DATE(br.created_at) = CURRENT_DATE - INTERVAL '1 day'"
        elif days_filter == '7':
            query += " AND br.created_at >= NOW() - INTERVAL '7 days'"
        elif days_filter == '30':
            query += " AND br.created_at >= NOW() - INTERVAL '30 days'"
        
        cur.execute(query, tuple(params))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return jsonify({
            "total": 0,
            "fr": 0,
            "nfr": 0,
            "categories": {},
            "latencies": []
        })

    total = len(rows)
    fr = sum(1 for r in rows if r["classification"] == "FR")
    nfr = sum(1 for r in rows if r["classification"] == "NFR")

    categories = {}
    latencies = []

    for r in rows:
        if r["classification"] == "NFR" and r["category"]:
            categories[r["category"]] = categories.get(r["category"], 0) + 1

        if r["latency"]:
            latencies.append(r["latency"])

    return jsonify({
        "total": total,
        "fr": fr,
        "nfr": nfr,
        "categories": categories,
        "latencies": latencies
    })

@app.route("/api/reset_batch", methods=["POST"])
def reset_batch():
    global batch_results_storage
    batch_results_storage = []
    return jsonify({"status": "cleared"})

@app.route("/api/compare_prompting")
@login_required
def compare_prompting():

    user_id = session.get("user_id")
    days_filter = request.args.get("days_filter", "all")
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    query = """
        SELECT br.prompting_technique,
               COUNT(*) AS total,
               SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) AS fr,
               SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) AS nfr,
               AVG(r.latency) AS avg_latency
        FROM batch_runs br
        JOIN batch_results r ON br.id = r.batch_run_id
        WHERE br.user_id = %s
    """
    params = [user_id]
    if days_filter == 'today':
        query += " AND DATE(br.created_at) = CURRENT_DATE"
    elif days_filter == 'yesterday':
        query += " AND DATE(br.created_at) = CURRENT_DATE - INTERVAL '1 day'"
    elif days_filter == '7':
        query += " AND br.created_at >= NOW() - INTERVAL '7 days'"
    elif days_filter == '30':
        query += " AND br.created_at >= NOW() - INTERVAL '30 days'"
        
    query += " GROUP BY br.prompting_technique"

    cur.execute(query, tuple(params))

    data = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify(data)

@app.route("/api/batch_runs")
@login_required
def get_batch_runs():
    user_id = session.get("user_id")
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT id, model, prompting_technique, created_at
        FROM batch_runs
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify(rows)

@app.route("/api/technique_comparison")
@login_required
def technique_comparison():

    batch_run_id = request.args.get("batch_run_id")
    days_filter = request.args.get("days_filter", "all")
    user_id = session.get("user_id")

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if batch_run_id:
        cur.execute("""
            SELECT br.prompting_technique,
                   COUNT(*) as total,
                   SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) as fr,
                   SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) as nfr,
                   AVG(r.latency) as avg_latency
            FROM batch_runs br
            JOIN batch_results r ON br.id = r.batch_run_id
            WHERE br.id = %s AND br.user_id = %s
            GROUP BY br.prompting_technique
        """, (batch_run_id, user_id))
    else:
        query = """
            SELECT br.prompting_technique,
                   COUNT(*) as total,
                   SUM(CASE WHEN r.classification='FR' THEN 1 ELSE 0 END) as fr,
                   SUM(CASE WHEN r.classification='NFR' THEN 1 ELSE 0 END) as nfr,
                   AVG(r.latency) as avg_latency
            FROM batch_runs br
            JOIN batch_results r ON br.id = r.batch_run_id
            WHERE br.user_id = %s
        """
        params = [user_id]
        if days_filter == 'today':
            query += " AND DATE(br.created_at) = CURRENT_DATE"
        elif days_filter == 'yesterday':
            query += " AND DATE(br.created_at) = CURRENT_DATE - INTERVAL '1 day'"
        elif days_filter == '7':
            query += " AND br.created_at >= NOW() - INTERVAL '7 days'"
        elif days_filter == '30':
            query += " AND br.created_at >= NOW() - INTERVAL '30 days'"
            
        query += " GROUP BY br.prompting_technique"
        
        cur.execute(query, tuple(params))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify(rows)


# ===============================
# RUN
# ===============================

if __name__ == '__main__':
    app.run(debug=True, port=5000)