from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
from code_integration import classify
import re
import time
import os
import pandas as pd
import random
app = Flask(__name__)

# --- Mock Data & Logic ---

MODELS = ["groq_gpt", "groq_llama3", "gemini", "cohere", "claude", "mistral"]
CATEGORIES = ["Accuracy", "Usability", "Performance", "Efficiency", "Security", "Privacy", "Fairness & Bias", "Explainability", "Interpretability", "Transparency", "Accessibility", "Reliability", "Robustness", "Maintainability", "Scalability", "Interoperability", "Completeness & Consistency", "Trust", "Safety & Governance"]

def parse_backend_response(raw_response, model, strategy, latency=0):
    """
    Parses the 3-line string response from backend.py into a dict
    expected by the frontend.
    """
    
    # 1. Is NFR: Yes/No
    # 2. NFR Type: ...
    # 3. Reason: ...
    
    classification = "NFR"
    category = None
    reason = "No reason provided"
    
    lines = raw_response.split('\n')
    for line in lines:
        if "Is NFR:" in line:
            val = line.split(":", 1)[1].strip().lower()
            classification = "NFR" if "yes" in val else "FR"
        elif "NFR Type:" in line:
            category = line.split(":", 1)[1].strip()
            if category == "-" or category.lower() in ["none", "n/a", "na"] or not category:
                category = None
        elif "Reason:" in line:
            reason = line.split(":", 1)[1].strip()

    # Force category to None if it matches FR
    if classification == "FR":
        category = None

    return {
        "classification": classification,
        "category": category,
        "classification_full": f"{classification} - {category}" if category else classification,
        "confidence": "N/A", # Backend doesn't return confidence score yet
        "model": model,
        "strategy": strategy,
        "latency": latency,
        "reason": reason,
        "raw_response": raw_response
    }

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html', page='index')

@app.route('/single', methods=['GET', 'POST'])
def single():
    if request.method == 'POST':
        data = request.json
        story = data.get('story')
        model = data.get('model', 'groq_gpt') # Default to a valid backend model
        strategy = data.get('strategy', 'Zero-shot')
        
        # Map frontend model names to backend model keys
        model_map = {
            "ChatGPT": "groq_gpt",
            "Gemini": "gemini",
            "Claude": "claude",
            "Groq": "groq_llama3",
            "Cohere": "cohere",
            "Mistral": "mistral"
        }
        backend_model = model_map.get(model, "groq_gpt")

        # Map frontend strategy names to backend technique names
        strategy_map = {
            "Zero-shot": "zero_shot",
            "Few-shot": "few_shot",
            "Chain-of-Thought": "chain_of_thought",
            "Role-Based": "role_based",
            "ReAct": "react"
        }
        # Fallback to key if not in map (or lowercased)
        technique = strategy_map.get(strategy, "zero_shot")
        
        # Call backend
        try:
            start_t = time.time()
            raw_response = classify(backend_model, story, technique)
            latency = round(time.time() - start_t, 2)
            result = parse_backend_response(raw_response, model, strategy, latency)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e), "classification": "Error", "category": None}), 500
    return render_template('single.html', page='single')

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        # Get parameters
        count = int(request.form.get('count', 10))
        model = request.form.get('model', 'groq_gpt')
        strategy = request.form.get('strategy', 'Zero-shot')
        
        # Map frontend model names to backend model keys
        model_map = {
            "ChatGPT": "groq_gpt",
            "Gemini": "gemini",
            "Claude": "claude",
            "Groq": "groq_llama3",
            "Cohere": "cohere",
            "Mistral": "mistral"
        }
        backend_model = model_map.get(model, "groq_gpt")
        
        results = []
        fr_count = 0
        nfr_count = 0
        total_time = 0
        
        # Handle File Upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                df = pd.read_csv(file)
                # Check for likely column names
                story_col = None
                for col in ['story', 'user_story', 'User Story', 'text', 'content']:
                    if col in df.columns:
                        story_col = col
                        break
                
                if not story_col:
                     return jsonify({"error": "CSV must contain a column named 'story', 'user_story', or 'text'."}), 400
                
                # Rename to 'story' for consistency
                df = df.rename(columns={story_col: 'story'})
                
                # If count is specified, limit rows? Or process all? 
                # Let's limit to count if it's reasonable, or just process top N for demo speed
                if count > len(df):
                    count = len(df)
                sampled_df = df.head(count).reset_index(drop=True)

            except Exception as e:
                return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

        else:
            # Fallback to existing batch logic (from directory)
            # Read batch data
            batch_dir = "batches"
            if not os.path.exists(batch_dir):
                return jsonify({"error": "No batch data found. Please run prepare_data.py first."}), 500
            
            # Pick first batch file for demo purposes
            batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.csv')]
            if not batch_files:
                 return jsonify({"error": "No CSV files in batches folder."}), 500
            
            df = pd.read_csv(os.path.join(batch_dir, batch_files[0]))
            
            # Sample 'count' rows
            if count > len(df):
                count = len(df)
            sampled_df = df.sample(n=count).reset_index(drop=True)
        
        def generate():
            fr_count = 0
            nfr_count = 0
            total_time = 0
            processed_so_far = 0
            category_counts = {c: 0 for c in CATEGORIES}

            for index, row in sampled_df.iterrows():
                story = row['story'] # Assuming 'story' column exists
                
                try:
                    # Use selected strategy
                    strategy_map = {
                        "Zero-shot": "zero_shot",
                        "Few-shot": "few_shot",
                        "Chain-of-Thought": "chain_of_thought",
                        "Role-Based": "role_based",
                        "ReAct": "react"
                    }
                    technique = strategy_map.get(strategy, "zero_shot")

                    start_t = time.time()
                    raw_response = classify(backend_model, story, technique) 
                    latency = round(time.time() - start_t, 2)
                    res = parse_backend_response(raw_response, model, strategy, latency)
                except Exception as e:
                    res = {"classification": "Error", "category": None, "latency": 0.0, "error": str(e)}

                if res['classification'] == 'FR':
                    fr_count += 1
                else:
                    nfr_count += 1
                    cat = res.get('category')
                    # Clean up category matching
                    if cat:
                        # Simple matching if backend returns slightly different string
                        matched = False
                        for c in CATEGORIES:
                            if c.lower() in cat.lower():
                                category_counts[c] += 1
                                matched = True
                                break
                        if not matched: 
                            # If category is "Performance" it matches, if "Non-Functional" maybe not.
                            pass

                total_time += res.get('latency', 0)
                processed_so_far += 1
                
                # Yield result immediately
                yield json.dumps({
                    "type": "result",
                    "story": story,
                    "result": res,
                    "current_stats": {
                        "total": processed_so_far,
                        "fr_count": fr_count,
                        "nfr_count": nfr_count,
                        "avg_time": round(total_time / processed_so_far, 2) if processed_so_far > 0 else 0,
                        "category_counts": category_counts
                    }
                }) + "\n"
                
            # Final summary (optional, as client updates stats incrementally)
            yield json.dumps({
                "type": "summary",
                "summary": {
                    "total": processed_so_far,
                    "fr_count": fr_count,
                    "nfr_count": nfr_count,
                    "avg_time": round(total_time / processed_so_far, 2) if processed_so_far > 0 else 0,
                    "category_counts": category_counts
                }
            }) + "\n"

        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

    return render_template('batch.html', page='batch')

@app.route('/comparison')
def comparison():
    # Mock data for initial render or API call
    return render_template('comparison.html', page='comparison')

@app.route('/api/comparison-data')
def comparison_data():
    # Check if real results exist
    results_file = "binary_results.csv"
    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            # Group by Model and take average of metrics
            summary = df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].mean().reset_index()
            # Rename columns to lowercase for frontend compatibility
            data = []
            for _, row in summary.iterrows():
                data.append({
                    "model": row['Model'],
                    "accuracy": round(row['Accuracy'], 2),
                    "precision": round(row['Precision'], 2),
                    "recall": round(row['Recall'], 2),
                    "f1": round(row['F1'], 2),
                    "avg_latency": 0.0 # Not in binary_results.csv usually, placeholder
                })
            # Sort by F1 desc
            data.sort(key=lambda x: x['f1'], reverse=True)
            return jsonify(data)
        except Exception as e:
            print(f"Error reading results: {e}")
            # Fallback to mock if error

    # Generate mock comparison metrics (Fallback)
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
    # Sort by accuracy desc for highlight
    data.sort(key=lambda x: x['accuracy'], reverse=True)
    return jsonify(data)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', page='analytics')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
