"""
Interactive Classification Interface
A Streamlit-based web interface for FR/NFR classification using different models and prompts.

Usage:
streamlit run interactive_interface.py

Requirements:
pip install streamlit plotly
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
import os

# Import our components (make sure these files are in the same directory)
try:
    from enhanced_base_model import ModelWrapper, EnhancedEvaluator, PromptTemplates
    from comprehensive_comparison import ComprehensiveComparison
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure enhanced_base_model.py and comprehensive_comparison.py are in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="FR/NFR Classification Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - streamlined for performance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class InteractiveClassifier:
    """Interactive classification interface"""
    
    def __init__(self):
        if 'wrapper' not in st.session_state:
            st.session_state.wrapper = ModelWrapper(cache_enabled=True)
        if 'evaluator' not in st.session_state:
            st.session_state.evaluator = EnhancedEvaluator()
        if 'classification_history' not in st.session_state:
            st.session_state.classification_history = []
        
        self.wrapper = st.session_state.wrapper
        self.evaluator = st.session_state.evaluator
        
        # Model configurations
        self.models = {
            'Gemini 2.5 Pro': self.wrapper.classify_with_gemini,
            'Groq LLaMA 3.3': self.wrapper.classify_with_groq_llama,
            'Groq DeepSeek': self.wrapper.classify_with_groq_deepseek,
            'Cohere Command R+': self.wrapper.classify_with_cohere,
            'Claude 3 Haiku': self.wrapper.classify_with_claude,
            'Mistral Local': self.wrapper.classify_with_mistral
        }
        
        self.prompts = {
            'Zero Shot': PromptTemplates.BASIC_PROMPT,
            'Few Shot': PromptTemplates.ENHANCED_PROMPT,
            'Chain Of Thought': PromptTemplates.FEW_SHOT_PROMPT
        }

def main():
    # Initialize the classifier
    classifier = InteractiveClassifier()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 FR/NFR Classification Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose Model",
            list(classifier.models.keys()),
            index=0
        )
        
        # Prompt selection
        selected_prompt = st.selectbox(
            "Choose Prompt Strategy",
            list(classifier.prompts.keys()),
            index=1  # Default to Enhanced
        )
        
        st.divider()
        
        # Navigation
        st.header("📱 Navigation")
        page = st.radio(
            "Select Page",
            ["Single Classification", "Batch Processing", "Model Comparison", "Analytics Dashboard"],
            index=0
        )
    
    # Main content based on selected page
    if page == "Single Classification":
        single_classification_page(classifier, selected_model, selected_prompt)
    elif page == "Batch Processing":
        batch_processing_page(classifier, selected_model, selected_prompt)
    elif page == "Model Comparison":
        model_comparison_page(classifier)
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()

def single_classification_page(classifier, selected_model, selected_prompt):
    """Single user story classification interface"""
    st.header("📝 Single Story Classification")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input User Story")
        user_story = st.text_area(
            "Enter your user story:",
            height=150,
            placeholder="Example: As a user, I want the system to respond within 2 seconds so that I have a smooth experience."
        )
        
        if st.button("🔍 Classify Story", type="primary", use_container_width=True):
            if user_story.strip():
                classify_single_story(classifier, user_story, selected_model, selected_prompt)
            else:
                st.warning("Please enter a user story to classify.")
    
    with col2:
        st.subheader("ℹ️ Quick Reference")
        st.markdown("""
        **FR:** System features/capabilities  
        **NFR:** Quality attributes/constraints
        """)
        
        # Example stories - simplified
        st.subheader("💡 Examples")
        if st.button("FR: Login Feature", use_container_width=True):
            st.session_state.example_story = "As a user, I want to log into the system using my email and password."
        if st.button("NFR: Performance", use_container_width=True):
            st.session_state.example_story = "As a user, I want the login process to complete within 3 seconds."
    
    # Show recent classifications - optimized
    if st.session_state.classification_history:
        st.divider()
        st.subheader("📊 Recent Classifications")
        history_df = pd.DataFrame(st.session_state.classification_history[-5:])  # Last 5 for performance
        st.dataframe(
            history_df[['timestamp', 'story_preview', 'model', 'classification']],
            use_container_width=True,
            hide_index=True
        )

def classify_single_story(classifier, user_story, selected_model, selected_prompt):
    """Classify a single user story"""
    with st.spinner(f"Classifying with {selected_model}..."):
        try:
            start_time = time.time()
            
            # Get model function and prompt template
            model_fn = classifier.models[selected_model]
            prompt_template = classifier.prompts[selected_prompt]
            
            # Classify
            response = model_fn(user_story, prompt_template)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Parse response
            binary_pred, type_pred, reason = classifier.evaluator.parse_response(response)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if binary_pred == "NFR":
                    st.success(f"**Classification: NFR**")
                elif binary_pred == "FR":
                    st.info(f"**Classification: FR**")
                else:
                    st.error(f"**Classification: {binary_pred}**")
            
            with col2:
                st.metric("NFR Type", type_pred if type_pred != "None" else "N/A")
            
            with col3:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            # Detailed analysis - streamlined
            st.subheader("🔍 Analysis")
            
            # Confidence calculation
            confidence = "High" if len(reason) > 20 and "❌" not in response else "Medium"
            if "PARSE_ERROR" in binary_pred:
                confidence = "Low"
            
            confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
            
            st.markdown(f"**Reasoning:** {reason}")
            st.markdown(f"**Confidence:** {confidence_color[confidence]} {confidence}")
            
            with st.expander("View Model Response"):
                st.code(response, language="text")
            
            # Save to history
            st.session_state.classification_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'story': user_story,
                'story_preview': user_story[:50] + "..." if len(user_story) > 50 else user_story,
                'model': selected_model,
                'prompt': selected_prompt,
                'classification': binary_pred,
                'nfr_type': type_pred,
                'reason': reason,
                'confidence': confidence,
                'processing_time': processing_time,
                'raw_response': response
            })
            
        except Exception as e:
            st.error(f"Classification failed: {str(e)}")

def batch_processing_page(classifier, selected_model, selected_prompt):
    """Batch processing interface - optimized"""
    st.header("📦 Batch Processing")
    
    uploaded_file = st.file_uploader("Upload CSV (must have 'user_story' column)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'user_story' not in df.columns:
                st.error("CSV file must contain a 'user_story' column")
                return
            
            st.success(f"✅ Loaded {len(df)} stories")
            
            with st.expander("📋 Preview"):
                st.dataframe(df.head(5))
            
            max_stories = st.slider("Stories to process", 1, min(100, len(df)), min(25, len(df)))
            
            if st.button("🚀 Process", type="primary"):
                process_batch(classifier, df, selected_model, selected_prompt, max_stories)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        if st.button("📊 Try Sample Data"):
            create_sample_data_demo(classifier, selected_model, selected_prompt)

def process_batch(classifier, df, selected_model, selected_prompt, max_stories):
    """Process multiple user stories - optimized"""
    stories_to_process = df['user_story'].head(max_stories).tolist()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    model_fn = classifier.models[selected_model]
    prompt_template = classifier.prompts[selected_prompt]
    
    for i, story in enumerate(stories_to_process):
        status_text.text(f"Processing {i+1}/{len(stories_to_process)}...")
        
        try:
            start_time = time.time()
            response = model_fn(story, prompt_template)
            binary_pred, type_pred, reason = classifier.evaluator.parse_response(response)
            
            results.append({
                'user_story': story,
                'classification': binary_pred,
                'nfr_type': type_pred,
                'processing_time': time.time() - start_time
            })
        except Exception as e:
            results.append({
                'user_story': story,
                'classification': 'ERROR',
                'nfr_type': 'ERROR',
                'processing_time': 0
            })
        
        progress_bar.progress((i + 1) / len(stories_to_process))
        time.sleep(0.05)  # Reduced wait time
    
    status_text.text("✅ Completed!")
    results_df = pd.DataFrame(results)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(results_df))
    col2.metric("FR", (results_df['classification'] == 'FR').sum())
    col3.metric("NFR", (results_df['classification'] == 'NFR').sum())
    col4.metric("Avg Time", f"{results_df['processing_time'].mean():.2f}s")
    
    # Visualization
    classification_counts = results_df['classification'].value_counts()
    fig = px.pie(
        values=classification_counts.values,
        names=classification_counts.index,
        title="Classification Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Results
    st.subheader("📊 Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Download
    st.download_button(
        "📥 Download CSV",
        data=results_df.to_csv(index=False),
        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def model_comparison_page(classifier):
    """Model comparison interface - optimized"""
    st.header("⚖️ Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_models = st.multiselect(
            "Models:",
            list(classifier.models.keys()),
            default=list(classifier.models.keys())[:2]
        )
    
    with col2:
        selected_prompts = st.multiselect(
            "Prompts:",
            list(classifier.prompts.keys()),
            default=['Few Shot']
        )
    
    # Simplified test data options
    test_option = st.radio("Test Data:", ["Sample Stories", "Upload File"])
    
    test_stories = []
    
    if test_option == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            test_df = pd.read_csv(uploaded_file)
            if 'user_story' in test_df.columns:
                test_stories = test_df['user_story'].head(10).tolist()  # Limit to 10
                st.success(f"Loaded {len(test_stories)} stories")
    else:
        test_stories = [
            "As a user, I want to log into the system with my credentials.",
            "As a user, I want the system to respond within 2 seconds.",
            "As a user, I want to search for products by name.",
            "As a user, I want my data to be encrypted and secure."
        ]
        st.info(f"Using {len(test_stories)} sample stories")
    
    if test_stories and selected_models and selected_prompts:
        if st.button("🚀 Run Comparison", type="primary"):
            run_model_comparison(classifier, test_stories, selected_models, selected_prompts)

def run_model_comparison(classifier, test_stories, selected_models, selected_prompts):
    """Run model comparison - optimized"""
    results = []
    total = len(selected_models) * len(selected_prompts)
    current = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for model_name in selected_models:
        for prompt_name in selected_prompts:
            current += 1
            status_text.text(f"Testing {model_name} ({current}/{total})...")
            
            model_fn = classifier.models[model_name]
            prompt_template = classifier.prompts[prompt_name]
            
            predictions = []
            times = []
            
            for story in test_stories:
                start = time.time()
                try:
                    response = model_fn(story, prompt_template)
                    pred, _, _ = classifier.evaluator.parse_response(response)
                    predictions.append(pred)
                except:
                    predictions.append("ERROR")
                times.append(time.time() - start)
                time.sleep(0.05)
            
            results.append({
                'Model': model_name,
                'Prompt': prompt_name,
                'Avg_Time': sum(times) / len(times),
                'Error_Rate': predictions.count("ERROR") / len(predictions)
            })
            
            progress_bar.progress(current / total)
    
    status_text.text("✅ Completed!")
    results_df = pd.DataFrame(results)
    
    st.subheader("📊 Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Single chart
    fig = px.bar(
        results_df,
        x='Model',
        y='Avg_Time',
        color='Prompt',
        title="Processing Time Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

def analytics_dashboard_page():
    """Analytics dashboard - optimized"""
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.classification_history:
        st.info("No classification history available. Classify some user stories to see analytics!")
        return
    
    df = pd.DataFrame(st.session_state.classification_history)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(df))
    with col2:
        st.metric("FR", (df['classification'] == 'FR').sum())
    with col3:
        st.metric("NFR", (df['classification'] == 'NFR').sum())
    with col4:
        st.metric("Avg Time", f"{df['processing_time'].mean():.2f}s")
    
    # Key visualizations only
    col1, col2 = st.columns(2)
    
    with col1:
        # Classification distribution
        classification_counts = df['classification'].value_counts()
        fig_dist = px.pie(
            values=classification_counts.values,
            names=classification_counts.index,
            title="Classification Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Model usage
        model_counts = df['model'].value_counts()
        fig_models = px.bar(
            x=model_counts.index,
            y=model_counts.values,
            title="Model Usage"
        )
        st.plotly_chart(fig_models, use_container_width=True)
    
    # Recent activity
    st.subheader("📋 Recent Activity")
    recent_df = df.tail(10)[['timestamp', 'story_preview', 'model', 'classification']].copy()
    st.dataframe(recent_df, use_container_width=True, hide_index=True)
    
    # Export
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        if st.button("Clear History"):
            st.session_state.classification_history = []
            st.success("History cleared!")
            st.rerun()

def create_sample_data_demo(classifier, selected_model, selected_prompt):
    """Demo with sample data - optimized"""
    sample_data = [
        {"story": "As a user, I want to log into the system using my email and password.", "expected": "FR"},
        {"story": "As a user, I want the system to respond within 2 seconds.", "expected": "NFR"},
        {"story": "As a user, I want to search for products by name.", "expected": "FR"}
    ]
    
    results = []
    model_fn = classifier.models[selected_model]
    prompt_template = classifier.prompts[selected_prompt]
    
    for item in sample_data:
        try:
            response = model_fn(item["story"], prompt_template)
            pred, _, _ = classifier.evaluator.parse_response(response)
            match = "✅" if item["expected"] == pred else "❌"
        except:
            pred = "ERROR"
            match = "❌"
        
        results.append({
            "Story": item["story"][:50] + "...",
            "Expected": item["expected"],
            "Predicted": pred,
            "Match": match
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    matches = (results_df["Match"] == "✅").sum()
    st.success(f"Accuracy: {matches}/{len(results_df)} ({matches/len(results_df):.1%})")

if __name__ == "__main__":
    main()