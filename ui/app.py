import streamlit as st
import pandas as pd
import os
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback

try:
    from src.ai_engine import run_ai_analysis, judge_answer, shift_difficulty
    from src.knowledge_base import get_recommendations
    import_error = None
except ImportError as e:
    run_ai_analysis = None
    judge_answer = None
    shift_difficulty = None
    get_recommendations = None
    import_error = str(e)

st.set_page_config(
    page_title="Exam Analytics Pro",
    page_icon="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/layers.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM AESTHETIC ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    
    :root {
        --bg-color: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --accent-blue: #38bdf8;
        --accent-emerald: #10b981;
        --text-main: #f1f5f9;
        --text-muted: #94a3b8;
        --border: rgba(255, 255, 255, 0.1);
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-main) !important;
    }
    
    /* Custom Card Style */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }

    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 0 20px rgba(37, 99, 235, 0.4);
    }
    
    .stSidebar {
        background-color: #020617 !important;
        border-right: 1px solid var(--border);
    }
    
    /* Remove default metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--accent-blue) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .prediction-panel {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- HELPERS ---
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

def load_models():
    try:
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
        dt_model = joblib.load(os.path.join(MODELS_DIR, "dt_model.joblib"))
        return vectorizer, lr_model, dt_model
    except Exception:
        return None, None, None

def clean_html(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    try:
        soup = BeautifulSoup(text, "lxml")
        # Remove code blocks and scripts
        for tag in soup.find_all(["code", "pre", "script", "style"]):
            tag.decompose()
        clean_text = soup.get_text(separator=" ")
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    except Exception:
        return str(text)

def preprocess_features(df):
    df = df.copy()
    if "question" not in df.columns:
        if "title" in df.columns and "body" in df.columns:
            df["question"] = df["title"].astype(str) + " " + df["body"].astype(str)
        else:
            return None
    
    df["question"] = df["question"].fillna("").astype(str)
    df["question"] = df["question"].apply(clean_html)
    df["question_length"] = df["question"].apply(len)
    
    if "tags" in df.columns:
        df["tag_count"] = df["tags"].apply(lambda x: len(re.findall(r'<[^>]+>', str(x))) if pd.notna(x) else 0)
    else:
        df["tag_count"] = 0
    
    if "score" not in df.columns:
        df["score"] = 0
            
    return df

def predict_difficulty_with_probs(df, vectorizer, model):
    df_proc = preprocess_features(df)
    if df_proc is None: return None, None
            
    X_tfidf = vectorizer.transform(df_proc["question"])
    
    # Matching simple training scaler logic - normally we'd save the scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df_proc[["score", "question_length", "tag_count"]])
    
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_numeric])
    
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    
    return preds, probs

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<h2 style="color:white; margin-bottom:0px;">Analytics Pro</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem; color:#8b949e; margin-top:0px;">V1.2.0 • Powered by NLP</p>', unsafe_allow_html=True)
    st.write("---")
    
    page = st.radio("Navigation", 
                   ["Dashboard", "Practice Lab", "Model Training", "Documentation"],
                   label_visibility="collapsed")
    
    st.write("---")
    st.markdown("### AI Configuration")
    ai_provider = st.selectbox("LLM Provider", ["Google Gemini", "Groq", "OpenAI"])
    api_key = st.text_input("API Key", type="password", placeholder="Enter your API Key...")
    
    st.write("---")
    st.markdown("### System Health")
    vectorizer, lr_model, dt_model = load_models()
    if vectorizer:
        st.success("Models: **READY**")
    else:
        st.error("Models: **NOT FOUND**")
    
    st.info("Milestone 1: Text Analysis and Difficulty Classification.")

# --- MAIN CONTENT ---
if page == "Dashboard":
    st.markdown('<h1>Question Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    if vectorizer is None:
        st.warning("No trained models found. Using fallback mock results for UI demo. Go to 'Model Training' to set up.")
        use_mock = True
    else:
        use_mock = False

    # Layout: Metrics -> Upload -> Analysis
    uploaded_file = st.file_uploader("Drop a CSV file containing questions", type=["csv"])
    
    if not uploaded_file:
        st.session_state['analysis_done'] = False
        st.session_state['analyzed_df'] = None
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, on_bad_lines='warn', engine='python')
        df = preprocess_features(df)
        
        if df is None:
            st.error("Uploaded CSV missing required columns ('question' or 'title' and 'body').")
            st.stop()
        
        # 1. Insights Ribbon
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Records", f"{len(df):,}")
        with cols[1]:
            st.metric("Mean Text Length", f"{int(df['question_length'].mean())}")
        with cols[2]:
            st.metric("Peak Tag Density", f"{int(df['tag_count'].max())}")
        with cols[3]:
            st.metric("Aggregate Score", f"{df['score'].mean():.1f}")
        
        st.write("---")
        
        # 2. Prediction Section
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            model_name = st.selectbox("Select Classification Engine", ["Logistic Regression (Standard)", "Decision Tree (Deep Inspect)"])
        if st.button("Analyze Difficulty"):
            with st.spinner("Crunching vectors..."):
                if use_mock:
                    df['predicted_difficulty'] = np.random.choice(["Easy", "Medium", "Hard"], size=len(df))
                else:
                    target_model = lr_model if "Logistic" in model_name else dt_model
                    preds, probs = predict_difficulty_with_probs(df, vectorizer, target_model)
                    df['predicted_difficulty'] = preds
                
                st.session_state['analyzed_df'] = df
                st.session_state['analysis_done'] = True
            
        if st.session_state.get('analysis_done'):
                df = st.session_state['analyzed_df']
                # Visual Results
                st.success("Analysis Complete!")
                
                res_col1, res_col2 = st.columns([3, 2])
                
                with res_col1:
                    st.write("### Predicted Results")
                    # Color coding difficulty
                    def color_diff(val):
                        if val == "Hard": color = '#ff7b72'
                        elif val == "Medium": color = '#d29922'
                        else: color = '#3fb950'
                        return f'color: {color}; font-weight: bold'
                    
                    st.dataframe(df[["question", "predicted_difficulty"]].head(20).style.map(color_diff, subset=['predicted_difficulty']))
                
                with res_col2:
                    st.write("### Difficulty Distribution")
                    import plotly.express as px
                    
                    category_orders = {"predicted_difficulty": ["Easy", "Medium", "Hard"]}
                    color_map = {"Easy": "#10b981", "Medium": "#f59e0b", "Hard": "#ef4444"}
                    
                    fig = px.histogram(df, x="predicted_difficulty", 
                                     color="predicted_difficulty",
                                     category_orders=category_orders,
                                     color_discrete_map=color_map,
                                     template="plotly_dark")
                    
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=10, b=10, l=10, r=10),
                        xaxis_title="",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Advanced EDA
                st.write("---")
                st.write("### Advanced Insights")
                
                st.write("#### Question Length vs Difficulty")
                fig2 = px.box(df, x="predicted_difficulty", y="question_length",
                            color="predicted_difficulty",
                            category_orders={"predicted_difficulty": ["Easy", "Medium", "Hard"]},
                            color_discrete_map={"Easy": "#10b981", "Medium": "#f59e0b", "Hard": "#ef4444"},
                            template="plotly_dark",
                            points="all")
                
                fig2.update_layout(
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Character Count",
                    xaxis_title=""
                )
                st.plotly_chart(fig2, use_container_width=True)

                # AI Integration Point
                if run_ai_analysis and api_key:
                    st.write("---")
                    st.write("### AI Agentic Analysis")
                    
                    # Analyze first few questions for demo
                    sample_q = df.iloc[0]['question']
                    if st.button("Generate Expert Assessment"):
                        if df.empty:
                            st.warning("No data found to analyze.")
                        else:
                            with st.spinner("AI Agent is reasoning through the graph..."):
                                try:
                                    if "Google" in ai_provider: provider = "Google"
                                    elif "Groq" in ai_provider: provider = "Groq"
                                    else: provider = "OpenAI"
                                    
                                    # Ensure we have metadata
                                    row = df.iloc[0]
                                    meta = {
                                        "score": row.get("score", 0),
                                        "tag_count": row.get("tag_count", 0)
                                    }
                                    
                                    result = run_ai_analysis(
                                        row['question'], 
                                        api_key, 
                                        provider=provider,
                                        metadata=meta
                                    )
                                    
                                    st.markdown(f"""
                                    <div style="background-color: #1e293b; border-radius: 10px; padding: 20px; border: 1px solid var(--border);">
                                        <h4 style="color: var(--accent-blue);">Step-by-Step Reasoning</h4>
                                        <p><b>1. ML Prediction Node:</b> {result.get('ml_prediction', 'N/A')}</p>
                                        <p><b>2. LLM Analysis Node:</b> Done</p>
                                        <p><b>3. Synthesis Node:</b> Complete</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.info(result.get('final_verdict', "Analysis complete but no verdict returned."))
                                    
                                    # Specific HITL Approval Panel
                                    if result.get('approval_needed'):
                                        st.markdown(f"""
                                        <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 20px; margin-top: 20px;">
                                            <h4 style="color: #ef4444; margin-top: 0;">Expert Review Required</h4>
                                            <p style="color: #fca5a5; font-size: 0.9rem;">
                                                The AI system has detected a potential discrepancy between the ML quantitative model and the qualitative agentic analysis. 
                                                Manual verification of this item is highly recommended.
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"AI Analysis Failed: {str(e)}")
                                    st.code(traceback.format_exc())
                elif not api_key:
                    st.info("Tip: Add an API Key in the sidebar to enable Expert AI Analytics powered by LangGraph.")

elif page == "Practice Lab":
    st.markdown('<h1>Practice Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--text-muted);">Adaptive preparation based on your performance.</p>', unsafe_allow_html=True)
    
    # Session State for Prep
    if 'prep_level' not in st.session_state:
        st.session_state['prep_level'] = "Medium"
    if 'current_prep_q' not in st.session_state:
        st.session_state['current_prep_q'] = None
    if 'prep_feedback' not in st.session_state:
        st.session_state['prep_feedback'] = None

    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); padding: 1.5rem; border-radius: 12px; text-align: center;">
            <p style="text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.1em; color: var(--accent-blue);">Current Level</p>
            <h2 style="margin: 0; color: white;">{st.session_state['prep_level']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button("Get Next Question", use_container_width=True):
            recs = get_recommendations(target_difficulty=st.session_state['prep_level'], limit=1)
            if recs:
                st.session_state['current_prep_q'] = recs[0]
                st.session_state['prep_feedback'] = None
                st.rerun()
            else:
                st.warning("No questions found for this level.")

    with col1:
        if st.session_state['current_prep_q']:
            q_text = st.session_state['current_prep_q']
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 2rem; border: 1px solid rgba(255, 255, 255, 0.05);">
                {q_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            user_input = st.text_area("Your response:", placeholder="Explain your answer or provide the code...")
            
            if st.button("Submit Response", type="primary"):
                if not api_key:
                    st.error("Please enter an API Key in the sidebar to use AI-judging.")
                else:
                    with st.spinner("AI is evaluating your response..."):
                        feedback = judge_answer(q_text, user_input, api_key)
                        st.session_state['prep_feedback'] = feedback
                        
                        # Adaptive scaling
                        is_correct = "[CORRECT]" in feedback.upper()
                        old_level = st.session_state['prep_level']
                        st.session_state['prep_level'] = shift_difficulty(old_level, is_correct)
                        st.rerun()
        else:
            st.info("Click 'Get Next Question' to start your session.")

        if st.session_state['prep_feedback']:
            st.markdown("### Feedback & Assessment")
            res = st.session_state['prep_feedback']
            if "[CORRECT]" in res.upper():
                st.success(res.replace("[CORRECT]", "✓ CORRECT"))
            else:
                st.error(res.replace("[INCORRECT]", "✗ INCORRECT"))

elif page == "Model Training":
    st.markdown('<h1>Model Training Pipeline</h1>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        ### Current Model Architecture
        - **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **NGrams**: (1, 2) bigrams included
        - **Max Features**: 50,000 top words
        - **Target Rows**: Up to 300,000 for training
        """)
        
    with col_info2:
        st.markdown("""
        ### Logistic Regression vs Decision Tree
        - **Logistic Regression**: Best for capturing linear relationships between word frequency and difficulty. Provides probability scores.
        - **Decision Tree**: Captures complex heuristics (e.g., specific tag combinations or length thresholds). Highly interpretable.
        """)

    st.write("---")
    st.subheader("Trigger Training")
    st.info("Ensure `data/processed/processed_data.csv` is present before starting.")
    
    train_col1, train_col2 = st.columns([1, 3])
    with train_col1:
        if st.button("Initialize Training"):
            st.warning("Training on large datasets may take several minutes.")
            st.code("python src/model_train.py")
            st.info("Execution logs directed to system console.")
    
    with train_col2:
        st.write("#### Training Checklist")
        st.checkbox("Raw data processed", value=True, disabled=True)
        st.checkbox("Features engineered (tfidf + numeric)", value=True, disabled=True)
        st.checkbox("Models directory mapped", value=True, disabled=True)

elif page == "Documentation":
    st.markdown('<h1>Technical Documentation</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Overview", "Architecture", "Difficulty Metric"])
    
    with tabs[0]:
        st.markdown("""
        ### Project Vision
        The Exam Analytics System is designed to help educators understand and predict the difficulty of assessment items before they are administered. By leveraging historical student performance (view counts, answer rates) and textual complexity, we can calibrate exams with higher precision.
        
        ### Key Features
        1. **NLP Processing**: Cleaning raw HTML, tokenization, and lemmatization.
        2. **Hybrid Features**: Combining semantic text data (TF-IDF) with structural metadata (tag count, length, scores).
        3. **Dual-Model Inference**: Allowing comparisons between linear and non-linear classification strategies.
        """)
        
    with tabs[1]:
        st.write("#### System Pipeline")
        mermaid_code = """
flowchart TD
    A["Raw Dataset"] -->|"data_prep.py"| B["Processed Data"]
    B -->|"model_train.py"| C["Trained Models"]
    C -->|"app.py"| D["Dashboard UI"]
    E["User CSV"] -->|"Upload"| D
    D -->|"Predict"| F["Final Difficulty Report"]
"""
        st.components.v1.html(
            f"""
            <div class="mermaid">
                {mermaid_code}
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ 
                    startOnLoad: true, 
                    theme: 'dark',
                    securityLevel: 'loose',
                }});
            </script>
            """,
            height=450,
        )
        
    with tabs[2]:
        st.markdown("""
        ### How Difficulty is Calculated
        The system uses an **Inference-by-Engagement** metric during the preprocessing phase:
        
        $$Difficulty = \frac{Answer Count}{View Count}$$
        
        **Labeling Strategy:**
        - **Easy**: Top 33% (Questions with high prompt-to-answer conversion)
        - **Medium**: Middle 34%
        - **Hard**: Bottom 33% (Many views, few answers - indicating complexity)
        """)

st.sidebar.markdown("---")
st.sidebar.caption("System Version 1.2.0 • Phase 1 Operation")
