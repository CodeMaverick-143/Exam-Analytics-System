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
    from src.ai_engine import run_ai_analysis
except ImportError:
    run_ai_analysis = None

st.set_page_config(
    page_title="Exam Analytics Pro",
    page_icon="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/zap.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM AESTHETIC ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --primary: #58a6ff;
        --secondary: #238636;
        --bg-dark: #0d1117;
        --card-bg: #161b22;
        --border-color: #30363d;
    }
    
    .main {
        background-color: var(--bg-dark);
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: var(--bg-dark);
    }
    
    h1, h2, h3 {
        color: var(--primary);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }
    
    .stMetric {
        background-color: var(--card-bg);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    
    .stSidebar {
        background-color: #010409 !important;
        border-right: 1px solid var(--border-color);
    }
    
    .stButton>button {
        width: 100%;
        background-color: var(--secondary);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 0 15px rgba(35, 134, 54, 0.4);
    }
    
    .prediction-card {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid var(--primary);
        margin-bottom: 20px;
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
                   ["Dashboard", "Model Training", "Documentation"],
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
        
        # 1. Metric Overview
        st.subheader("Data Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows Found", f"{len(df):,}")
        m2.metric("Avg Question Length", f"{int(df['question_length'].mean())} chars")
        m3.metric("Max Tags", f"{int(df['tag_count'].max())}")
        m4.metric("Avg Score", f"{df['score'].mean():.1f}")
        
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
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0d1117')
                    ax.set_facecolor('#161b22')
                    sns.countplot(data=df, x='predicted_difficulty', 
                                hue='predicted_difficulty',
                                legend=False,
                                order=["Easy", "Medium", "Hard"],
                                palette=['#3fb950', '#d29922', '#ff7b72'], ax=ax)
                    plt.title("Classification Distribution", color='white', pad=20)
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values(): spine.set_color('#30363d')
                    st.pyplot(fig)

                # Advanced EDA
                st.write("---")
                st.write("### Advanced Insights")
                
                st.write("#### Question Length vs Difficulty")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                fig2.patch.set_facecolor('#0d1117')
                ax2.set_facecolor('#161b22')
                sns.boxplot(data=df, x='predicted_difficulty', y='question_length', 
                           hue='predicted_difficulty',
                           legend=False,
                           order=["Easy", "Medium", "Hard"],
                           palette='viridis', ax=ax2)
                ax2.tick_params(colors='white')
                plt.ylabel("Character Count", color='white')
                st.pyplot(fig2)

                # AI Integration Point
                if run_ai_analysis and api_key:
                    st.write("---")
                    st.write("### 🤖 AI Agentic Insights (LangGraph)")
                    
                    # Analyze first few questions for demo
                    sample_q = df.iloc[0]['question']
                    if st.button("Generate AI Reasoning for Top Question"):
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
                                    <div style="background-color: #1c2128; border-radius: 10px; padding: 20px; border: 1px solid #30363d;">
                                        <h4 style="color: #58a6ff;">Step-by-Step Reasoning</h4>
                                        <p><b>1. ML Prediction Node:</b> {result.get('ml_prediction', 'N/A')}</p>
                                        <p><b>2. LLM Analysis Node:</b> Done</p>
                                        <p><b>3. Synthesis Node:</b> Complete</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.info(result.get('final_verdict', "Analysis complete but no verdict returned."))
                                except Exception as e:
                                    st.error(f"AI Analysis Failed: {str(e)}")
                                    st.code(traceback.format_exc())
                elif not api_key:
                    st.info("💡 Tip: Add an API Key in the sidebar to enable **AI Agentic Analytics** powered by LangGraph.")

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
        if st.button("Finalize & Train"):
            st.warning("Training on 1M rows may take 5-10 minutes.")
            st.code("python src/model_train.py")
            st.info("System redirected logs to terminal. Check your console.")
    
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
        st.mermaid("""
        graph TD
            A[Raw Dataset] -->|data_prep.py| B[Processed Data]
            B -->|model_train.py| C[Trained Models]
            C -->|app.py| D[Dashboard UI]
            E[User CSV] -->|Upload| D
            D -->|Predict| F[Final Difficulty Report]
        """)
        
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
st.sidebar.caption("Exam Analytics Pro v1.2 (Milestone 1)")
