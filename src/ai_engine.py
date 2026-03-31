import os
import joblib
import pandas as pd
import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup

# --- CONSTANTS ---
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    metadata: dict  # score, tags, etc.
    ml_prediction: Optional[str]
    llm_analysis: Optional[str]
    final_verdict: Optional[str]
    api_key: str
    provider: str  # 'OpenAI' or 'Google'

# --- ML UTILITIES ---
def clean_html(text):
    if not isinstance(text, str): return ""
    soup = BeautifulSoup(text, "lxml")
    for tag in soup.find_all(["code", "pre", "script", "style"]):
        tag.decompose()
    return re.sub(r'\s+', ' ', soup.get_text(separator=" ")).strip()

def get_ml_prediction(question_text, score=0, tag_count=0):
    try:
        if not os.path.exists(os.path.join(MODELS_DIR, "vectorizer.joblib")):
            return "Unknown (Model missing)"
            
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
        
        clean_text = clean_html(question_text)
        X_tfidf = vectorizer.transform([clean_text])
        
        # Simple manual scaling to match training logic if scaler not saved
        X_numeric = [[score, len(clean_text), tag_count]] 
        
        from scipy.sparse import hstack
        X = hstack([X_tfidf, X_numeric])
        
        prediction = lr_model.predict(X)[0]
        return prediction
    except Exception as e:
        print(f"ML Prediction Error: {e}")
        return f"Unknown ({str(e)})"

# --- NODES ---
def ml_classification_node(state: AgentState):
    print("--- Executing ML Classification Node ---")
    question = state["question"]
    meta = state.get("metadata", {})
    
    prediction = get_ml_prediction(
        question, 
        score=meta.get("score", 0), 
        tag_count=meta.get("tag_count", 0)
    )
    
    return {"ml_prediction": prediction}

def llm_reasoning_node(state: AgentState):
    print("--- Executing LLM Reasoning Node ---")
    if not state["api_key"]:
        return {"llm_analysis": "No API Key provided. Skipping LLM analysis."}
    
    try:
        # Initialize LLM
        if state["provider"] == "OpenAI":
            llm = ChatOpenAI(api_key=state["api_key"], model="gpt-4o")
        elif state["provider"] == "Groq":
            llm = ChatGroq(api_key=state["api_key"], model="llama-3.3-70b-versatile")
        else:
            llm = ChatGoogleGenerativeAI(google_api_key=state["api_key"], model="gemini-1.5-pro")
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert pedagogical analyst. Your job is to analyze exam questions for their difficulty and cognitive depth."),
            ("user", """
            Analyze the following exam question. 
            The ML model predicted a difficulty of: {ml_prediction}
            
            Question: {question}
            
            Provide a detailed reasoning covering:
            1. Bloom's Taxonomy Level (Remembering, Understanding, Applying, Analyzing, Evaluating, Creating)
            2. Conceptual Complexity
            3. Linguistic Complexity
            4. Why the ML model might have picked '{ml_prediction}'
            """)
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        res = chain.invoke({
            "question": state["question"],
            "ml_prediction": state["ml_prediction"]
        })
        
        return {"llm_analysis": res}
    except Exception as e:
        return {"llm_analysis": f"LLM node failed: {str(e)}"}

def synthesis_node(state: AgentState):
    print("--- Executing Synthesis Node ---")
    ml_pred = state.get("ml_prediction", "Unknown")
    llm_analysis = state.get("llm_analysis", "No analysis performed.")
    
    verdict = f"### Final Verdict: {ml_pred}\n\n{llm_analysis}"
    return {"final_verdict": verdict}

# --- GRAPH CONSTRUCTION ---
def create_analysis_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("ml_predictor", ml_classification_node)
    workflow.add_node("llm_reasoner", llm_reasoning_node)
    workflow.add_node("synthesizer", synthesis_node)
    
    # Define Edges
    workflow.add_edge(START, "ml_predictor")
    workflow.add_edge("ml_predictor", "llm_reasoner")
    workflow.add_edge("llm_reasoner", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

# --- ENTRY POINT ---
def run_ai_analysis(question, api_key, provider="Google", metadata=None):
    if metadata is None:
        metadata = {"score": 0, "tag_count": 0}
        
    app = create_analysis_graph()
    inputs = {
        "question": question,
        "metadata": metadata,
        "api_key": api_key,
        "provider": provider,
        "ml_prediction": None,
        "llm_analysis": None,
        "final_verdict": None
    }
    
    final_state = app.invoke(inputs)
    return final_state
