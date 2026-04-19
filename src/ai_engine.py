import os
import joblib
import pandas as pd
import re
import numpy as np
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from src.knowledge_base import get_retriever

# --- CONSTANTS ---
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    metadata: dict
    api_key: str
    provider: str  # 'Groq', 'OpenAI' or 'Google'
    ml_prediction: Optional[str]
    taxonomy_analysis: Optional[str] = None
    linguistic_analysis: Optional[str] = None
    critique: Optional[str] = None
    analysis_result: Optional[dict] = None
    final_verdict: Optional[str]
    context: Optional[str] = None
    approval_needed: bool = False
    iterations: int = 0

class AnalysisResult(BaseModel):
    difficulty_label: str = Field(description="The final difficulty rating: Easy, Medium, or Hard")
    bloom_taxonomy: str = Field(description="The primary Bloom's Taxonomy level identified")
    complexity_score: float = Field(description="Numeric complexity score from 0.0 to 1.0")
    linguistic_depth: str = Field(description="Brief assessment of the language and sentence structure complexity")
    reasoning: List[str] = Field(description="Bullet points justifying the classification")

def sanitize_serializable(obj):
    """
    Recursively converts numpy types and other non-standard types to native Python 
    types to ensure compatibility with msgpack/LangGraph serialization.
    """
    if isinstance(obj, dict):
        return {k: sanitize_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    return obj

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
        
        # Load and use the scaler if available
        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_numeric = scaler.transform([[score, len(clean_text), tag_count]])
        else:
            # Fallback to manual scaling if scaler missing (less accurate)
            X_numeric = [[score, len(clean_text), tag_count]] 
        
        from scipy.sparse import hstack
        X = hstack([X_tfidf, X_numeric])
        
        prediction = lr_model.predict(X)[0]
        # Convert numpy type to native Python type for serialization
        return sanitize_serializable(prediction)
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

# --- AGENT NODES ---

def retriever_node(state: AgentState):
    print("--- Executing Retriever Node ---")
    # Complexity assessment for routing
    complexity = 1.0 if len(state["question"]) > 500 else 0.5
    
    # Fetch Standard Pedagogical Context
    std_retriever = get_retriever(search_type="standard")
    std_docs = std_retriever.invoke(state["question"])
    context = "### Pedagogical Standards:\n" + "\n\n".join([f"- {d.page_content}" for d in std_docs])
    
    # Fetch Few-Shot Gold Standards
    gold_retriever = get_retriever(search_type="gold_standard")
    gold_docs = gold_retriever.invoke(state["question"])
    context += "\n\n### Reference Benchmarks (Few-Shot):\n" + "\n\n".join([f"- {d.page_content}" for d in gold_docs])
    
    return {"context": context}

def get_llm(state: AgentState):
    is_complex = len(state.get("question", "")) > 600
    
    if state["provider"] == "Groq" or not state["provider"]:
        # Dynamic routing within Groq
        model = "llama-3.3-70b-versatile" if is_complex else "llama3-8b-8192"
        return ChatGroq(api_key=state["api_key"], model=model)
    elif state["provider"] == "OpenAI":
        model = "gpt-4o" if is_complex else "gpt-4o-mini"
        return ChatOpenAI(api_key=state["api_key"], model=model)
    return ChatGoogleGenerativeAI(google_api_key=state["api_key"], model="gemini-1.5-pro")

def taxonomy_agent_node(state: AgentState):
    print("--- Executing Taxonomy Specialist Node ---")
    llm = get_llm(state)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialist in Bloom's Taxonomy. Use the provided context and reference benchmarks to identify the cognitive and linguistic depth."),
        ("user", """
        CONTEXT & BENCHMARKS:
        {context}
        
        NEW QUESTION TO ANALYZE:
        {question}
        
        ML PREDICTED DIFFICULTY:
        {ml_prediction}
        
        Identify the Bloom's Level and explain your reasoning based on the provided context.
        """)
    ])
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({
        "question": state["question"], 
        "ml_prediction": state["ml_prediction"],
        "context": state.get("context", "No context available.")
    })
    return {"taxonomy_analysis": res}

def linguistic_agent_node(state: AgentState):
    print("--- Executing Linguistic Specialist Node ---")
    llm = get_llm(state)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialist in readability and linguistic structure. Analyze the question's textual complexity."),
        ("user", "Question: {question}")
    ])
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({"question": state["question"]})
    return {"linguistic_analysis": res}

def critic_node(state: AgentState):
    print("--- Executing Critic Node ---")
    llm = get_llm(state)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an adjudicator. Compare the ML prediction with the specialist analyses. Identify any contradictions."),
        ("user", """
        ML Prediction: {ml_prediction}
        Taxonomy Analysis: {taxonomy_analysis}
        Linguistic Analysis: {linguistic_analysis}
        Question: {question}
        
        Is the ML prediction consistent with the qualitative analysis? Provide a consistency score (0-100).
        """)
    ])
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({
        "ml_prediction": state["ml_prediction"],
        "taxonomy_analysis": state["taxonomy_analysis"],
        "linguistic_analysis": state["linguistic_analysis"],
        "question": state["question"]
    })
    
    # Simple logic to detect if we need approval/refinement
    score_match = re.search(r'(\d+)', res)
    score = int(score_match.group(1)) if score_match else 100
    
    return {"critique": res, "approval_needed": score < 70, "iterations": state.get("iterations", 0) + 1}

def refiner_node(state: AgentState):
    print("--- Executing Refiner Node ---")
    llm = get_llm(state)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialist synthesizer. Address the critic's concerns and refine the final assessment."),
        ("user", "Question: {question}\nCritic Notes: {critique}")
    ])
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({"question": state["question"], "critique": state["critique"]})
    # This node could update the specific analyses if needed, or just prepare for final synthesis
    return {"taxonomy_analysis": f"Refined: {res}"}

def synthesizer_node(state: AgentState):
    print("--- Executing Final Synthesizer Node ---")
    llm = get_llm(state).with_structured_output(AnalysisResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Consolidate all analyses into a final structured report."),
        ("user", """
        Taxonomy: {taxonomy_analysis}
        Linguistics: {linguistic_analysis}
        Critic Notes: {critique}
        Original Question: {question}
        """)
    ])
    chain = prompt | llm
    res = chain.invoke({
        "taxonomy_analysis": state["taxonomy_analysis"],
        "linguistic_analysis": state["linguistic_analysis"],
        "critique": state["critique"],
        "question": state["question"]
    })
    return {"analysis_result": res.dict()}

def synthesis_node(state: AgentState):
    print("--- Executing Synthesis Node ---")
    ml_pred = state.get("ml_prediction", "Unknown")
    analysis = state.get("analysis_result", {})
    critique = state.get("critique", "No critique performed.")
    approval = state.get("approval_needed", False)
    
    verdict = f"### Final Verdict: {analysis.get('difficulty_label', ml_pred)}\n\n"
    if approval:
        verdict = "⚠️ **PENDING EXPERT APPROVAL**\n\n" + verdict
        
    verdict += f"- **Taxonomy**: {analysis.get('bloom_taxonomy', 'N/A')}\n"
    verdict += f"- **Linguistic Depth**: {analysis.get('linguistic_depth', 'N/A')}\n"
    verdict += f"- **Complexity Score**: {analysis.get('complexity_score', 0) * 100:.1f}%\n\n"
    
    verdict += "#### Expert Reasoning:\n"
    for step in analysis.get('reasoning', []):
        verdict += f"- {step}\n"
        
    if approval:
        verdict += f"\n---\n**Reviewer Notes (Consistency Score Low):**\n{critique}"
        
    return {"final_verdict": verdict}

# --- GRAPH CONSTRUCTION ---
def should_continue(state: AgentState):
    if state.get("iterations", 0) >= 2:
        return "synthesizer"
    if state.get("approval_needed"):
        return "refiner"
    return "synthesizer"

def create_analysis_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    
    # Add Nodes (unchanged)
    workflow.add_node("ml_predictor", ml_classification_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("taxonomy_expert", taxonomy_agent_node)
    workflow.add_node("linguistic_expert", linguistic_agent_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("reporter", synthesis_node)
    
    # Define Edges (unchanged)
    workflow.add_edge(START, "ml_predictor")
    workflow.add_edge("ml_predictor", "retriever")
    workflow.add_edge("retriever", "taxonomy_expert")
    workflow.add_edge("retriever", "linguistic_expert")
    workflow.add_edge(["taxonomy_expert", "linguistic_expert"], "critic")
    
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "refiner": "refiner",
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_edge("refiner", "critic")
    workflow.add_edge("synthesizer", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile(checkpointer=checkpointer)

# --- ENTRY POINT ---
def run_ai_analysis(question, api_key, provider="Google", metadata=None, thread_id="1"):
    if metadata is None:
        metadata = {"score": 0, "tag_count": 0}
        
    # Sanitize inputs to ensure msgpack serializability (NumPy -> Python types)
    metadata = sanitize_serializable(metadata)
    question = str(question) # Ensure string type
    
    # Setup persistence
    memory = MemorySaver()
    app = create_analysis_graph(checkpointer=memory)
    
    # Ensure thread_id is a string (numpy.int64 thread_ids can break serialization)
    config = {"configurable": {"thread_id": str(thread_id)}}
    
    inputs = {
        "question": question,
        "metadata": metadata,
        "api_key": api_key,
        "provider": provider,
        "ml_prediction": None,
        "taxonomy_analysis": None,
        "linguistic_analysis": None,
        "critique": None,
        "analysis_result": None,
        "final_verdict": None,
        "context": None,
        "approval_needed": False,
        "iterations": 0
    }
    
    # Final sanitization of all inputs to catch any stray non-serializable types
    inputs = sanitize_serializable(inputs)
    
    final_state = app.invoke(inputs, config=config)
    return final_state

# --- ADAPTIVE UTILITIES ---

def judge_answer(question, user_answer, api_key):
    """Uses Groq to evaluate a user's answer for correctness and depth."""
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert tutor. Evaluate the user's answer to the question. Be encouraging but rigorous."),
        ("user", f"Question: {question}\nUser Answer: {user_answer}\n\nIs this correct? Provide a brief explanation and a result: [CORRECT] or [INCORRECT].")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

def shift_difficulty(current_difficulty, is_correct):
    """Calibrates next question difficulty based on performance."""
    levels = ["Easy", "Medium", "Hard"]
    idx = levels.index(current_difficulty)
    
    if is_correct:
        # Move up if possible
        new_idx = min(idx + 1, 2)
    else:
        # Move down if possible
        new_idx = max(idx - 1, 0)
        
    return levels[new_idx]
