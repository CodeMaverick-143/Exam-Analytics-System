import os
from langchain_chroma import Chroma
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document

# --- CONFIGURATION ---
KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/knowledge_base'))
os.makedirs(KB_DIR, exist_ok=True)

# --- STANDARDS DATA ---
PEDAGOGICAL_STANDARDS = [
    {
        "title": "Bloom's Taxonomy: Remembering",
        "content": "Retrieving, recognizing, and recalling relevant knowledge from long-term memory. Typical verbs: Cite, define, describe, identify, label, list, match, name, quote, recall."
    },
    {
        "title": "Bloom's Taxonomy: Understanding",
        "content": "Determining the meaning of instructional messages, including oral, written, and graphic communication. Typical verbs: Abstract, classify, compare, contrast, demonstrate, differentiate, discuss, explain."
    },
    {
        "title": "Bloom's Taxonomy: Applying",
        "content": "Carrying out or using a procedure in a given situation. Typical verbs: Calculate, complete, compute, demonstrate, dramatize, employ, examine, illustrate, implement."
    },
    {
        "title": "Bloom's Taxonomy: Analyzing",
        "content": "Breaking material into constituent parts, detecting how the parts relate to one another and to an overall structure or purpose. Typical verbs: Analyze, arrange, break down, categorize, classify, compare, connect."
    },
    {
        "title": "Bloom's Taxonomy: Evaluating",
        "content": "Making judgments based on criteria and standards. Typical verbs: Appraise, argue, assess, choose, conclude, critique, decide, defend, estimate, judge, justify."
    },
    {
        "title": "Bloom's Taxonomy: Creating",
        "content": "Putting elements together to form a novel, coherent whole or make an original product. Typical verbs: Arrange, assemble, build, collect, combine, compile, compose, constitute, construct, create."
    },
    {
        "title": "Depth of Knowledge (DOK) Level 1: Recall",
        "content": "Recall of a fact, information, or procedure. Requires only rote memory or following simple steps."
    },
    {
        "title": "Depth of Knowledge (DOK) Level 2: Skill/Concept",
        "content": "Use of information or conceptual knowledge. Requires engagement beyond basic recall, often involving mental processing of decisions."
    },
    {
        "title": "Depth of Knowledge (DOK) Level 3: Strategic Thinking",
        "content": "Requires reasoning, developing a plan or a sequence of steps, and some complexity. More than one possible answer."
    }
]

GOLD_STANDARD_QUESTIONS = [
    {
        "question": "What is the result of 2 + 2?",
        "difficulty": "Easy",
        "taxonomy": "Remembering"
    },
    {
        "question": "Explain how photosynthesis converts light energy into chemical energy in plants.",
        "difficulty": "Medium",
        "taxonomy": "Understanding"
    },
    {
        "question": "Design a software architecture for a global real-time ride-sharing application, considering latency, consistency, and availability.",
        "difficulty": "Hard",
        "taxonomy": "Creating"
    },
    {
        "question": "Write a Python function to find the first non-repeating character in a string.",
        "difficulty": "Easy",
        "taxonomy": "Applying"
    },
    {
        "question": "What is the bias-variance tradeoff in machine learning?",
        "difficulty": "Medium",
        "taxonomy": "Understanding"
    },
    {
        "question": "Implement a self-attention mechanism from scratch using NumPy.",
        "difficulty": "Hard",
        "taxonomy": "Creating"
    },
    {
        "question": "What does the 'self' keyword represent in Python classes?",
        "difficulty": "Easy",
        "taxonomy": "Remembering"
    },
    {
        "question": "Compare and contrast L1 and L2 regularization techniques.",
        "difficulty": "Medium",
        "taxonomy": "Analyzing"
    }
]

def initialize_kb():
    print("Initializig Pedagogical Knowledge Base...")
    
    # We use a simple fake embedding or a lightweight local one if possible
    # For this demo, we'll use a standard community embedding if available, 
    # but to ensure it runs without huge model downloads here, I'll use a simpler approach.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        print("HuggingFaceEmbeddings not found, using deterministic fake embeddings for demo.")
        embeddings = DeterministicFakeEmbedding(size=384)

    documents = [
        Document(page_content=item["content"], metadata={"title": item["title"], "type": "standard"})
        for item in PEDAGOGICAL_STANDARDS
    ]
    
    documents += [
        Document(
            page_content=f"Question: {q['question']}\nDifficulty: {q['difficulty']}\nTaxonomy: {q['taxonomy']}", 
            metadata={"title": f"Gold Standard: {q['difficulty']}", "type": "gold_standard"}
        )
        for q in GOLD_STANDARD_QUESTIONS
    ]
    
    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=KB_DIR
    )
    print(f"Knowledge Base initialized with {len(documents)} standards in {KB_DIR}")
    return vector_store

def get_retriever(search_type="standard"):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        embeddings = DeterministicFakeEmbedding(size=384)
        
    vector_store = Chroma(
        persist_directory=KB_DIR,
        embedding_function=embeddings
    )
    
    # Check if empty - if so, initialize
    try:
        count = vector_store._collection.count()
        if count == 0:
            print("Knowledge Base is empty. Initializing...")
            vector_store = initialize_kb()
    except Exception:
        print("Knowledge Base DIR not found or inaccessible. Initializing...")
        vector_store = initialize_kb()
    
    search_filter = {"type": search_type}
    return vector_store.as_retriever(
        search_kwargs={"k": 3, "filter": search_filter}
    )

def get_recommendations(target_difficulty="Medium", limit=3):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        embeddings = DeterministicFakeEmbedding(size=384)
        
    vector_store = Chroma(
        persist_directory=KB_DIR,
        embedding_function=embeddings
    )
    
    # Check if empty - if so, initialize
    try:
        count = vector_store._collection.count()
        if count == 0:
            print("Knowledge Base is empty. Initializing...")
            vector_store = initialize_kb()
    except Exception:
        print("Knowledge Base missing. Initializing...")
        vector_store = initialize_kb()
    
    # Filter by difficulty for recommendation
    results = vector_store.get(where={"title": f"Gold Standard: {target_difficulty}"})
    
    recs = []
    if results["documents"]:
        import random
        # Randomly sample to provide variety
        docs = results["documents"]
        indices = random.sample(range(len(docs)), min(len(docs), limit))
        for i in indices:
            recs.append(docs[i])
            
    return recs

if __name__ == "__main__":
    initialize_kb()
