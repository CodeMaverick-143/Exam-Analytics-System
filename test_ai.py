import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_engine import run_ai_analysis

try:
    print("Testing AI Engine...")
    # Using an empty API Key and provider - it should at least enter the ML node
    res = run_ai_analysis(
        "Explain quantum mechanics.", 
        "", # No API key
        provider="Google"
    )
    print("Success! Final Verdict:")
    print(res['final_verdict'])
except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
