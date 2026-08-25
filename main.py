import sys
import os
import time

# Kaggle uses exec() to run the script, which means __file__ is NOT defined!
sys.path.append("/kaggle_simulations/agent")
if '__file__' in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from heuristic_agent import HeuristicAgent
except Exception as e:
    print(f"[FATAL ERROR] Agent failed to load: {e}")
    HeuristicAgent = None

_global_agent = None

def agent(obs):
    """
    Main entry point for the Kaggle environment.
    Delegates all logic to the HeuristicAgent state machine.
    """
    global _global_agent
    
    if HeuristicAgent is None:
        print("[ERROR] Failed to import HeuristicAgent.")
        return {"farmer": ["PASS"], "hands": [], "market": []}
        
    if _global_agent is None:
        _global_agent = HeuristicAgent()
        
    # We can measure turn time just to be safe
    try:
        start_time = time.perf_counter()
        action = _global_agent(obs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if elapsed_ms > 500:
            print(f"[WARN] Turn took {elapsed_ms:.2f}ms! Be careful with the 1s limit.")
    except Exception as e:
        import traceback
        print(f"[FATAL ERROR] Agent crashed during turn: {e}")
        traceback.print_exc()
        action = {"farmer": ["PASS"], "hands": [], "market": []}
        
    return action
