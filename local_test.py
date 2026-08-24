import json
import sys
import time
from kaggle_environments import make

import os
sys.path.insert(0, os.path.abspath('kaggric'))

from kaggric.main import agent

def run_local_test():
    print("Initializing environment...")
    # 500 steps = 20 days. Enough to guarantee several weeds.
    env = make("kaggriculture", configuration={"episodeSteps": 500}, debug=True)
    
    print("Running simulation: agent vs random...")
    start = time.time()
    env.run([agent, "random"])
    end = time.time()
    
    final = env.steps[-1]
    print("\n--- Match Results ---")
    for i, s in enumerate(final):
        role = "Your Agent" if i == 0 else "Random Agent"
        print(f"Player {i} ({role}): reward={s['reward']}, status={s['status']}")
        
    print(f"\nTotal elapsed time: {end - start:.1f} seconds")
    print("Replay saved successfully!")

if __name__ == "__main__":
    run_local_test()
