import os
import sys
import math
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from kaggle_environments import make

# Add the current directory to sys.path so we can import our local heuristic_agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heuristic_agent import HeuristicAgent, SpyTracker

def make_obs(obs, spy_tracker) -> np.ndarray:
    player = obs["player"]
    opp = 1 - player
    
    farm     = obs["farms"][player]
    opp_farm = obs["farms"][opp]
    priv     = obs.get("private", {})
    market   = obs["market"]
    prices   = market["prices"]
    inv      = market["inventory"]

    # Time features
    step = obs.get("step", 0)
    day = step // 24
    hour = step % 24
    
    day_frac  = day / 30.0
    hour_frac = hour / 24.0

    # Economy
    my_money   = math.log1p(farm["money"]) / math.log1p(50000)
    opp_money  = math.log1p(opp_farm["money"]) / math.log1p(50000)
    my_quads   = len(farm.get("unlocked_quadrants", [])) / 4.0
    opp_quads  = len(opp_farm.get("unlocked_quadrants", [])) / 4.0
    my_workers = (1 + len(farm.get("hands", []))) / 16.0

    # Market prices normalized to base prices
    base_prices = {
        "WHEAT":25, "CARROT":35, "TOMATO":60, "STRAWBERRY":120, "MELON":250,
        "EGG":50, "MILK":160, "WOOL":200, "FERTILIZER":100
    }
    p = [prices.get(k, base_prices[k]) / base_prices[k] for k in base_prices.keys()]
    
    # Market inventory deviation from I0=10000
    inv_dev = [(inv.get(k, 10000) - 10000) / 10000.0 for k in base_prices.keys()]

    # Shed contents
    shed = priv.get("shed", {})
    shed_vec = [min(shed.get(k, 0), 100) / 100.0 for k in base_prices.keys()]

    # SpyTracker estimates (Opponent Shed)
    spy_shed = [min(spy_tracker.opp_shed.get(k, 0), 100) / 100.0 for k in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]]

    # Town shops (one-hot counts, capped at 8)
    shop_names = sorted(["BAKERY","PIZZA_SHOP","BRUNCH_SPOT","YARN_STORE","ICE_CREAM_SHOP","PET_CAFE","SMOOTHIE_SHOP","FARMERS_MARKET"])
    shops = obs.get("town", {}).get("unlocked_shops", [])
    shop_vec = [shops.count(s) / 4.0 for s in shop_names]

    obs_vec = [
        day_frac, hour_frac,
        my_money, opp_money,
        my_quads, opp_quads,
        my_workers,
    ] + p + inv_dev + shed_vec + spy_shed + shop_vec
    
    return np.array(obs_vec, dtype=np.float32)

def compute_reward(obs_prev, obs_curr, terminal, won):
    player = obs_curr["player"]
    opp = 1 - player
    
    prices = obs_curr["market"]["prices"]

    def net_worth(farm, shed, is_final):
        # Kaggle only counts pure cash at the end. 
        # By making shed goods worthless on the final step, we teach the agent
        # it MUST liquidate its assets before the match ends, completely preventing endgame hoarding.
        if is_final:
            return farm["money"]
        return farm["money"] + sum(shed.get(k, 0) * prices.get(k, 0) for k in shed)

    nw_me_now  = net_worth(obs_curr["farms"][player], obs_curr.get("private", {}).get("shed", {}), terminal)
    nw_me_prev = net_worth(obs_prev["farms"][player], obs_prev.get("private", {}).get("shed", {}), False)
    
    nw_opp_now = obs_curr["farms"][opp]["money"]
    nw_opp_prev = obs_prev["farms"][opp]["money"]

    delta_me  = nw_me_now  - nw_me_prev
    delta_opp = nw_opp_now - nw_opp_prev

    # Scale down dense reward so the terminal Win/Loss signal dominates
    dense = (delta_me - delta_opp) / 5000.0
    terminal_reward = (10.0 if won else -10.0) if terminal else 0.0

    return dense + terminal_reward

class KaggricEnv(gym.Env):
    def __init__(self, opponent_fn=None):
        super(KaggricEnv, self).__init__()
        
        if opponent_fn is None:
            # Default to playing against our own heuristic if no opponent provided
            self.opponent_agent = HeuristicAgent()
            self.opponent_fn = lambda obs: self.opponent_agent(obs)
        else:
            self.opponent_fn = opponent_fn
            
        # 51-dimensional observation vector
        self.observation_space = Box(low=-10.0, high=10.0, shape=(51,), dtype=np.float32)
        
        # Action space: [target_crop(6), sell_strategy(3), hire_target(4), land_strategy(2), fertilize_flag(2)]
        self.action_space = MultiDiscrete([6, 3, 4, 2, 2])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Load the environment. If local kaggriculture.py is copied, use it.
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggriculture.py")
        if os.path.exists(env_path):
            self.kaggle_env = make(env_path, configuration={"episodeSteps": 722})
        else:
            self.kaggle_env = make("kaggriculture", configuration={"episodeSteps": 722})
            
        self.kaggle_env.reset()
        
        self.heuristic = HeuristicAgent()
        
        # Reset the opponent if it's a HeuristicAgent that maintains state
        if hasattr(self.opponent_fn, '__self__') and isinstance(self.opponent_fn.__self__, HeuristicAgent):
            self.opponent_fn.__self__.__init__()
            
        self.spy = SpyTracker(player_id=0)
        self.goals = [3, 1, 1, 1, 0] # default goals
        self.day_count = 0
        
        # Extract true obs from state 0
        self.last_obs = dict(self.kaggle_env.state[0].observation)
        
        return make_obs(self.last_obs, self.spy), {}

    def step(self, action):
        self.goals = action
        
        obs_prev = self.last_obs
        
        # Run 24 Kaggle turns (one full day)
        done = False
        won = False
        
        for _ in range(24):
            if self.kaggle_env.done:
                done = True
                break
                
            obs_me = dict(self.kaggle_env.state[0].observation)
            obs_opp = dict(self.kaggle_env.state[1].observation)
            
            # Update our spy tracker
            self.spy.update(obs_me)
            
            # Get actions
            my_action = self.heuristic(obs_me, goals=self.goals)
            opp_action = self.opponent_fn(obs_opp)
            
            # Step the environment
            self.kaggle_env.step([my_action, opp_action])
            
        # Evaluate end of day
        obs_curr = dict(self.kaggle_env.state[0].observation)
        self.last_obs = obs_curr
        
        self.day_count += 1
        
        if self.kaggle_env.done:
            done = True
            
        if done:
            my_money = obs_curr["farms"][0]["money"]
            opp_money = obs_curr["farms"][1]["money"]
            won = my_money > opp_money
            
        reward = compute_reward(obs_prev, obs_curr, done, won)
        
        return make_obs(obs_curr, self.spy), float(reward), done, False, {}
