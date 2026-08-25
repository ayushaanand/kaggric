import math
from collections import defaultdict

# Fixed shop prices for seeds and animals
SEED_COSTS = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "MELON": 80,
    "STRAWBERRY": 100
}

ANIMAL_COSTS = {
    "GOOSE": 300,
    "COW": 400,
    "SHEEP": 500
}

class SpyTracker:
    """Tracks opponent's hidden state by diffing public observations."""
    def __init__(self, player_id):
        self.player_id = player_id
        self.opp_id = 1 - player_id
        
        self.opp_shed = defaultdict(int)
        self.unresolved_spend = 0
        self.prev_obs = None
        
        self.opp_active_animals = defaultdict(int)

    def update(self, obs):
        if self.prev_obs is None:
            self.prev_obs = obs
            return
            
        prev_opp = self.prev_obs["farms"][self.opp_id]
        curr_opp = obs["farms"][self.opp_id]
        
        # 1. Calculate Money Drops (Potential hidden purchases)
        # Note: We simplify by ignoring exact market product matching for V1
        # and just treating all unexplained money drops as 'unresolved spend'
        money_spent = prev_opp["money"] - curr_opp["money"]
        
        # Deduct public hires (approximate cost tracking)
        if curr_opp["hires_today"] > prev_opp["hires_today"]:
            money_spent -= 1 # Base cost approximation
            
        # Deduct public land buys
        new_quads = len(curr_opp["unlocked_quadrants"]) - len(prev_opp["unlocked_quadrants"])
        if new_quads > 0:
            # Approx cost: 1k, 2k, 4k
            money_spent -= 1000 * new_quads 
            
        if money_spent > 0:
            self.unresolved_spend += money_spent
            
        # 2. Diff the Board (Harvests & Resolutions)
        board_size = len(curr_opp["tiles"])
        for y in range(board_size):
            for x in range(board_size):
                p_tile = prev_opp["tiles"][y][x]
                c_tile = curr_opp["tiles"][y][x]
                
                # Was it an empty tile that is now a plant? (Resolving Dark Matter)
                if p_tile is None and isinstance(c_tile, dict) and c_tile.get("kind") == "PLANT":
                    crop = c_tile.get("crop")
                    if crop in SEED_COSTS:
                        self.unresolved_spend = max(0, self.unresolved_spend - SEED_COSTS[crop])
                        
                # Was it a plant that got harvested? (Inflow to Shadow Inventory)
                if (isinstance(p_tile, dict) and isinstance(c_tile, dict) and 
                    p_tile.get("kind") == "PLANT" and c_tile.get("kind") == "PLANT"):
                    p_yield = p_tile.get("yield_units", 0)
                    c_yield = c_tile.get("yield_units", 0)
                    if p_yield > c_yield:
                        harvested = p_yield - c_yield
                        self.opp_shed[c_tile["crop"]] += harvested
                
                # Animal placement resolution
                if (isinstance(p_tile, dict) and isinstance(c_tile, dict) and 
                    p_tile.get("animal") is None and c_tile.get("animal") is not None):
                    animal = c_tile["animal"]
                    if animal in ANIMAL_COSTS:
                        self.unresolved_spend = max(0, self.unresolved_spend - ANIMAL_COSTS[animal])
                        self.opp_active_animals[animal] += 1
                        
                # Animal feeding (Outflow of Wheat)
                if (isinstance(p_tile, dict) and isinstance(c_tile, dict) and 
                    c_tile.get("animal") is not None):
                    if not p_tile.get("fed_today", False) and c_tile.get("fed_today", False):
                        self.opp_shed["WHEAT"] = max(0, self.opp_shed["WHEAT"] - 1)

        # 3. Handle Shed Overflow (Cap at 100 items)
        total_items = sum(self.opp_shed.values())
        if total_items > 100:
            scale = 100.0 / total_items
            for k in self.opp_shed:
                self.opp_shed[k] = int(self.opp_shed[k] * scale)
                
        self.prev_obs = obs


class HeuristicAgent:
    def __init__(self):
        self.tracker = None
        self.target_crop = "STRAWBERRY" # High ROI default

    def get_move_dir(self, fr, to):
        """Simple Manhattan directional movement."""
        dx = to[0] - fr[0]
        dy = to[1] - fr[1]
        
        if dx > 0: return "EAST"
        elif dx < 0: return "WEST"
        elif dy > 0: return "SOUTH"
        elif dy < 0: return "NORTH"
        return "PASS"

    def __call__(self, obs, goals=None):
        player_id = obs["player"]
        
        # Initialize tracker on turn 1
        if self.tracker is None:
            self.tracker = SpyTracker(player_id)
            
        self.tracker.update(obs)
        
        my_farm = obs["farms"][player_id]
        farmer_pos = my_farm["farmer"]
        my_money = my_farm["money"]
        private = obs.get("private", {})
        
        market_actions = []
        farmer_action = "PASS"
        
        # --- 1. MACRO: Market Logic & Expansion ---
        step = obs.get("step", 0)
        day = step // 24 # 0 to 29
        
        # Parse goals if provided (from RL wrapper)
        # goals = [target_crop, sell_strategy, hire_target, land_strategy, fertilize_flag]
        goal_crop, goal_sell, goal_hire, goal_land, goal_fert = None, None, None, None, None
        if goals is not None and len(goals) >= 4:
            goal_crop = goals[0]
            goal_sell = goals[1]
            goal_hire = goals[2]
            goal_land = goals[3]
            if len(goals) >= 5:
                goal_fert = goals[4]

        # 1a. Dynamic Crop Targeting
        prices = obs.get("town", {}).get("prices", {})
        if goal_crop is not None and goal_crop < 5:
            self.target_crop = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"][goal_crop]
        else:
            if prices.get("MELON", 0) > prices.get("STRAWBERRY", 0) + 20: 
                self.target_crop = "MELON"
            else:
                self.target_crop = "STRAWBERRY"
            
        # 1c. Paced Selling
        for item, amount in private.get("shed", {}).items():
            if amount > 0:
                sell_vol = 0
                if goal_sell == 1:
                    sell_vol = 2 # Trickle
                elif goal_sell == 2:
                    sell_vol = amount # Dump
                elif goal_sell == 0:
                    sell_vol = 0 # Hold
                else:
                    # Heuristic default
                    sell_vol = 1
                    if amount > 20: sell_vol = 2
                    if amount > 50: sell_vol = 4
                    if day >= 28: sell_vol = amount # THE ENDGAME SQUEEZE
                
                if sell_vol > 0:
                    market_actions.append(["SELL", item, min(sell_vol, amount)])
                
        # Seed buying is now handled after board scan

        # --- 2. MICRO: Task Dispatcher & Pathfinding ---
        
        # Collect all jobs
        needs_harvest = []
        needs_water = []
        needs_dig = []
        empty_tiles = []
        
        board_size = len(my_farm["tiles"])
        for y in range(board_size):
            for x in range(board_size):
                tile = my_farm["tiles"][y][x]
                if tile is None:
                    empty_tiles.append((x, y))
                elif isinstance(tile, dict):
                    if tile.get("kind") == "WEED":
                        needs_dig.append((x, y))
                    elif tile.get("kind") == "PLANT":
                        if tile.get("yield_units", 0) > 0:
                            needs_harvest.append((x, y))
                        if not tile.get("watered_today", False):
                            needs_water.append((x, y))
                            
        # Dynamic Seed Purchasing: Only buy what our current workforce can handle
        my_seeds = private.get("seeds", {})
        current_workers = 1 + len(my_farm.get("hands", []))
        workforce_capacity = current_workers * 2
        desired_seeds = min(len(empty_tiles), workforce_capacity) + 2
        
        # ENDGAME SQUEEZE: Stop buying seeds after Day 20 (they won't have time to grow!)
        if day >= 20: desired_seeds = 0
        
        current_seeds = my_seeds.get(self.target_crop, 0)
        seeds_needed = desired_seeds - current_seeds
        
        if seeds_needed > 0 and my_money >= SEED_COSTS.get(self.target_crop, 0):
            buy_amount = min(5, seeds_needed, int(my_money // SEED_COSTS.get(self.target_crop, 1)))
            if buy_amount > 0:
                market_actions.append(["BUY_SEED", self.target_crop, buy_amount])
                
        # 1b. Land Expansion: Strict financial buffer!
        unlocked_quads = len(my_farm.get("unlocked_quadrants", []))
        land_cost = unlocked_quads * 1000 # 1000, 2000, 3000
        
        do_expand = False
        if goal_land == 1 and my_money >= land_cost:
            do_expand = True
        elif goal_land is None:
            # Heuristic default
            if day < 15 and unlocked_quads < 4 and len(empty_tiles) < 5 and my_money > land_cost + 2000:
                do_expand = True
                
        if do_expand and unlocked_quads < 4:
            market_actions.append(["BUY_LAND"])
            
        # Auto-Hiring
        total_jobs = len(needs_harvest) + len(needs_water) + len(needs_dig) + seeds_needed
        
        desired_workers = max(1, total_jobs // 6)
        if goal_hire is not None:
            if goal_hire == 0: desired_workers = 1
            elif goal_hire == 1: desired_workers = 4
            elif goal_hire == 2: desired_workers = 8
            elif goal_hire == 3: desired_workers = 12
        
        num_hands = len(my_farm.get("hands", []))
        # Calculate Fibonacci cost of the next hire (1, 1, 2, 3, 5, 8, 13...)
        a, b = 1, 1
        for _ in range(num_hands):
            a, b = b, a + b
        next_hire_cost = a
        
        # Only hire if we actually need them, we have 5x the cost in the bank, and the next hire is < $30
        if (num_hands + 1) < desired_workers and my_money > (next_hire_cost * 5) and next_hire_cost <= 34:
            market_actions.append(["HIRE"])

        # Prioritize jobs into Tiers
        # Sort empty tiles by distance to the Shed so they default to nearby, 
        # but we won't slice the array so workers can still plant on the tile they are standing on!
        shed_x, shed_y = (board_size // 2) - 1, (board_size // 2) - 1
        empty_tiles.sort(key=lambda p: abs(p[0] - shed_x) + abs(p[1] - shed_y))
        
        seeds_available = my_seeds.get(self.target_crop, 0)
        
        # Merge DIG and PLANT into the same tier! 
        expansion_jobs = [(p, "DIG") for p in needs_dig] + [(p, ["PLANT", self.target_crop]) for p in empty_tiles]
        
        job_tiers = [
            [(p, "WATER") for p in needs_water], # WATER first so we don't accidentally harvest unwatered crops (which might be possible in some engines)
            [(p, "HARVEST") for p in needs_harvest],
            expansion_jobs
        ]

        claimed_targets = set()
        # Track how many seeds we have committed to planting this turn
        seeds_committed = 0
        
        def assign_action(worker_pos):
            nonlocal seeds_committed
            
            # Greedy Pathfinding: For each priority tier, find the closest available job
            for tier in job_tiers:
                best_job = None
                best_dist = 999
                for pos, act in tier:
                    if pos not in claimed_targets:
                        # If this is a planting job, ensure we have seeds available!
                        if isinstance(act, list) and act[0] == "PLANT":
                            if seeds_committed >= seeds_available:
                                continue
                                
                        dist = abs(worker_pos[0] - pos[0]) + abs(worker_pos[1] - pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_job = (pos, act)
                
                # If we found a job in this tier, take it!
                if best_job:
                    pos, act = best_job
                    claimed_targets.add(pos)
                    
                    if isinstance(act, list) and act[0] == "PLANT":
                        seeds_committed += 1
                        
                    if worker_pos[0] == pos[0] and worker_pos[1] == pos[1]:
                        return act
                    else:
                        return self.get_move_dir(worker_pos, pos)
            
            return "PASS"

        farmer_action = assign_action(farmer_pos)
        
        hands_actions = []
        for hand_pos in my_farm.get("hands", []):
            act = assign_action(hand_pos)
            hands_actions.append([act] if isinstance(act, str) else act)

        # Write debug logs to stdout so Kaggle can capture them in the replay logs
        step = obs.get("step", 0)
        day = step // 24
        
        if step == 0:
            print("=== NEW EPISODE START ===")
        
        # Log what we are doing (Kaggle captures print statements into the UI)
        print(f"Turn {step} (Day {day}) | F_Pos: {farmer_pos} | F_Act: {farmer_action} | Hands: {hands_actions} | Market: {market_actions}")
        
        # Log what we think the opponent is doing
        if step > 0 and step % 24 == 0:
            print(f"  [SPYTRACKER] Opponent Shadow Inventory: {dict(self.tracker.opp_shed)}")
            print(f"  [SPYTRACKER] Unresolved Spend (Hidden Assets): ${self.tracker.unresolved_spend}")
            print(f"  [SPYTRACKER] Active Animals: {dict(self.tracker.opp_active_animals)}")

        return {
            "farmer": [farmer_action] if isinstance(farmer_action, str) else farmer_action,
            "hands": hands_actions,
            "market": market_actions
        }


