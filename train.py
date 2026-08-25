import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_env import KaggricEnv
from heuristic_agent import HeuristicAgent

def make_env():
    # Helper to instantiate our environment for vectorized training
    def _init():
        # Play against the heuristic agent for Phase 1
        heuristic_opponent = HeuristicAgent()
        return KaggricEnv(opponent_fn=lambda obs: heuristic_opponent(obs))
    return _init

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    
    print("Setting up vectorized environments...")
    # Use SubprocVecEnv for multiprocessing, or DummyVecEnv for debugging
    num_envs = 8 
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        envs,
        n_steps=120,        # 120 days = 4 full episodes per rollout buffer
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # Encourage exploration
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Save a checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // num_envs, 1),
        save_path='./models/',
        name_prefix='ppo_kaggric'
    )
    
    print("Starting training (Phase 1 vs Heuristic)...")
    # 300,000 steps = 10,000 episodes (since 1 episode = 30 steps)
    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
    
    print("Saving final model...")
    model.save("models/ppo_kaggric_final")
    
    print("Training complete!")
