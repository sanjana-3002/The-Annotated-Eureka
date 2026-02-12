"""
RL Training Module with Custom Reward Functions
"""
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable, Dict, Any, Tuple
import config


class CustomRewardWrapper(Wrapper):
    """Wrapper to apply custom reward function to environment"""
    
    def __init__(self, env, reward_fn: Callable):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.last_obs = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply custom reward function
        try:
            custom_reward = self.reward_fn(
                obs=self.last_obs,
                action=action,
                next_obs=obs,
                done=terminated or truncated,
                info=info
            )
            # Use custom reward instead of environment reward
            reward = custom_reward
        except Exception as e:
            print(f"Error in custom reward function: {e}")
            # Fall back to original reward
            pass
        
        self.last_obs = obs
        done = terminated or truncated
        
        return obs, reward, terminated, truncated, info


class RLTrainer:
    """Train RL agents with custom reward functions"""
    
    def __init__(self, env_name: str = config.ENV_NAME):
        self.env_name = env_name
        
    def load_reward_function(self, reward_code: str) -> Callable:
        """Load reward function from code string"""
        
        # Create a namespace for execution
        namespace = {'np': np, 'numpy': np}
        
        # Execute the code to define the function
        try:
            exec(reward_code, namespace)
            
            # Get the compute_reward function
            if 'compute_reward' not in namespace:
                raise ValueError("Reward code must define 'compute_reward' function")
            
            return namespace['compute_reward']
            
        except Exception as e:
            print(f"Error loading reward function: {e}")
            raise
    
    def create_env_with_reward(self, reward_fn: Callable):
        """Create environment with custom reward function"""
        
        def make_env():
            env = gym.make(self.env_name)
            env = CustomRewardWrapper(env, reward_fn)
            return env
        
        return make_env
    
    def train_with_reward(self, 
                         reward_code: str, 
                         total_timesteps: int = config.TOTAL_TIMESTEPS,
                         verbose: int = 0) -> Tuple[PPO, Dict[str, Any]]:
        """Train a PPO agent with custom reward function"""
        
        print(f"Training agent for {total_timesteps} timesteps...")
        
        try:
            # Load reward function
            reward_fn = self.load_reward_function(reward_code)
            
            # Create vectorized environment
            env = DummyVecEnv([self.create_env_with_reward(reward_fn) 
                              for _ in range(config.N_ENVS)])
            
            # Create PPO agent
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config.PPO_LEARNING_RATE,
                n_steps=config.PPO_N_STEPS,
                batch_size=config.PPO_BATCH_SIZE,
                n_epochs=config.PPO_N_EPOCHS,
                gamma=config.PPO_GAMMA,
                gae_lambda=config.PPO_GAE_LAMBDA,
                clip_range=config.PPO_CLIP_RANGE,
                verbose=verbose,
                tensorboard_log=config.TENSORBOARD_LOG if config.USE_TENSORBOARD else None
            )
            
            # Train the model
            model.learn(total_timesteps=total_timesteps)
            
            # Evaluate the trained model
            eval_env = gym.make(self.env_name)
            eval_env = CustomRewardWrapper(eval_env, reward_fn)
            
            mean_reward, std_reward = evaluate_policy(
                model, 
                eval_env, 
                n_eval_episodes=config.EVAL_EPISODES,
                deterministic=True
            )
            
            eval_env.close()
            env.close()
            
            # Return model and metrics
            metrics = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'timesteps': total_timesteps
            }
            
            print(f"✓ Training complete! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            return model, metrics
            
        except Exception as e:
            print(f"✗ Error during training: {e}")
            raise
    
    def evaluate_reward(self, reward_code: str) -> Dict[str, float]:
        """Quick evaluation of a reward function"""
        
        try:
            model, metrics = self.train_with_reward(reward_code)
            return metrics
        except Exception as e:
            print(f"Error evaluating reward: {e}")
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'timesteps': 0,
                'error': str(e)
            }


def create_baseline_rewards() -> Dict[str, str]:
    """Create baseline reward functions for comparison"""
    
    baselines = {}
    
    # Sparse reward (environment default)
    baselines['sparse'] = """import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    # Use environment's default sparse reward (1 per timestep)
    return 1.0
"""
    
    # Human-designed dense reward
    baselines['human'] = """import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    
    # Reward staying upright
    angle_reward = 1.0 - abs(pole_angle) / 0.418  # Normalize to [0, 1]
    
    # Penalize large velocities
    velocity_penalty = -0.01 * abs(pole_vel)
    
    # Penalize cart moving too far from center
    position_penalty = -0.01 * abs(cart_pos)
    
    # Combine rewards
    reward = angle_reward + velocity_penalty + position_penalty
    
    # Bonus for surviving
    reward += 0.1
    
    return reward
"""
    
    return baselines


# Example usage
if __name__ == "__main__":
    trainer = RLTrainer()
    
    # Test with sparse reward
    baselines = create_baseline_rewards()
    
    print("Testing RL trainer with sparse reward...")
    metrics = trainer.evaluate_reward(baselines['sparse'])
    print(f"Sparse reward result: {metrics}")
    
    print("\nTesting RL trainer with human-designed reward...")
    metrics = trainer.evaluate_reward(baselines['human'])
    print(f"Human reward result: {metrics}")
    
    print("\n✓ RL trainer test complete!")
