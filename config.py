"""
Configuration file for Eureka MWE
"""
import os

# ============================================================================
# OpenAI API Configuration
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-3.5-turbo-16k"  # Use 16k context for environment code
LLM_TEMPERATURE = 1.0  # Higher temperature for diverse reward generation
LLM_MAX_TOKENS = 2048

# ============================================================================
# Eureka Algorithm Parameters
# ============================================================================
NUM_ITERATIONS = 3  # Number of Eureka iterations
NUM_SAMPLES_PER_ITERATION = 4  # Number of reward candidates per iteration

# ============================================================================
# RL Training Parameters
# ============================================================================
ENV_NAME = "CartPole-v1"
TOTAL_TIMESTEPS = 50000  # Training steps per reward candidate
N_ENVS = 4  # Number of parallel environments
EVAL_EPISODES = 10  # Episodes for evaluation

# RL Algorithm (PPO) hyperparameters
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2

# ============================================================================
# Logging and Output
# ============================================================================
OUTPUT_DIR = "./results"
REWARD_DIR = "./rewards"
VERBOSE = True
SAVE_BEST_ONLY = True

# Tensorboard logging
USE_TENSORBOARD = True
TENSORBOARD_LOG = "./tensorboard_logs"

# ============================================================================
# Evaluation Metrics
# ============================================================================
# For CartPole: max reward is 500 (survives 500 steps)
BASELINE_HUMAN_REWARD = 200  # Typical human-designed reward performance
BASELINE_SPARSE_REWARD = 150  # Simple sparse reward performance

# ============================================================================
# Reward Function Template
# ============================================================================
REWARD_FUNCTION_TEMPLATE = """
def compute_reward(obs, action, next_obs, done, info):
    '''
    Custom reward function for CartPole-v1
    
    Args:
        obs: Current observation [cart_pos, cart_vel, pole_angle, pole_vel]
        action: Action taken (0=left, 1=right)
        next_obs: Next observation after action
        done: Whether episode is done
        info: Additional info dict
    
    Returns:
        reward: float
    '''
    # Your reward logic here
    reward = 0.0
    
    return reward
"""

# ============================================================================
# Environment Context (for LLM)
# ============================================================================
ENVIRONMENT_DESCRIPTION = """
CartPole-v1 Environment:
- Goal: Balance a pole on a cart by moving the cart left or right
- Observation Space: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
  - cart_position: -4.8 to 4.8
  - cart_velocity: -Inf to Inf
  - pole_angle: -0.418 rad to 0.418 rad (-24° to 24°)
  - pole_angular_velocity: -Inf to Inf
- Action Space: Discrete(2)
  - 0: Push cart to the left
  - 1: Push cart to the right
- Episode Termination:
  - Pole angle > ±12 degrees
  - Cart position > ±2.4
  - Episode length > 500 steps
- Success Criteria: 
  - Survive 500 steps (max reward)
  - Average reward > 475 over 100 episodes is considered solved
"""
