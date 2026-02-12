"""
LLM-based Reward Function Generator for Eureka MWE
"""
import os
import re
from typing import List, Dict, Optional
from openai import OpenAI
import config

class RewardGenerator:
    """Generate reward functions using LLM (GPT-3.5/GPT-4)"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the reward generator with OpenAI API"""
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        
    def create_initial_prompt(self) -> str:
        """Create the initial prompt for reward generation"""
        
        prompt = f"""You are an expert in reinforcement learning reward design. Your task is to generate Python reward functions for a CartPole environment.

{config.ENVIRONMENT_DESCRIPTION}

Your task: Generate a reward function that will help an RL agent learn to balance the pole for as long as possible.

Requirements:
1. Write a complete Python function called 'compute_reward'
2. The function signature must be: compute_reward(obs, action, next_obs, done, info)
3. Use numpy for any mathematical operations (imported as np)
4. The reward should be DENSE (provide feedback at every step, not just at success/failure)
5. Consider multiple aspects: pole angle, angular velocity, cart position, etc.
6. Be creative! Try novel reward formulations.
7. Include brief comments explaining your reward design choices

Example structure:
```python
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    # Extract state variables
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    
    # Your reward logic here
    # Consider: pole angle penalty, velocity penalties, position penalties, etc.
    
    reward = # your formula
    
    return reward
```

Generate a novel, effective reward function now. Only output the Python code, nothing else."""
        
        return prompt
    
    def create_reflection_prompt(self, previous_rewards: List[Dict], iteration: int) -> str:
        """Create prompt with reflection on previous rewards"""
        
        # Sort by performance
        sorted_rewards = sorted(previous_rewards, key=lambda x: x['performance'], reverse=True)
        
        # Create performance summary
        performance_summary = "Previous reward function performances:\n\n"
        for i, reward_info in enumerate(sorted_rewards):
            performance_summary += f"Reward {i+1} (Score: {reward_info['performance']:.2f}):\n"
            performance_summary += f"```python\n{reward_info['code']}\n```\n"
            performance_summary += f"Analysis: {reward_info.get('analysis', 'N/A')}\n\n"
        
        prompt = f"""You are an expert in reinforcement learning reward design. This is iteration {iteration} of reward function evolution for CartPole.

{config.ENVIRONMENT_DESCRIPTION}

{performance_summary}

Task: Based on the performance of previous reward functions, generate an IMPROVED reward function.

Guidelines:
1. Analyze what worked well in the best-performing rewards
2. Identify potential issues in lower-performing rewards
3. Combine successful elements or try novel approaches
4. Consider:
   - Are penalties too harsh or too lenient?
   - Is the reward signal clear enough?
   - Are we rewarding the right behaviors?
   - Could different scaling factors help?

Generate an improved reward function. Only output the Python code, nothing else."""
        
        return prompt
    
    def generate_rewards(self, 
                        num_samples: int, 
                        previous_rewards: Optional[List[Dict]] = None,
                        iteration: int = 0) -> List[str]:
        """Generate multiple reward function candidates"""
        
        print(f"\n{'='*60}")
        print(f"Generating {num_samples} reward candidates for iteration {iteration}...")
        print(f"{'='*60}\n")
        
        # Create appropriate prompt
        if previous_rewards is None or len(previous_rewards) == 0:
            prompt = self.create_initial_prompt()
        else:
            prompt = self.create_reflection_prompt(previous_rewards, iteration)
        
        # Generate multiple reward candidates
        reward_codes = []
        for i in range(num_samples):
            print(f"Generating reward candidate {i+1}/{num_samples}...")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert reward function designer for reinforcement learning."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=config.LLM_MAX_TOKENS
                )
                
                # Extract code from response
                reward_code = self._extract_code(response.choices[0].message.content)
                reward_codes.append(reward_code)
                
                print(f"✓ Generated reward candidate {i+1}")
                
            except Exception as e:
                print(f"✗ Error generating reward {i+1}: {e}")
                # Fallback to simple reward
                reward_codes.append(self._get_fallback_reward())
        
        return reward_codes
    
    def _extract_code(self, response_text: str) -> str:
        """Extract Python code from LLM response"""
        
        # Try to extract code between ```python and ```
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # Try to extract code between ``` and ```
        pattern = r"```\n(.*?)\n```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # If no code blocks found, assume entire response is code
        return response_text
    
    def _get_fallback_reward(self) -> str:
        """Return a simple fallback reward if LLM fails"""
        return """import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    # Simple fallback reward
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    
    # Penalize large pole angles
    angle_penalty = -abs(pole_angle)
    
    # Small reward for surviving
    survival_reward = 1.0
    
    reward = survival_reward + angle_penalty
    
    return reward
"""

    def save_reward(self, reward_code: str, iteration: int, sample_id: int, output_dir: str):
        """Save generated reward code to file"""
        import os
        
        # Create directory structure
        iter_dir = os.path.join(output_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Save reward code
        filepath = os.path.join(iter_dir, f"reward_{sample_id}.py")
        with open(filepath, 'w') as f:
            f.write(reward_code)
        
        return filepath


# Example usage and testing
if __name__ == "__main__":
    # Test reward generator
    generator = RewardGenerator()
    
    # Generate initial rewards
    rewards = generator.generate_rewards(num_samples=2, iteration=0)
    
    for i, reward in enumerate(rewards):
        print(f"\n--- Reward {i} ---")
        print(reward)
        
        # Save reward
        generator.save_reward(reward, iteration=0, sample_id=i, output_dir=config.REWARD_DIR)
    
    print("\n✓ Reward generation test complete!")
