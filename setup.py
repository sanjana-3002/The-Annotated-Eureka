#!/usr/bin/env python3
"""
Quick Setup and Test Script for Eureka MWE
"""
import os
import sys
import subprocess


def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def install_dependencies():
    """Install required packages"""
    print("\n" + "="*60)
    print("Installing dependencies...")
    print("="*60 + "\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        return False


def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: OPENAI_API_KEY not found in environment")
        print("   To set it, run:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False
    print("✓ OpenAI API key found")
    return True


def test_imports():
    """Test that all imports work"""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60 + "\n")
    
    try:
        import gymnasium
        print("✓ gymnasium")
        
        import stable_baselines3
        print("✓ stable_baselines3")
        
        import torch
        print("✓ torch")
        
        import openai
        print("✓ openai")
        
        import matplotlib
        print("✓ matplotlib")
        
        print("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


def test_environment():
    """Test CartPole environment"""
    print("\n" + "="*60)
    print("Testing CartPole environment...")
    print("="*60 + "\n")
    
    try:
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        obs, info = env.reset()
        
        print(f"✓ Environment created")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print("✓ Environment test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def run_quick_test():
    """Run a quick test of the reward generator"""
    print("\n" + "="*60)
    print("Running quick component test...")
    print("="*60 + "\n")
    
    try:
        # Test reward generator (without API call)
        print("Testing reward generator structure...")
        from reward_generator import RewardGenerator
        print("✓ RewardGenerator imported")
        
        # Test RL trainer
        print("Testing RL trainer structure...")
        from rl_trainer import RLTrainer, create_baseline_rewards
        print("✓ RLTrainer imported")
        
        baselines = create_baseline_rewards()
        print(f"✓ Created {len(baselines)} baseline rewards")
        
        print("\n✓ Component test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("EUREKA MWE - SETUP & TEST")
    print("="*60 + "\n")
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("OpenAI API Key", check_openai_key),
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  Some checks failed. Please fix issues before continuing.")
        return
    
    # Ask to install dependencies
    print("\n" + "="*60)
    response = input("Install dependencies? (y/n): ").strip().lower()
    if response == 'y':
        if not install_dependencies():
            return
        
        # Test imports after installation
        if not test_imports():
            return
    
    # Test environment
    if not test_environment():
        return
    
    # Run component test
    if not run_quick_test():
        return
    
    # Success!
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60)
    print("\nYou're ready to run Eureka!")
    print("\nTo start the main algorithm:")
    print("  python eureka_loop.py")
    print("\nTo test individual components:")
    print("  python reward_generator.py")
    print("  python rl_trainer.py")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
