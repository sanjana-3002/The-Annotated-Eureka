"""
Main Eureka Algorithm Loop
"""
import os
import json
import time
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

import config
from reward_generator import RewardGenerator
from rl_trainer import RLTrainer, create_baseline_rewards
from utils import create_output_dirs, plot_results, save_results


class EurekaLoop:
    """Main Eureka algorithm implementation"""
    
    def __init__(self, 
                 num_iterations: int = config.NUM_ITERATIONS,
                 num_samples: int = config.NUM_SAMPLES_PER_ITERATION,
                 output_dir: str = config.OUTPUT_DIR):
        
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.output_dir = output_dir
        
        # Initialize components
        self.reward_generator = RewardGenerator()
        self.rl_trainer = RLTrainer()
        
        # Storage for results
        self.all_results = []
        self.best_reward_per_iteration = []
        
        # Create output directories
        create_output_dirs(output_dir)
        
        print(f"\n{'='*70}")
        print(f"EUREKA ALGORITHM - Minimal Working Example")
        print(f"{'='*70}")
        print(f"Environment: {config.ENV_NAME}")
        print(f"Iterations: {num_iterations}")
        print(f"Samples per iteration: {num_samples}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")
    
    def run_baseline_comparison(self) -> Dict[str, Any]:
        """Run baseline reward functions for comparison"""
        
        print("\n" + "="*70)
        print("RUNNING BASELINE COMPARISONS")
        print("="*70 + "\n")
        
        baselines = create_baseline_rewards()
        baseline_results = {}
        
        for name, reward_code in baselines.items():
            print(f"\nEvaluating {name} baseline...")
            try:
                metrics = self.rl_trainer.evaluate_reward(reward_code)
                baseline_results[name] = {
                    'code': reward_code,
                    'metrics': metrics,
                    'performance': metrics['mean_reward']
                }
                print(f"✓ {name.capitalize()} baseline: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            except Exception as e:
                print(f"✗ Error in {name} baseline: {e}")
                baseline_results[name] = {
                    'code': reward_code,
                    'metrics': {'mean_reward': 0, 'std_reward': 0},
                    'performance': 0,
                    'error': str(e)
                }
        
        return baseline_results
    
    def run_iteration(self, iteration: int, previous_rewards: List[Dict] = None) -> List[Dict]:
        """Run a single Eureka iteration"""
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{self.num_iterations}")
        print(f"{'='*70}\n")
        
        # Generate reward candidates
        reward_codes = self.reward_generator.generate_rewards(
            num_samples=self.num_samples,
            previous_rewards=previous_rewards,
            iteration=iteration
        )
        
        # Evaluate each reward candidate
        iteration_results = []
        
        for sample_id, reward_code in enumerate(reward_codes):
            print(f"\n--- Evaluating Reward Candidate {sample_id + 1}/{self.num_samples} ---")
            
            # Save reward code
            reward_path = self.reward_generator.save_reward(
                reward_code, 
                iteration, 
                sample_id, 
                config.REWARD_DIR
            )
            
            try:
                # Train and evaluate
                start_time = time.time()
                model, metrics = self.rl_trainer.train_with_reward(
                    reward_code,
                    total_timesteps=config.TOTAL_TIMESTEPS,
                    verbose=0
                )
                training_time = time.time() - start_time
                
                # Store results
                result = {
                    'iteration': iteration,
                    'sample_id': sample_id,
                    'code': reward_code,
                    'performance': metrics['mean_reward'],
                    'std': metrics['std_reward'],
                    'training_time': training_time,
                    'metrics': metrics,
                    'reward_path': reward_path
                }
                
                iteration_results.append(result)
                
                print(f"✓ Performance: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                print(f"  Training time: {training_time:.1f}s")
                
            except Exception as e:
                print(f"✗ Error evaluating reward: {e}")
                result = {
                    'iteration': iteration,
                    'sample_id': sample_id,
                    'code': reward_code,
                    'performance': 0.0,
                    'std': 0.0,
                    'training_time': 0.0,
                    'error': str(e),
                    'reward_path': reward_path
                }
                iteration_results.append(result)
        
        # Sort by performance
        iteration_results.sort(key=lambda x: x['performance'], reverse=True)
        
        # Print iteration summary
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1} SUMMARY")
        print(f"{'='*70}")
        for i, result in enumerate(iteration_results):
            print(f"{i+1}. Reward {result['sample_id']}: {result['performance']:.2f} ± {result.get('std', 0):.2f}")
        print(f"Best: {iteration_results[0]['performance']:.2f}")
        print(f"{'='*70}\n")
        from utils import log_iteration_details
        log_iteration_details(iteration, iteration_results, self.output_dir)
        return iteration_results
    
    def run(self) -> Dict[str, Any]:
        """Run complete Eureka algorithm"""
        
        start_time = time.time()
        
        # Run baselines first
        baseline_results = self.run_baseline_comparison()
        
        # Run Eureka iterations
        previous_rewards = None
        
        for iteration in range(self.num_iterations):
            # Run iteration
            iteration_results = self.run_iteration(iteration, previous_rewards)
            
            # Store results
            self.all_results.extend(iteration_results)
            self.best_reward_per_iteration.append(iteration_results[0])
            
            # Prepare for next iteration (use top performers as context)
            previous_rewards = iteration_results[:2]  # Top 2 rewards
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'baselines': baseline_results,
            'all_iterations': self.all_results,
            'best_per_iteration': self.best_reward_per_iteration,
            'total_time': total_time,
            'config': {
                'num_iterations': self.num_iterations,
                'num_samples': self.num_samples,
                'total_timesteps': config.TOTAL_TIMESTEPS,
                'env_name': config.ENV_NAME
            }
        }
        
        # Print final summary
        self.print_final_summary(final_results)
        
        # Save results
        self.save_final_results(final_results)
        
        # Generate plots
        plot_results(final_results, self.output_dir)
        
        # Create summary table
        from utils import create_summary_table
        create_summary_table(final_results, self.output_dir)
        return final_results
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print final summary of Eureka run"""
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70 + "\n")
        
        # Baseline results
        print("Baseline Comparisons:")
        for name, data in results['baselines'].items():
            print(f"  {name.capitalize()}: {data['performance']:.2f}")
        
        # Best result per iteration
        print("\nEureka Progression:")
        for i, best in enumerate(results['best_per_iteration']):
            print(f"  Iteration {i+1}: {best['performance']:.2f} ± {best.get('std', 0):.2f}")
        
        # Overall best
        overall_best = max(results['best_per_iteration'], key=lambda x: x['performance'])
        print(f"\nOverall Best Reward:")
        print(f"  Performance: {overall_best['performance']:.2f} ± {overall_best.get('std', 0):.2f}")
        print(f"  From: Iteration {overall_best['iteration'] + 1}, Sample {overall_best['sample_id']}")
        
        # Comparison to baselines
        best_baseline = max(results['baselines'].values(), key=lambda x: x['performance'])
        improvement = ((overall_best['performance'] - best_baseline['performance']) / 
                      best_baseline['performance'] * 100)
        print(f"\nImprovement over best baseline: {improvement:+.1f}%")
        
        print(f"\nTotal runtime: {results['total_time']/60:.1f} minutes")
        print("="*70 + "\n")
    
    def save_final_results(self, results: Dict[str, Any]):
        """Save final results to JSON"""
        
        output_file = os.path.join(self.output_dir, 'eureka_results.json')
        
        # Prepare serializable results
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'config': results['config'],
            'baselines': {
                name: {
                    'performance': data['performance'],
                    'metrics': data['metrics']
                }
                for name, data in results['baselines'].items()
            },
            'best_per_iteration': [
                {
                    'iteration': r['iteration'],
                    'performance': r['performance'],
                    'std': r.get('std', 0)
                }
                for r in results['best_per_iteration']
            ],
            'overall_best': {
                'iteration': results['best_per_iteration'][-1]['iteration'],
                'performance': results['best_per_iteration'][-1]['performance'],
                'code': results['best_per_iteration'][-1]['code']
            },
            'total_time': results['total_time']
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✓ Results saved to: {output_file}")


def main():
    """Main entry point"""
    
    # Create and run Eureka
    eureka = EurekaLoop(
        num_iterations=config.NUM_ITERATIONS,
        num_samples=config.NUM_SAMPLES_PER_ITERATION,
        output_dir=config.OUTPUT_DIR
    )
    
    results = eureka.run()
    
    return results


if __name__ == "__main__":
    main()
