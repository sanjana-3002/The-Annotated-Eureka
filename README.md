# Eureka MWE: Human-Level Reward Design via Coding LLMs

A minimal working example implementing the core concepts from the paper:
**"Eureka: Human-Level Reward Design via Coding Large Language Models"**

## What This Demonstrates

This MWE shows how LLMs can automatically design reward functions for reinforcement learning:

1. **Zero-shot Generation**: LLM generates reward function code from environment description
2. **Evaluation**: Train RL agents with each reward and measure performance  
3. **Reflection**: Provide performance feedback to the LLM
4. **Evolution**: LLM improves rewards based on feedback over multiple iterations

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup OpenAI API

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Run Eureka

```bash
python eureka_mwe.py
```

**Expected runtime**: ~20-30 minutes (3 iterations × 4 rewards × 5 min training)

## Configuration

Edit `eureka_mwe.py` to adjust parameters:

```python
history = eureka.run(
    n_iterations=3,    # Number of evolutionary cycles
    n_samples=4,       # Reward candidates per iteration  
    timesteps=20000    # RL training steps per reward
)
```

**For faster testing**:
```python
history = eureka.run(n_iterations=2, n_samples=2, timesteps=10000)
```

## Output

Results are saved to `experiments/results/`:

- `iter{N}_reward{M}.py` - Generated reward function code
- `summary_{timestamp}.json` - Performance metrics
- `figures/reward_evolution.png` - Visualization of reward improvement

## Environment

**MountainCar-v0**: A classic RL problem demonstrating reward engineering challenges

- **Goal**: Drive car to flag at position >= 0.5
- **Challenge**: Car not powerful enough to drive straight up
- **Solution**: Build momentum by going back and forth
- **Why this environment**: Sparse default reward makes it ideal for showing Eureka's value

## Architecture

```
Input: Environment Code + Task Description
   ↓
[LLM] Generate 4 reward function candidates
   ↓
[RL] Train PPO agent with each reward (parallel)
   ↓
[Evaluate] Rank rewards by performance
   ↓
[Reflect] Create feedback summary
   ↓
[Iterate] LLM improves rewards
   ↓
Output: Best reward function + Trained policy
```

## Key Simplifications from Original Eureka

| Component | Original | MWE | Reason |
|-----------|----------|-----|--------|
| LLM | GPT-4 | GPT-3.5-turbo | Cost-effective |
| Environment | Isaac Gym (GPU) | Gymnasium (CPU) | Accessibility |
| Iterations | 5 | 3 | Time constraint |
| Samples/iter | 16 | 4 | Faster evaluation |
| Parallel envs | 4096 | 1 | CPU-friendly |
| Training steps | 100k+ | 20k | Reasonable convergence |

## Example Output

```
ITERATION 1/3
==================
Generating 4 reward functions...
Training RL agents...
  Evaluating reward 1/4...
  Evaluating reward 2/4...
  Evaluating reward 3/4...
  Evaluating reward 4/4...

SUMMARY:
Rank 1: Reward 3 - Mean: -95.2
Rank 2: Reward 1 - Mean: -108.5
Rank 3: Reward 2 - Mean: -125.3
Rank 4: Reward 4 - Mean: -142.7
```

## Cost Estimate

OpenAI API costs (GPT-3.5-turbo):
- ~$0.10-0.20 per complete run (3 iterations × 4 samples)
- Total tokens: ~15,000-20,000

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'gymnasium'`
**Solution**: Run `pip install -r requirements.txt`

**Issue**: `OpenAI API key required`
**Solution**: Set environment variable `export OPENAI_API_KEY="your-key"`

**Issue**: Training taking too long
**Solution**: Reduce `timesteps` parameter to 10000 or 15000

**Issue**: Poor reward performance
**Solution**: This is expected! Eureka improves over iterations. Wait for iteration 2-3.

## Code Structure

```
eureka-mwe/
├── eureka_mwe.py          # Main implementation (all-in-one)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── experiments/
│   └── results/           # Generated rewards and metrics
└── figures/
    └── reward_evolution.png  # Visualization
```

## For the Course Project

This MWE serves as Section 4 of your annotated paper. To complete the project:

1. **Background (Section 1)**: Explain reward engineering problem
2. **Architecture (Section 2)**: Diagram the complete pipeline with annotations
3. **Training (Section 3)**: Explain the evolutionary loop in detail
4. **MWE (Section 4)**: Run this code, analyze results, discuss insights
5. **Discussion (Section 5)**: Weaknesses, limitations, future directions
6. **Contributions (Section 6)**: Detail each member's work

## Key Insights to Discuss

1. **Why does this work?** LLMs understand environment dynamics from code
2. **Reward evolution**: How do rewards improve across iterations?
3. **Success cases**: What makes a good reward function?
4. **Failure modes**: When does Eureka struggle?
5. **Comparison**: How do Eureka rewards differ from human-designed ones?

## References

- Paper: https://arxiv.org/abs/2310.12931
- Project page: https://eureka-research.github.io/
- Original code: https://github.com/eureka-research/Eureka

## License

MIT License - For educational purposes (CS 577 Course Project)
