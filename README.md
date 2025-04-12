# Blackjack Gymnasium

A reinforcement learning project that implements various algorithms to solve the Blackjack environment from OpenAI's Gymnasium.

## Overview

This project explores different reinforcement learning approaches to train agents that can play Blackjack optimally. The goal is to maximize expected rewards by making strategic decisions (hit or stick) based on the current game state.

## Implemented Algorithms

### Deep Q-Network (DQN)

- Neural network function approximation
- Experience replay buffer for stable learning
- Target network to reduce overestimation bias
- Policy and value function visualization

### Q-Learning

- Tabular state-action value function
- Epsilon-greedy exploration
- Temporal difference learning

### SARSA (Planned)

- On-policy learning algorithm
- State-Action-Reward-State-Action update rule

## Features

- Comprehensive training and testing pipelines
- Policy and value function visualization
- Human-readable gameplay rendering
- Configurable hyperparameters
- Pre-trained model support

## Requirements

- Python 3.12+
- Dependencies from `requirements.txt`:
  - PyTorch
  - Gymnasium
  - NumPy
  - Matplotlib
  - Seaborn
  - Pandas

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/blackjack-gymnasium.git
cd blackjack-gymnasium

# Install dependencies with uv
uv pip install -r requirements.txt
```

## Usage

### DQN Agent

```bash
python src/dqn.py
```

The script will train a new DQN agent for 100,000 episodes by default. The model will be saved to `./models/dqn_blackjack.pth`.

To use a pre-trained model, set `use_pre_trained_model = True` in `src/dqn.py` before running.

### Q-Learning Agent

```bash
uv run src/q_learning.py
```

## The Blackjack Environment

In Blackjack, the goal is to obtain cards that sum to as close as possible to 21 without exceeding it. The agent can choose to:

- Hit (1): Take another card
- Stick (0): Stop taking cards

Each game state consists of:

- Player's current sum (between 12-21)
- Dealer's face-up card (1-10)
- Whether the player has a usable ace (True/False)

## Visualization

The project generates visualizations of:

1. **State Value Function**: A 3D surface plot showing the expected return from each state
2. **Policy Function**: A heatmap showing the optimal action (hit or stick) for each state

Both visualizations are created for two scenarios:

- With a usable ace (counted as 11)
- Without a usable ace (counted as 1)

## Project Structure

```
blackjack-gymnasium/
├── models/            # Saved model weights
├── src/               # Source code
│   ├── dqn.py         # Deep Q-Network implementation
│   ├── q_learning.py  # Q-Learning implementation
│   └── sarsa.py       # SARSA implementation (planned)
├── requirements.txt   # Project dependencies
├── pyproject.toml     # Python project configuration
└── README.md          # Project documentation
```

## License

MIT License
