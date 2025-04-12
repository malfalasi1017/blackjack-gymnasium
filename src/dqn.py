import gymnasium as gym
import numpy as np

import random
import torch
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

from collections import deque

use_pre_trained_model = False
total_episodes = 100_000

# Hpyerparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_INTERVAL = 500
TRAIN_STARTS = 10000
TRAIN_FREQ = 2
MAX_STEPS = 1000
MODEL_PATH = "./models/dqn_blackjack.pth"

# Neural Network for DQN
class DQNNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256) 
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    
def create_dqn_grids(model, usable_ace=False):
    """Create value and policy grid given a DQN model."""
    # Initialize state_value and policy dictionaries
    state_value = {}
    policy = {}
    
    # For each player sum (12-21) and dealer card (1-10)
    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            # Create state tensor
            state = torch.tensor([player_sum, dealer_card, float(usable_ace)], dtype=torch.float32)
            
            # Get Q-values from the model
            with torch.no_grad():
                q_values = model(state).numpy()
            
            # Store state value and policy
            state_tuple = (player_sum, dealer_card, usable_ace)
            state_value[state_tuple] = float(np.max(q_values))
            policy[state_tuple] = int(np.argmax(q_values))
    
    # Create meshgrid for visualization
    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11),
    )
    
    # Create value grid
    value = np.zeros_like(player_count, dtype=float)
    for i, player_sum in enumerate(range(12, 22)):
        for j, dealer_card in enumerate(range(1, 11)):
            state_tuple = (player_sum, dealer_card, usable_ace)
            value[j, i] = state_value.get(state_tuple, 0.0)
    
    value_grid = player_count, dealer_count, value
    
    # Create policy grid
    policy_grid = np.zeros_like(player_count, dtype=int)
    for i, player_sum in enumerate(range(12, 22)):
        for j, dealer_card in enumerate(range(1, 11)):
            state_tuple = (player_sum, dealer_card, usable_ace)
            policy_grid[j, i] = policy.get(state_tuple, 0)
    
    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title):
    """Creates a plot using a value and policy grid."""
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # Plot state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # Plot policy (0=stick, 1=hit in Blackjack)
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # Add legend
    legend_elements = [
        Patch(facecolor="grey", edgecolor="black", label="Stick (0)"),
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit (1)"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig

def visualize_dqn_policy(model):
    """Visualize the DQN policy with and without usable ace."""
    print("\nGenerating policy visualization...")
    
    # With usable ace
    value_grid, policy_grid = create_dqn_grids(model, usable_ace=True)
    fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    
    # Without usable ace
    value_grid, policy_grid = create_dqn_grids(model, usable_ace=False)
    fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
    
    plt.show()

def main():
    # Initalize environment
    env = gym.make("Blackjack-v1", natural=True)
    state_dim = 3
    action_dim = env.action_space.n

    # Initialize networks
    online_net = DQNNetwork(state_dim, action_dim)
    target_net = DQNNetwork(state_dim, action_dim)

    if os.path.exists(MODEL_PATH) and use_pre_trained_model:
        train = False
        print("Loading pre-trained model...")
        online_net.load_state_dict(torch.load(MODEL_PATH))
        target_net.load_state_dict(online_net.state_dict())
    else:
        print("Training new model...")
        train = True

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)


    """ Training Phase """
    if train:
        steps = 0
        epsilon = EPSILON_START

        episode_reward = []

        for episode in range(total_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            total_reward = 0

            while not done:
                steps += 1

                # Epsilon-greedy action selection
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = torch.argmax(online_net(state)).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = torch.tensor(next_state, dtype=torch.float32)

                # Store transition in replay buffer
                replay_buffer.store((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                # Reduce epsilon
                epsilon = max(EPSILON_END, EPSILON_START - steps / EPSILON_DECAY)

                # Train the model every TRAIN_FREQ steps
                if replay_buffer.size() > TRAIN_STARTS and steps % TRAIN_FREQ == 0:
                    batch = replay_buffer.sample(batch_size=BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.stack(states)
                    actions = torch.tensor(actions)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.stack(next_states)
                    dones = torch.tensor(dones, dtype=torch.float32)

                    # Compute Q-values using target network
                    with torch.no_grad():
                        max_next_q_values = target_net(next_states).max(dim=1)[0]
                        targets = rewards + GAMMA * max_next_q_values * (1 - dones)

                    # Compute Q-values for actions taken
                    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                    # Compute loss and update
                    loss = torch.nn.MSELoss()(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Update target network
                if steps % TARGET_UPDATE_INTERVAL == 0:
                    target_net.load_state_dict(online_net.state_dict()) 
            
            episode_reward.append(total_reward)

            # Print reward every 100 episodes
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{total_episodes}, Reward: {total_reward}")

        # Save the model
        torch.save(online_net.state_dict(), MODEL_PATH)

    env.close()

    """ Testing Phase """
    test_episodes = 100
    test_rewards = []

    print("\nStarting testing phase...")

    for episode_idx in range(test_episodes):
        # Use human rendering only for first 5 episodes
        render_mode = "human" if episode_idx < 5 else None
        test_env = gym.make("Blackjack-v1", natural=True, render_mode=render_mode)

        state, _ = test_env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                action = torch.argmax(online_net(state)).item()

            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32)

            state = next_state
            total_reward += reward

            # Slow down rendering for clarity
            if episode_idx < 5:
                time.sleep(0.2)
        
        # Store this episode's reward
        test_rewards.append(total_reward)
        test_env.close()
        
        if (episode_idx + 1) % 10 == 0:
            print(f"Test episode {episode_idx + 1}/{test_episodes}, Reward: {total_reward}")

    avg_test_reward = np.mean(test_rewards)

    print("\nFinal Testing Results:")
    print(f"Average Test Reward: {avg_test_reward:.2f}")
    print(f"Number of positive rewards: {sum(r > 0 for r in test_rewards)}/{test_episodes}")
    
    # Visualize the agent's learned policy
    visualize_dqn_policy(online_net)

if __name__ == "__main__":
    main()