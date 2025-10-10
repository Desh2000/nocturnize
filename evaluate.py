# evaluate.py - A script to evaluate and visualize the trained agent's performance.
#
# This script loads a pre-trained model and compares its performance against
# baseline strategies. It calculates key metrics and generates plots that are
# essential for the final project report.
#
# To run: `python evaluate.py` from the project's root directory.
# -----------------------------------------------------------------------------

# --- Path Setup ---
# Ensures we can import our custom modules from the 'src' directory.
import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'src')))

# --- Imports ---
import matplotlib.pyplot as plt
import config
from environment import EVChargingEnv
import tensorflow as tf
import numpy as np

# --- Evaluation Parameters ---
MODEL_PATH = "dqn_nocturnize_1000_episodes_optimized.h5"  # The brain we want to test
NUM_TEST_EPISODES = 100  # How many nights to test for statistical significance

# --- Helper Function to Run a Simulation ---


def run_simulation(env, agent=None, strategy=None):
    """Runs a single episode using either a trained agent or a fixed strategy."""
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        if agent:  # If we are testing a trained agent
            # Set epsilon to 0 to ensure the agent only uses its learned knowledge (no random actions)
            agent.epsilon = 0
            state_reshaped = np.reshape(
                state, [1, env.observation_space.shape[0]])
            action = agent.choose_action(state_reshaped)
        elif strategy:  # If we are testing a baseline strategy
            action = strategy(state)
        else:
            raise ValueError("Must provide either an agent or a strategy.")

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    # Check if the episode was a success (car charged by deadline)
    success = info.get('success', False)
    return total_reward, success

# --- Baseline Strategies ---


def naive_strategy(state):
    """A simple strategy: always charge if the battery is not full."""
    battery_level = state[0]
    if battery_level < 99:
        return 1  # Action 1: Charge
    return 0  # Action 0: Don't Charge


# --- Main Evaluation Logic ---
if __name__ == "__main__":
    print(f"--- Evaluating model: {MODEL_PATH} ---")

    # 1. Initialize Environment and Load Trained Agent
    env = EVChargingEnv()

    # This is where we load the saved brain
    trained_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # We need to create a dummy agent to hold the model, but we won't train it.
    # The actual DQNAgent class isn't needed here, just its model.
    # A simpler way is to just call the model directly.

    # 2. Evaluate the Trained Agent
    agent_rewards = []
    agent_successes = 0
    print(f"\nRunning {NUM_TEST_EPISODES} episodes with the DQN agent...")
    for i in range(NUM_TEST_EPISODES):
        state, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_reshaped = np.reshape(
                state, [1, env.observation_space.shape[0]])
            q_values = trained_model.predict(state_reshaped, verbose=0)
            action = np.argmax(q_values[0])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        agent_rewards.append(total_reward)
        if info.get('success', False):
            agent_successes += 1

    # 3. Evaluate the Naive Baseline Strategy
    naive_rewards = []
    naive_successes = 0
    print(f"Running {NUM_TEST_EPISODES} episodes with the Naive strategy...")
    for _ in range(NUM_TEST_EPISODES):
        total_reward, success = run_simulation(env, strategy=naive_strategy)
        naive_rewards.append(total_reward)
        if success:
            naive_successes += 1

    # 4. Calculate and Print Final Statistics
    avg_agent_reward = np.mean(agent_rewards)
    agent_success_rate = (agent_successes / NUM_TEST_EPISODES) * 100

    avg_naive_reward = np.mean(naive_rewards)
    naive_success_rate = (naive_successes / NUM_TEST_EPISODES) * 100

    print("\n--- Evaluation Results ---")
    print(f"DQN Agent:")
    print(f"  - Average Reward: {avg_agent_reward:.2f} LKR")
    print(f"  - Success Rate: {agent_success_rate:.1f}%")
    print(f"\nNaive Strategy (Always Charge):")
    print(f"  - Average Reward: {avg_naive_reward:.2f} LKR")
    print(f"  - Success Rate: {naive_success_rate:.1f}%")

    # 5. Generate and Save a Comparison Plot
    strategies = ['DQN Agent', 'Naive Strategy']
    avg_rewards = [avg_agent_reward, avg_naive_reward]

    # Create a 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    plt.figure(figsize=(8, 6))
    plt.bar(strategies, avg_rewards, color=['#4CAF50', '#F44336'])
    plt.ylabel('Average Reward (LKR)')
    plt.title('Performance Comparison: DQN Agent vs. Naive Strategy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a file
    plot_filename = 'results/performance_comparison.png'
    plt.savefig(plot_filename)
    print(f"\nComparison plot saved to: {plot_filename}")
    plt.show()
