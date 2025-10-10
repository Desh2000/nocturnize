# -----------------------------------------------------------------------------
# train.py - The main script to train the DQN agent. (Optimized Version)
#
# This script brings all the components together. The key optimization is that
# the agent now learns once per episode (night), not on every single step (hour),
# which dramatically speeds up the training process.
# -----------------------------------------------------------------------------

# --- Path Setup ---
# CRITICAL: This block MUST come before any imports from our 'src' folder.
import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'src')))

# --- Imports ---
import config
from dqn_agent import DQNAgent
from environment import EVChargingEnv
import numpy as np

# --- Main Training Logic ---
if __name__ == "__main__":
    print("--- Starting OPTIMIZED DQN Agent Training for nocturnize ---")

    # 1. Initialize Environment and Agent
    env = EVChargingEnv()
    state_shape = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    agent = DQNAgent(
        state_shape=state_shape,
        action_space_size=action_space_size,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR
    )

    # --- Training Loop ---
    for episode in range(1, config.TOTAL_EPISODES + 1):
        state, info = env.reset()
        state = np.reshape(state, [1, state_shape])

        total_reward = 0
        done = False

        # This inner loop runs for each hour of the night.
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_shape])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # --- End of Episode ---

        # 6. Agent Learns from Memory (OPTIMIZED: Happens once per episode)
        # We wait until the agent has enough experiences in its memory before learning.
        if len(agent.replay_buffer) > config.BATCH_SIZE:
            agent.learn(config.BATCH_SIZE)

        # Print a summary of the episode's performance.
        print(f"Episode: {episode}/{config.TOTAL_EPISODES} | "
              f"Total Reward: {total_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.2f}")

    # --- Training Complete ---
    print("\n--- Training Finished ---")

    # 7. Save the Trained Model
    model_filename = f"dqn_nocturnize_{config.TOTAL_EPISODES}_episodes_optimized.h5"
    agent.model.save(model_filename)
    print(f"Trained model saved to: {model_filename}")
