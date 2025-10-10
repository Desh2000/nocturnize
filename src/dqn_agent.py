# -----------------------------------------------------------------------------
# dqn_agent.py - The brain of our operation.
#
# This file contains the implementation of our Deep Q-Network (DQN) agent.
# It includes two main classes:
#   1. ReplayBuffer: Stores the agent's experiences for later learning.
#   2. DQNAgent: The main agent class that holds the neural network,
#               chooses actions, and learns from the ReplayBuffer.
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# --- 1. The Agent's Memory: Replay Buffer ---
# This class stores the agent's experiences and allows for random sampling.


class ReplayBuffer:
    """
    A simple fixed-size buffer to store experience tuples.
    This is a crucial component of the DQN algorithm.
    """

    def __init__(self, buffer_size):
        # A deque is a double-ended queue. It's efficient for adding and removing items.
        # We set a maxlen to ensure the buffer doesn't grow indefinitely.
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to the memory."""
        # The 'experience' is a tuple containing all the information from one step.
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Samples a random batch of experiences from memory."""
        # random.sample is an efficient way to get a random subset of the buffer.
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

# --- 2. The Decision-Maker: DQN Agent ---


class DQNAgent:
    """
    The main Deep Q-Network agent class.
    """

    def __init__(self, state_shape, action_space_size, learning_rate, discount_factor):
        # The shape of the input state (e.g., [battery, hour, price, deadline])
        self.state_shape = state_shape
        # The number of possible actions (e.g., 2 for Charge/Don't Charge)
        self.action_space_size = action_space_size

        # --- Hyperparameters ---
        self.gamma = discount_factor       # Discount factor for future rewards
        self.learning_rate = learning_rate  # Learning rate for the optimizer

        # --- Epsilon-Greedy Strategy Parameters for Exploration ---
        # Initial exploration rate (100% random actions)
        self.epsilon = 1.0
        self.epsilon_min = 0.01     # Minimum exploration rate
        self.epsilon_decay = 0.995  # Rate at which exploration decreases

        # --- Instantiate the Replay Buffer ---
        self.replay_buffer = ReplayBuffer(buffer_size=2000)

        # --- Build the Neural Network Model ---
        # This is the core of the agent that learns to predict Q-values.
        self.model = self._build_model()

    def _build_model(self):
        """Builds the deep neural network."""
        model = Sequential()
        # Input Layer: Must match the shape of our state.
        model.add(Dense(32, input_dim=self.state_shape, activation='relu'))
        # Hidden Layer 1: A layer with 32 neurons.
        model.add(Dense(32, activation='relu'))
        # Output Layer: Must have a neuron for each possible action.
        # 'linear' activation is used for regression-like outputs (Q-values).
        model.add(Dense(self.action_space_size, activation='linear'))

        # Compile the model with the Adam optimizer and Mean Squared Error loss.
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def choose_action(self, state):
        """
        Chooses an action using the epsilon-greedy policy.
        The agent will either explore (random action) or exploit (best known action).
        """
        # With probability epsilon, take a random action (explore).
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space_size)

        # Otherwise, use the model to predict the best action (exploit).
        q_values = self.model.predict(state, verbose=0)
        # Returns the index of the highest Q-value
        return np.argmax(q_values[0])

    def learn(self, batch_size):
        """
        Trains the neural network using a random batch of experiences from the replay buffer.
        """
        # Don't try to learn if the buffer doesn't have enough memories yet.
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a random minibatch of experiences.
        minibatch = self.replay_buffer.sample(batch_size)

        # --- The Core DQN Learning Logic ---
        # We loop through each memory in the minibatch to calculate the target Q-value.
        for state, action, reward, next_state, done in minibatch:
            # If the episode is over, the target is simply the immediate reward.
            if done:
                target = reward
            else:
                # If not done, the target is the reward plus the discounted value of the future.
                # The future value is the maximum Q-value for the next state, predicted by our model.
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state, verbose=0)[0])

            # Get the model's current Q-value predictions for the original state.
            target_f = self.model.predict(state, verbose=0)
            # Update only the Q-value for the action that was actually taken.
            target_f[0][action] = target

            # Train the model on this one memory (state -> target_f).
            # This teaches the model to make its future predictions closer to our calculated 'target'.
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # --- Decay Epsilon ---
        # After each learning step, we slightly reduce epsilon to decrease random exploration over time.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
