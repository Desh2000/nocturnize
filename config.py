# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# config.py - Main configuration file for the nocturnize project.
#
# This file centralizes all the key parameters for the simulation environment,
# CEB tariff structure, and the Reinforcement Learning agent's training process.
# By modifying the values here, you can experiment with different scenarios
# without altering the core logic of the agent or the environment.
#
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------
# 1. SIMULATION ENVIRONMENT PARAMETERS
# -----------------------------------------------------------------------------
# These parameters define the physical characteristics of our simulation world.

# EV (Electric Vehicle) Specifications (based on a Nissan Leaf 40 kWh model)
EV_BATTERY_CAPACITY_KWH = 40.0  # Total capacity of the EV's battery in kilowatt-hours (kWh).

# Charger Specifications (based on a standard Level 2 home charger)
CHARGER_POWER_KW = 7.4  # Power of the charger in kilowatts (kW). This is how much energy is added per hour.

# Simulation Time Parameters
SIMULATION_START_HOUR = 18  # The simulation for each day starts at 6 PM (18:00).
SIMULATION_END_HOUR = 8     # The simulation for each day ends at 8 AM (08:00) of the next day.
MIN_CHARGE_TARGET_PCT = 95.0 # The required battery charge percentage by the deadline.

# -----------------------------------------------------------------------------
# 2. SRI LANKA CEB TARIFF PARAMETERS (as of late 2024/early 2025)
# -----------------------------------------------------------------------------
# This section models the real-world Time-of-Day (ToD) tariff structure
# provided by the Ceylon Electricity Board (CEB) for domestic users.
# All prices are in Sri Lankan Rupees (LKR) per kWh.

# Define the time periods in 24-hour format.
# Note: The 'end' hour is exclusive (e.g., 22.5 means up to 10:29 PM).
PEAK_HOURS = (18.5, 22.5)    # 6:30 PM to 10:30 PM
DAY_HOURS = (5.5, 18.5)      # 5:30 AM to 6:30 PM
OFF_PEAK_HOURS = (22.5, 5.5) # 10:30 PM to 5:30 AM (spans across midnight)

# Define the prices for each period.
PRICE_PEAK_LKR_PER_KWH = 54.0
PRICE_DAY_LKR_PER_KWH = 25.0
PRICE_OFF_PEAK_LKR_PER_KWH = 13.0

# -----------------------------------------------------------------------------
# 3. DQN AGENT & TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
# These are the "tuning knobs" for our Deep Q-Network agent. They control how
# the agent learns. Finding good values for these is a key part of ML.

# Neural Network Architecture
STATE_SIZE = 4              # [battery_level_pct, current_hour, hours_until_deadline, current_price]
ACTION_SIZE = 2             # 0: Do Not Charge, 1: Charge
HIDDEN_LAYER_SIZES = [64, 64] # Two hidden layers with 64 neurons each.

# Training Hyperparameters
LEARNING_RATE = 0.001       # How big of a step the algorithm takes during learning.
DISCOUNT_FACTOR = 0.95      # Gamma (γ). How much the agent values future rewards over immediate ones.
REPLAY_BUFFER_SIZE = 10000  # How many past experiences (state, action, reward, next_state) to store.
BATCH_SIZE = 64             # How many experiences to sample from the buffer for each learning step.

# Epsilon-Greedy Strategy for Exploration vs. Exploitation
EPSILON_START = 1.0         # Starting value of epsilon. 100% random actions at the beginning.
EPSILON_MIN = 0.01          # Minimum value of epsilon. Agent will always take at least 1% random actions.
EPSILON_DECAY = 0.995       # The rate at which epsilon decreases after each episode. (e.g., 1.0 -> 0.995 -> ...)

# Update Frequencies
TARGET_NETWORK_UPDATE_FREQ = 5 # How often (in episodes) to copy weights from the main network to the target network.

# Training Loop Settings
TOTAL_EPISODES = 2000       # The total number of charging nights (episodes) to simulate for training.
