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
# Total capacity of the EV's battery in kilowatt-hours (kWh).
EV_BATTERY_CAPACITY_KWH = 40.0

# Charger Specifications (based on a standard Level 2 home charger)
# Power of the charger in kilowatts (kW). This is how much energy is added per hour.
CHARGER_POWER_KW = 7.4

# Simulation Time Parameters
# The simulation for each day starts at 6 PM (18:00).
SIMULATION_START_HOUR = 18
# The simulation for each day ends at 8 AM (08:00) of the next day.
SIMULATION_END_HOUR = 8
# The required battery charge percentage by the deadline.
MIN_CHARGE_TARGET_PCT = 95.0

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
OFF_PEAK_HOURS = (22.5, 5.5)  # 10:30 PM to 5:30 AM (spans across midnight)

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
# [battery_level_pct, current_hour, hours_until_deadline, current_price]
STATE_SIZE = 4
ACTION_SIZE = 2             # 0: Do Not Charge, 1: Charge
HIDDEN_LAYER_SIZES = [64, 64]  # Two hidden layers with 64 neurons each.

# Training Hyperparameters
# How big of a step the algorithm takes during learning.
LEARNING_RATE = 0.001
# Gamma (γ). How much the agent values future rewards over immediate ones.
DISCOUNT_FACTOR = 0.95
# How many past experiences (state, action, reward, next_state) to store.
REPLAY_BUFFER_SIZE = 10000
# How many experiences to sample from the buffer for each learning step.
BATCH_SIZE = 64

# Epsilon-Greedy Strategy for Exploration vs. Exploitation
# Starting value of epsilon. 100% random actions at the beginning.
EPSILON_START = 1.0
# Minimum value of epsilon. Agent will always take at least 1% random actions.
EPSILON_MIN = 0.01
# The rate at which epsilon decreases after each episode. (e.g., 1.0 -> 0.995 -> ...)
EPSILON_DECAY = 0.995

# Update Frequencies
# How often (in episodes) to copy weights from the main network to the target network.
TARGET_NETWORK_UPDATE_FREQ = 5

# Training Loop Settings
# The total number of charging nights (episodes) to simulate for training.
TOTAL_EPISODES = 1000
