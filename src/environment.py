# -----------------------------------------------------------------------------
# environment.py - The Electric Vehicle (EV) Charging Simulation Environment.
#
# This file defines the custom reinforcement learning environment for the
# nocturnize project. It simulates the process of charging an EV overnight,
# taking into account real-world parameters like battery capacity, charger power,
# and dynamic electricity tariffs based on the Sri Lankan CEB ToD schedule.
#
# The environment is built using the Gymnasium library, which provides a standard
# API for RL environments. This ensures compatibility with various RL agents and
# algorithms.
#
# -----------------------------------------------------------------------------

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config  # Import our configuration file


class EVChargingEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating the smart charging of an EV.

    The agent's goal is to charge the EV's battery to a target level by a
    deadline while minimizing the electricity cost.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        """
        Initializes the environment, setting up the state and action spaces,
        and loading parameters from the config file.
        """
        super().__init__()

        # -------------------- Action Space Definition --------------------
        # The agent has two discrete actions it can take at each time step:
        # 0: Do NOT charge for the next hour.
        # 1: Charge for the next hour.
        self.action_space = spaces.Discrete(2)

        # -------------------- State (Observation) Space Definition --------------------
        # The state is a multi-dimensional array containing all the information
        # the agent needs to make a decision. We define the bounds for each dimension.
        # state = [battery_level_pct, current_hour, hours_until_deadline, current_price]
        low_bounds = np.array([0, 0, 0, 0], dtype=np.float32)
        # Price high bound is arbitrary but safe
        high_bounds = np.array([100, 23, 24, 100], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, dtype=np.float32)

        # -------------------- Simulation Parameters from Config --------------------
        # Load all the simulation settings from our centralized config file.
        self.battery_capacity = config.EV_BATTERY_CAPACITY_KWH
        self.charger_power = config.CHARGER_POWER_KW
        self.charge_target_pct = config.MIN_CHARGE_TARGET_PCT

        # -------------------- Environment State Variables --------------------
        # These variables will be updated at each step of the simulation.
        self.current_battery_pct = 0.0
        self.current_hour = 0
        self.deadline_hour = 0

        print("EV Charging Environment Initialized.")
        print(f"  - Battery Capacity: {self.battery_capacity} kWh")
        print(f"  - Charger Power: {self.charger_power} kW")

    def _get_price(self, hour):
        """
        Calculates the electricity price for a given hour based on the
        CEB ToD tariff structure from the config file.
        """
        # We handle the case where the off-peak period wraps around midnight.
        # e.g., (22.5, 5.5)
        if config.OFF_PEAK_HOURS[0] > config.OFF_PEAK_HOURS[1]:
            if hour >= config.OFF_PEAK_HOURS[0] or hour < config.OFF_PEAK_HOURS[1]:
                return config.PRICE_OFF_PEAK_LKR_PER_KWH
        else:  # For contiguous periods
            if config.OFF_PEAK_HOURS[0] <= hour < config.OFF_PEAK_HOURS[1]:
                return config.PRICE_OFF_PEAK_LKR_PER_KWH

        if config.PEAK_HOURS[0] <= hour < config.PEAK_HOURS[1]:
            return config.PRICE_PEAK_LKR_PER_KWH

        # Any other time is considered Day tariff
        return config.PRICE_DAY_LKR_PER_KWH

    def _get_observation(self):
        """
        Constructs the current state array from the environment's variables.
        This is what the agent will "see".
        """
        hours_until_deadline = (self.deadline_hour -
                                self.current_hour + 24) % 24
        if hours_until_deadline == 0 and self.current_hour == self.deadline_hour:
            hours_until_deadline = 24  # Full day if deadline is same as current hour next day

        current_price = self._get_price(self.current_hour)

        return np.array([
            self.current_battery_pct,
            self.current_hour,
            hours_until_deadline,
            current_price
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new, random starting state for a new episode.
        This simulates the start of a new charging night.
        """
        super().reset(seed=seed)

        # 1. Reset the clock to the start of the simulation (e.g., 6 PM).
        self.current_hour = config.SIMULATION_START_HOUR

        # 2. Set a random starting battery level (e.g., between 10% and 50%).
        # This makes the agent robust to different starting conditions.
        self.current_battery_pct = self.np_random.uniform(low=10.0, high=50.0)

        # 3. Set a random deadline (e.g., between 6 AM and 8 AM).
        # This forces the agent to learn an adaptive strategy.
        self.deadline_hour = self.np_random.integers(low=6, high=9)

        # Get the initial observation and info
        observation = self._get_observation()
        info = {}  # We can pass extra debug info here if needed

        print(f"\n----- New Episode -----")
        print(
            f"Start Time: {self.current_hour}:00, Deadline: {self.deadline_hour}:00")
        print(f"Initial Battery: {self.current_battery_pct:.2f}%")

        return observation, info

    def step(self, action):
        """
        Executes one time step in the environment based on the agent's action.
        """
        # 1. Calculate energy added and cost based on the action.
        cost = 0.0
        energy_added_kwh = 0.0
        if action == 1:  # Action is to charge
            energy_added_kwh = self.charger_power * 1.0  # 1 hour of charging
            cost = energy_added_kwh * self._get_price(self.current_hour)

            # Update battery percentage
            charge_added_pct = (energy_added_kwh / self.battery_capacity) * 100
            self.current_battery_pct += charge_added_pct
            self.current_battery_pct = min(
                self.current_battery_pct, 100.0)  # Cap at 100%

        # 2. Advance the simulation time by one hour.
        self.current_hour = (self.current_hour + 1) % 24

        # 3. Determine if the episode is done (deadline is reached).
        terminated = self.current_hour == self.deadline_hour

        # 4. Calculate the reward. This is the most critical part.
        reward = 0.0
        if not terminated:
            # For every step that is not the end, the reward is the negative cost.
            # This incentivizes the agent to avoid high costs.
            reward = -cost
        # If the deadline is reached, calculate a large final reward/penalty.
        else:
            if self.current_battery_pct >= self.charge_target_pct:
                # Large positive reward for successfully charging the car on time.
                # Success reward, but still penalize the final hour's cost.
                reward = 200.0 - cost
            else:
                # Large negative penalty for failing to meet the charge target.
                # The penalty is proportional to how much charge is missing.
                missing_pct = self.charge_target_pct - self.current_battery_pct
                reward = -500.0 - (missing_pct * 10)  # Heavy penalty

        # Get the new observation
        observation = self._get_observation()

        # For Gymnasium compatibility, we also return 'truncated' and 'info'
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
