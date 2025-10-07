# -----------------------------------------------------------------------------
# test_environment.py - A simple script to test the EVChargingEnv.
#
# To run: `python test_environment.py` from the project's root directory.
# -----------------------------------------------------------------------------

# --- Path Setup ---
# This is a robust way to ensure the 'src' directory is on the Python path.
# It finds the directory of this script and then adds the 'src' subfolder.
import time
from environment import EVChargingEnv
import sys
import os

# Get the absolute path of the directory containing this script (the project root).
project_root = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the 'src' directory.
src_path = os.path.join(project_root, "src")
# Add the 'src' directory to the Python path.
sys.path.insert(0, src_path)

# --- Imports ---
# Now that the path is set up, this import will work correctly.

# --- Main Test Logic ---
if __name__ == "__main__":
    print("--- Testing EV Charging Environment ---")
    print(f"Attempting to import modules from: {src_path}")  # Debug print

    # 1. Initialize the environment
    env = EVChargingEnv()

    # 2. Reset the environment to get the initial state
    observation, info = env.reset()

    print("\nInitial State:")
    print(f"  - Observation: {observation}")

    # 3. Run a short simulation for 5 steps with random actions
    for i in range(5):
        action = env.action_space.sample()

        print(f"\n--- Step {i+1} ---")
        print(f"Action Taken: {'CHARGE' if action == 1 else 'DO NOT CHARGE'}")

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"  - New Observation: {observation}")
        print(f"  - Reward Received: {reward:.2f} LKR")
        print(f"  - Is Episode Done? {terminated}")
        time.sleep(1)

    print("\n--- Test Complete ---")
