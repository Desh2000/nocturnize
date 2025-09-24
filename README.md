Project Charter & Technical Specification: nocturnize
Document Version: 1.0
Last Updated: 2025-09-24
Project Name: nocturnize
License: Apache-2.0

1. Introduction
1.1. Purpose
This document outlines the complete project plan, technical architecture, and development strategy for nocturnize. It serves as the central guidance for all team members, defining the project's objectives, scope, requirements, and execution plan. The primary goal of nocturnize is to develop and compare two Reinforcement Learning (RL) agents capable of creating optimal, cost-effective charging schedules for Electric Vehicles (EVs) in Sri Lanka.

1.2. Scope
The project encompasses the following key activities:

Simulation: Development of a realistic EV charging simulation environment that models battery mechanics, charging speeds, and the official CEB Time-of-Day (ToD) electricity tariffs.

Agent Development: Implementation, training, and evaluation of two distinct RL agents: a Deep Q-Network (DQN) agent and an Advantage Actor-Critic (A2C) agent.

Analysis: A comparative analysis of the performance of both agents against each other and against baseline, non-intelligent charging strategies.

Reporting: A final report and presentation detailing the project's findings, aligned with the SE4050-Deep Learning assignment criteria.

Out of scope for this project is the development of a production-ready mobile/web application and physical hardware integration. The project focuses on delivering a robust, simulated proof-of-concept.

1.3. Definitions and Acronyms
|

| Term | Definition |
| RL | Reinforcement Learning - A type of machine learning where an agent learns to make decisions. |
| DQN | Deep Q-Network - A value-based RL algorithm that uses a neural network to estimate action values. |
| A2C | Advantage Actor-Critic - A policy-based RL algorithm with two neural networks (Actor and Critic). |
| EV | Electric Vehicle. |
| CEB | Ceylon Electricity Board - The primary electricity utility in Sri Lanka. |
| ToD | Time-of-Day - An electricity tariff structure with different prices for different times of the day. |
| kWh | Kilowatt-hour - A unit of energy, used to measure EV battery capacity and electricity consumption. |
| Agent | The AI model (our "brain") that makes the charging decisions. |
| Env | Environment - The simulated world (our "game") in which the agent operates. |
| State | A snapshot of the environment at a given time (e.g., battery level, current time). |
| Action | A decision made by the agent (e.g., charge, don't charge). |
| Reward | A feedback signal from the environment that tells the agent if its action was good or bad. |

2. Overall Description
2.1. Product Perspective
nocturnize is a proof-of-concept intelligent software system designed to function as the "brain" for a smart EV charging station. It addresses the growing need for efficient energy consumption by automatically scheduling EV charging to coincide with the cheapest electricity rates, without compromising the user's need for a fully charged vehicle by a specific time.

2.2. Product Functions
Simulate Charging: Accurately model the EV charging process based on real-world parameters.

Learn Optimal Policy: Use RL to learn a charging strategy that minimizes cost while guaranteeing charge by a user-defined deadline.

Adapt to Conditions: Respond dynamically to variable electricity prices, user deadlines, and unexpected events (e.g., power cuts).

Evaluate Performance: Provide clear metrics to compare the cost-effectiveness of the intelligent agent against simple charging strategies.

2.3. User Classes and Characteristics
For this project, the primary "user" is the development team. However, we will design the system from the perspective of a future end-user:

The EV Owner: A typical domestic electricity consumer in Sri Lanka who owns an EV and wants to minimize their monthly electricity bill. They are assumed to have access to a home charging station.

2.4. Operating Environment
The system is designed to operate within a simulated environment that mirrors the key conditions of Sri Lanka:

Electricity Tariffs: Based on the official CEB Domestic Time-of-Day (ToD) tariff structure.

Vehicle: Parameters based on a common EV model in Sri Lanka (e.g., Nissan Leaf 40 kWh).

Software: The system will be developed in Python and will run on a standard desktop/laptop computer for training and evaluation.

2.5. Assumptions and Dependencies
The user provides an accurate desired departure time and required state of charge.

The EV remains plugged in for the entire duration of the charging window unless specified otherwise by the user.

The CEB ToD tariff structure remains the primary source of electricity pricing.

The project depends on open-source Python libraries: TensorFlow, gymnasium, numpy.

3. System Requirements
3.1. Functional Requirements (FR)
| ID | Requirement |
| FR-01 | The system shall allow configuration of EV battery capacity, charger power, and electricity price schedules. |
| FR-02 | The simulation environment shall accurately track time, battery state of charge, and cumulative charging cost. |
| FR-03 | The RL agent shall take the current state (battery level, time, deadline, price) as input. |
| FR-04 | The RL agent shall output a discrete action (e.g., CHARGE, DO_NOT_CHARGE). |
| FR-05 | The system shall ensure the EV is charged to the user-defined target by the specified deadline. |
| FR-06 | The system shall log the total cost for each complete charging cycle. |
| FR-07 | The system shall be able to train, save, and load two separate RL agents (DQN and A2C). |

3.2. Non-Functional Requirements (NFR)
| ID | Requirement |
| NFR-01 | The code should be modular, with clear separation between the environment, agent, and training logic. |
| NFR-02 | The code should be well-commented and follow PEP 8 Python style guidelines. |
| NFR-03 | The training process should show a clear trend of improving rewards over time. |
| NFR-04 | The GitHub repository should have a clear commit history with descriptive messages from all members. |

3.3. Use Cases / User Stories
As a developer, I want to configure the simulation with different EV models and price scenarios so that I can test the agent's adaptability.

As a developer, I want to run the training script for the DQN agent so that I can generate a trained model file.

As a developer, I want to run the evaluation script with a trained agent so that I can generate graphs comparing its performance to a naive "charge immediately" strategy.

(Future) As an EV owner, I want to set my departure time and desired charge level so that the system can charge my car for the lowest possible cost.

4. System Architecture
4.1. High-Level Architecture
The system follows a standard Reinforcement Learning architectural pattern, separating the Agent from the Environment. A Training Script orchestrates the interaction between them, using a Configuration File to set the parameters for the simulation and training.

4.2. Technologies Used
Programming Language: Python 3.9+

Core ML/RL Libraries:

TensorFlow 2.x (for building the neural networks)

Gymnasium (for creating the custom simulation environment)

Data Handling: NumPy

Version Control: Git & GitHub

4.3. Data Flow Diagram (DFD) - Single Training Step
5. Detailed Design
5.1. Component Design
config.py: A non-executable Python file containing all constants and hyperparameters. This includes CEB tariff rates and times, EV battery specs, and neural network learning rates.

src/environment.py: Contains the EVChargingEnv class, inheriting from gymnasium.Env. It manages the simulation's state, executes actions, and calculates rewards.

src/dqn_agent.py: Contains the DQNAgent class. It holds the Keras/TensorFlow neural network models (q_network and target_network), the replay buffer, and the logic for choosing actions (epsilon-greedy) and learning from experience.

train.py: The main executable script. It initializes the environment and agent from the config, then runs the main training loop for a specified number of episodes, periodically saving the agent's model weights.

5.2. Class Diagram (Simplified)
6. User Interface (UI) Design
The primary interface for this project will be a Command-Line Interface (CLI).

Training: The user will run python train.py --agent dqn --episodes 1000 --output_path models/dqn_v1.

Evaluation: The user will run python evaluate.py --agent dqn --model_path models/dqn_v1 --runs 100.

6.1. Future UI Mockup Concept
For commercialization, a mobile app interface would be designed.

Main Screen: Shows current battery percentage, estimated time to full, and current charging cost.

Scheduling Screen: A simple interface with a time slider for the user to set their next required "Departure Time" and a toggle for "Required Charge Level" (e.g., 80% for daily use, 100% for long trips).

7. Test Plan
7.1. Test Cases
| ID | Component | Test Case | Expected Result |
| TC-01 | environment | Call step(CHARGE) for one hour with a 7.4 kW charger. | Battery level increases by 7.4 / battery_capacity. Cost increases by 7.4 * current_price. |
| TC-02 | environment | Set a deadline and run simulation without charging. | The final reward should include the large penalty for failure. |
| TC-03 | dqn_agent | Initialize the agent. | Two identical neural network models (q_network, target_network) are created. |
| TC-04 | train | Run a short training loop (10 episodes). | The script completes without errors. The agent's model file is saved. |

7.2. QA Strategy
The quality assurance strategy relies on three pillars:

Unit Testing: Key logic within the environment (e.g., cost calculation, battery charging) will be tested with specific inputs to verify correctness.

Integration Testing: The main training loop will be run for a small number of episodes to ensure the agent and environment interact correctly.

Performance Validation: The final trained models will be evaluated over a large number of test episodes to measure their average cost savings and success rate, which are the primary acceptance criteria.

8. Deployment and Maintenance
8.1. Deployment Strategy
"Deployment" for this project consists of saving the trained model weights (.h5 or .weights.h5 files from TensorFlow/Keras) to a designated models/ directory in the repository. The final deliverable will be the full GitHub repository containing the source code, documentation, and the trained model files.

8.2. Maintenance & Support
Logging: During training, the script will print key metrics to the console every N episodes (e.g., Episode 100/1000, Average Reward: -55.4, Epsilon: 0.67).

Issue Tracking: The team will use GitHub Issues to create, assign, and track all development tasks (e.g., "Implement Replay Buffer," "Tune DQN Hyperparameters").

Appendices
A.1. Glossary
(See Section 1.3)

A.2. References
Human-level control through deep reinforcement learning - Mnih et al., 2015 (The original DQN paper).

Ceylon Electricity Board Official Website (for tariff information).

Public Utilities Commission of Sri Lanka (for official tariff schedules).

Gymnasium & TensorFlow Documentation.

A.3. Revision History
| Version | Date | Author(s) | Summary of Changes |
| 1.0 | 2025-09-24 | Isuru & Gemini | Initial creation of the project charter document. |
