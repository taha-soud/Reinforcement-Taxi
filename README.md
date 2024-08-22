# Relational-Taxi
# Taxi-v3 Q-Learning Project

This project involves training and evaluating a Q-Learning model on the Taxi-v3 environment from OpenAI's Gym. The project is divided into two phases: Training and Evaluation.

## Overview

The goal of the Taxi-v3 environment is for the taxi to pick up a passenger at one location and drop them off at another location. The taxi earns rewards for successfully dropping off the passenger and incurs penalties for illegal moves or wrong pickups/drop-offs.

## Project Structure

The project is divided into two phases:

1. **Phase 1: Training the Model**
2. **Phase 2: Testing and Evaluation**

### Phase 1: Training the Model

In this phase, we train the Q-Learning model using different hyperparameters such as the learning rate, discount factor, and exploration rate. The goal is to learn the optimal policy that will maximize the total reward.

#### Hyperparameters

- **Learning Rate (`learning_rate`)**: 0.1
- **Discount Factor (`discount_factor`)**: 0.5 (default), but other factors are explored in Phase 2.
- **Exploration Rate (`exploration`)**: 0.1
- **Epochs**: 1000, 5000, 10000 (varied during training)

#### Q-Learning Algorithm

- **Q-Table Initialization**: A table of zeros with dimensions `[state_space, action_space]` (500 x 6).
- **Training Process**:
  1. Reset the environment to get the initial state.
  2. For each epoch, perform the following:
     - Select an action using an epsilon-greedy policy (explore or exploit).
     - Perform the action in the environment and observe the reward and the next state.
     - Update the Q-value using the Q-Learning formula:
       \[
       Q(s, a) = (1 - \text{{learning_rate}}) \times Q(s, a) + \text{{learning_rate}} \times \left(\text{{reward}} + \text{{discount_factor}} \times \max(Q(s', a'))\right)
       \]
     - Accumulate rewards and steps for each epoch.

#### Training Script

The training script loops through different epoch settings (`epochs_list`) and trains the model. After training, it prints out the rewards and steps for each epoch and evaluates the model's performance.

### Phase 2: Testing and Evaluation

In this phase, we evaluate the model's performance using different discount factors. We analyze how the choice of discount factor affects the average trip length and average trip reward.

#### Hyperparameters

- **Learning Rate (`learning_rate`)**: 0.1
- **Discount Factors (`discount_factors`)**: [0.3, 0.5, 0.7]
- **Epochs**: 10,000

#### Evaluation Process

- **Trip Evaluation**: The model is evaluated over a fixed number of trips. For each trip, the model follows the learned policy (selecting actions based on the highest Q-value) to complete the task. The trip length and total reward are recorded.

- **Average Metrics**: For each discount factor, the average trip length and average trip reward are calculated and compared.

#### Results

The script runs the evaluation for each discount factor and prints out the following results:

- Discount Factor
- Average Trip Length
- Average Trip Reward

### Running the Project

1. **Install Dependencies**:
   ```bash
   pip install gym numpy
   python taxi_q_learning.py
