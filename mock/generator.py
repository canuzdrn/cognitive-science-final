import numpy as np
import pandas as pd

"""
This file contains the script that generates mock data
mock data has columns of Trial, Observation, Action, Reward and State
relevancy of columns are described in paper hence we try to come up with similar features
"""

# predefined params
num_trials = 1000
num_states = 2
actions = ['A', 'B']
observations = ['O1', 'O2']
reward_probs = {'O1': {'A': 0.7, 'B': 0.3}, 'O2': {'A': 0.4, 'B': 0.6}}

# transition probabilities of hmm
transition_probs = np.array([[0.9, 0.1], [0.2, 0.8]])

data = []
# init state
current_state = np.random.choice(num_states)

for t in range(num_trials):
    # generate random observations
    observation = np.random.choice(observations)
    
    if current_state == 0:  # Random state
        action = np.random.choice(actions)
    else:                   # Policy state
        action_probs = [reward_probs[observation][a] for a in actions]
        action = np.random.choice(actions, p=action_probs)
    
    reward = np.random.choice([0, 1], p=[1 - reward_probs[observation][action], reward_probs[observation][action]])
    data.append([t + 1, observation, action, reward, current_state])
    
    # state transition
    current_state = np.random.choice(num_states, p=transition_probs[current_state])

# create dataframe accordingly
columns = ['Trial', 'Observation', 'Action', 'Reward', 'State']
df = pd.DataFrame(data, columns=columns)
df.to_csv('sample_dataset.csv', index=False)