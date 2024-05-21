import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = pd.read_csv('sample_dataset.csv')

# definition of reward probs for given actions
reward_probs = {
    'O1': {'A': 0.7, 'B': 0.3},
    'O2': {'A': 0.4, 'B': 0.6}
}

def static_noise_inference(epsilon, data):
    actions = data['Action'].unique()
    A = len(actions)
    L_theta = 0
    
    for _, row in data.iterrows():
        observation = row['Observation']
        action = row['Action']
        
        # calculate action given observation P(at|ot)
        action_prob = reward_probs[observation][action]
        
        # log-likelihood
        L_theta += np.log(epsilon * (1 / A) + (1 - epsilon) * action_prob)
    
    return -L_theta  # negative of log-likelihood for minimization task

# hyperparam optimization for static noise inference (optimize epsilon)
result_static = minimize(static_noise_inference, x0=[0.1], args=(data,), bounds=[(0, 1)])
best_epsilon = result_static.x[0]
print(f'Optimized epsilon for Static Noise Inference: {best_epsilon}')
print(f'Static Noise Inference Log-Likelihood: {-result_static.fun}')


def dynamic_noise_inference(params, data):
    T_10, T_01 = params
    actions = data['Action'].unique()
    A = len(actions)
    num_trials = len(data)
    
    L_theta = 0
    lambda_t = np.zeros((num_trials, 2))
    lambda_t[0, :] = 0.5  # prob for each initial state
    
    epsilon = 1e-10
    
    for t in range(num_trials):
        row = data.iloc[t]
        observation = row['Observation']
        action = row['Action']
        
        # calculate the action probability P(at|ot)
        action_prob = reward_probs[observation][action]
        
        # likelihood calculation
        likelihood_random = 1 / A
        likelihood_policy = action_prob
        weighted_likelihood = (1 / A) * lambda_t[t-1, 0] + action_prob * lambda_t[t-1, 1] + epsilon
        L_t = np.log(weighted_likelihood)
        L_theta += L_t
        
        if t < num_trials - 1:
            # update prediction probs
            lambda_t[t+1, 0] = ((1 / A) * lambda_t[t, 0] * T_01 + action_prob * lambda_t[t, 1] * T_10) / weighted_likelihood
            lambda_t[t+1, 1] = (action_prob * lambda_t[t, 1] * (1 - T_10)) / weighted_likelihood
    
    return -L_theta  # negative likelihood for minimization task

# optimize hyperparams (T01 and T10 - state transition from attentive to inattentive and vice versa)
initial_params = [0.1, 0.1]
bounds = [(0, 1), (0, 1)]
result_dynamic = minimize(dynamic_noise_inference, x0=initial_params, args=(data,), bounds=bounds)
best_T_10, best_T_01 = result_dynamic.x
print(f'Optimized T_10 for Dynamic Noise Inference: {best_T_10}')
print(f'Optimized T_01 for Dynamic Noise Inference: {best_T_01}')
print(f'Dynamic Noise Inference Log-Likelihood: {-result_dynamic.fun}')

def compute_state_probabilities(params, data):
    T_10, T_01 = params
    actions = data['Action'].unique()
    A = len(actions)
    num_trials = len(data)
    
    lambda_t = np.zeros((num_trials, 2))
    lambda_t[0, :] = 0.5  # initial probs of states
    
    epsilon = 1e-10
    
    for t in range(num_trials):
        row = data.iloc[t]
        observation = row['Observation']
        action = row['Action']
        
        # P(at|ot)
        action_prob = reward_probs[observation][action]
        
        # likelihood calculation
        likelihood_random = 1 / A
        likelihood_policy = action_prob
        weighted_likelihood = (1 / A) * lambda_t[t-1, 0] + action_prob * lambda_t[t-1, 1] + epsilon
        
        if t < num_trials - 1:
            lambda_t[t+1, 0] = ((1 / A) * lambda_t[t, 0] * T_01 + action_prob * lambda_t[t, 1] * T_10) / weighted_likelihood
            lambda_t[t+1, 1] = (action_prob * lambda_t[t, 1] * (1 - T_10)) / weighted_likelihood
    
    return lambda_t

# optimized params for dynamic noise inference
optimized_params = [best_T_10, best_T_01]

# compute state probabilities
state_probabilities = compute_state_probabilities(optimized_params, data)

plt.figure(figsize=(12, 6))
plt.plot(state_probabilities[:, 0], label='Random State Probability')
plt.plot(state_probabilities[:, 1], label='Policy State Probability')
plt.xlabel('Trial')
plt.ylabel('State Probability')
plt.title('Inferred State Probabilities Over Trials')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data['Trial'], data['Reward'].cumsum() / (data['Trial']), label='Cumulative Accuracy', color='black')
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Trials')
plt.legend()
plt.show()

def compute_noise_levels_static(epsilon, data):
    return np.full(len(data), epsilon)

def compute_noise_levels_dynamic(params, data):
    T_10, T_01 = params
    actions = data['Action'].unique()
    A = len(actions)
    num_trials = len(data)
    
    lambda_t = np.zeros((num_trials, 2))
    lambda_t[0, :] = 0.5 
    
    epsilon = 1e-10
    
    for t in range(num_trials):
        row = data.iloc[t]
        observation = row['Observation']
        action = row['Action']
        
        # P(at|ot)
        action_prob = reward_probs[observation][action]
        
        likelihood_random = 1 / A
        likelihood_policy = action_prob
        weighted_likelihood = (1 / A) * lambda_t[t-1, 0] + action_prob * lambda_t[t-1, 1] + epsilon
        
        if t < num_trials - 1:
            # update prediction probs
            lambda_t[t+1, 0] = ((1 / A) * lambda_t[t, 0] * T_01 + action_prob * lambda_t[t, 1] * T_10) / weighted_likelihood
            lambda_t[t+1, 1] = (action_prob * lambda_t[t, 1] * (1 - T_10)) / weighted_likelihood
    
    return lambda_t[:, 0]

optimized_epsilon = best_epsilon
optimized_params_dynamic = [best_T_10, best_T_01]

# noise levels
noise_levels_static = compute_noise_levels_static(optimized_epsilon, data)
noise_levels_dynamic = compute_noise_levels_dynamic(optimized_params_dynamic, data)

plt.figure(figsize=(12, 6))
plt.scatter(data['Trial'], noise_levels_static, label='Static', color='red', s=20)
plt.scatter(data['Trial'], noise_levels_dynamic, label='Dynamic', color='blue', s=5)
plt.xlabel('Trial')
plt.ylabel('Noise Level / Accuracy')
plt.title('Examples of noise inference')
plt.legend()
plt.show()