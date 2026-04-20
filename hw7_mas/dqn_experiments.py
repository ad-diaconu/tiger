import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from itertools import count
from typing import Union, Tuple, Callable

# ---------------------------------------------------------
# Task 1: Replay Buffer
# ---------------------------------------------------------
class ReplayBuffer(object):
    def __init__(self, size: int = 10000):
        self.size    = size
        self.length  = 0
        self.idx     = -1
        
        self.states        = None
        self.states_next   = None
        self.actions       = None
        self.rewards       = None
        self.done          = None
        
    def store(self, s: Union[torch.Tensor, np.ndarray], a: int, r: float, 
              s_next: Union[torch.Tensor, np.ndarray], done: bool):
        
        if self.states is None:
            self.states      = torch.zeros([self.size] + list(s.shape))   
            self.states_next = torch.zeros_like(self.states)              
            self.actions     = torch.zeros((self.size, ))                 
            self.rewards     = torch.zeros((self.size, ))                 
            self.done        = torch.zeros((self.size, ))                 
        
        # Increment index using modulo to overwrite old experiences when full
        self.idx = (self.idx + 1) % self.size
        
        # Store current (s, a, r, s_next, done)
        self.states[self.idx]      = torch.as_tensor(s, dtype=torch.float32)
        self.actions[self.idx]     = a
        self.rewards[self.idx]     = r
        self.states_next[self.idx] = torch.as_tensor(s_next, dtype=torch.float32)
        self.done[self.idx]        = done
        
        # Increment buffer length 
        self.length = min(self.length + 1, self.size)
        
    def sample(self, batch_size: int = 128) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert self.length >= batch_size, "Can not sample from the buffer yet"
        indices = np.random.choice(a=np.arange(self.length), size=batch_size, replace=False)
        
        s      = self.states[indices]
        a      = self.actions[indices]
        r      = self.rewards[indices]
        s_next = self.states_next[indices]
        done   = self.done[indices]
        
        return s, a, r, s_next, done


# ---------------------------------------------------------
# Network Architecture & Epsilon Scheduler
# ---------------------------------------------------------
class DQN_RAM(nn.Module):
    def __init__(self, in_features: int, num_actions: int):
        super(DQN_RAM, self).__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def eps_generator(max_eps: float=0.9, min_eps: float=0.05, max_iter: int = 10000):
    crt_iter = -1
    while True:
        crt_iter += 1
        frac = min(crt_iter/max_iter, 1)
        eps = (1 - frac) * max_eps + frac * min_eps
        yield eps

def select_epilson_greedy_action(Q: nn.Module, s: Tensor, eps: float):
    rand = np.random.rand()
    if rand < eps:
        return np.random.choice(np.arange(Q.num_actions))
    with torch.no_grad():
        output = Q(s).argmax(dim=1).item()
    return output


# ---------------------------------------------------------
# Task 2 & 3: Target Computations
# ---------------------------------------------------------
@torch.no_grad()
def dqn_target(Q: nn.Module, target_Q: nn.Module, r_batch: Tensor, 
               s_next_batch: Tensor, done_batch: Tensor, gamma: float) -> Tensor:
    # Next Q value based on the max Q value of the target network
    next_Q_values = target_Q(s_next_batch).max(dim=1)[0]
    
    # If done flag is 1, next state doesn't exist so value is 0
    next_Q_values = next_Q_values * (1 - done_batch)
    return r_batch + (gamma * next_Q_values)

@torch.no_grad()
def ddqn_target(Q: nn.Module, target_Q: nn.Module, r_batch: Tensor, 
                s_next_batch: Tensor, done_batch: Tensor, gamma: float) -> Tensor:
    # Next action is selected using the BEHAVIOR network (Q)
    best_actions = Q(s_next_batch).argmax(dim=1)
    
    # The value of that action is evaluated using the TARGET network (target_Q)
    next_Q_values = target_Q(s_next_batch).gather(1, best_actions.unsqueeze(1)).squeeze(1)
    
    # Zero out terminal states
    next_Q_values = next_Q_values * (1 - done_batch)
    return r_batch + (gamma * next_Q_values)


# ---------------------------------------------------------
# Learning Algorithm
# ---------------------------------------------------------
def learning(
    env: gym.Env,
    target_function: Callable,
    batch_size: int = 128,
    gamma: float = 0.99,
    replay_buffer_size=10000,
    num_episodes: int = 400,   # Reduced for experiment speed
    learning_starts: int = 1000,
    learning_freq: int = 4,
    target_update_freq: int = 100,
    log_every: int = 100,
    learning_rate: float = 1e-3, # Added explicitly for experiments
    eps_decay_steps: int = 10000 # Added explicitly for experiments
):
    input_arg = env.observation_space.shape[0]
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Q = DQN_RAM(input_arg, num_actions).to(device)
    target_Q = DQN_RAM(input_arg, num_actions).to(device)
    target_Q.load_state_dict(Q.state_dict()) # initialize target identically
      
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(size=replay_buffer_size)
    
    # Init epsilon as instructed (init=0.9, end=0.05, nr_iter)
    eps_scheduler = iter(eps_generator(max_eps=0.9, min_eps=0.05, max_iter=eps_decay_steps))
    
    all_episode_rewards = []
    total_steps = 0
    num_param_updates = 0
    
    for episode in range(1, num_episodes + 1):
        s, _ = env.reset()
        episode_reward = 0
        
        for _ in count():
            total_steps += 1
            
            if total_steps > learning_starts:
                eps = next(eps_scheduler)
                s_tensor = torch.tensor(s).view(1, -1).float().to(device)
                a = select_epilson_greedy_action(Q, s_tensor, eps)
            else:
                a = np.random.choice(np.arange(num_actions))

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            episode_reward += r
            replay_buffer.store(s, a, r, s_next, done)

            if done:
                break
                
            s = s_next

            if (total_steps > learning_starts and total_steps % learning_freq == 0):
                for _ in range(learning_freq):
                    s_batch, a_batch, r_batch, s_next_batch, done_batch = replay_buffer.sample(batch_size)

                    s_batch      = s_batch.float().to(device)
                    a_batch      = a_batch.long().to(device)
                    r_batch      = r_batch.float().to(device)
                    s_next_batch = s_next_batch.float().to(device)
                    done_batch   = done_batch.long().to(device)

                    Q_values = Q(s_batch).gather(1, a_batch.unsqueeze(1)).view(-1)
                    target_Q_values = target_function(Q, target_Q, r_batch, s_next_batch, done_batch, gamma)

                    loss = criterion(target_Q_values, Q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    num_param_updates += 1

                    if num_param_updates % target_update_freq == 0:
                        target_Q.load_state_dict(Q.state_dict())

        all_episode_rewards.append(episode_reward)
        
        # Suppress printing to keep console clean during large experiment loops
        # if episode % log_every == 0 and total_steps > learning_starts:
        #     mean_episode_reward = np.mean(all_episode_rewards[-100:])
        #     print("Episode: %d, Mean reward: %.2f" % (episode, mean_episode_reward))

    return all_episode_rewards

# Helper to smooth curves
def moving_average_with_variance(data, window_size=50):
    if len(data) < window_size:
        return [], [], []
    indices = np.arange(window_size - 1, len(data))
    means = []
    upper_bounds = []
    lower_bounds = []
    for i in range(window_size - 1, len(data)):
        window = data[i - window_size + 1 : i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        means.append(mean_val)
        upper_bounds.append(mean_val + std_val)
        lower_bounds.append(mean_val - std_val)
    return indices, means, [lower_bounds, upper_bounds]

# ---------------------------------------------------------
# Experiment Execution & Plotting
# ---------------------------------------------------------
def run_and_plot_experiment(experiment_name, param_name, param_values, baseline_kwargs):
    print(f"\n--- Running Experiment: {experiment_name} ---")
    
    fig, axes = plt.subplots(1, len(param_values), figsize=(18, 5), sharey=True)
    fig.suptitle(f"{experiment_name} - DQN vs DDQN", fontsize=16)
    
    env = gym.make("CartPole-v1", max_episode_steps=100)
    
    for idx, val in enumerate(param_values):
        print(f"Testing {param_name} = {val}...")
        
        # Update kwargs for this iteration
        kwargs = baseline_kwargs.copy()
        kwargs[param_name] = val
        
        # Run DQN
        dqn_rewards = learning(env=env, target_function=dqn_target, **kwargs)
        # Run DDQN
        ddqn_rewards = learning(env=env, target_function=ddqn_target, **kwargs)
        
        # Compute smooth plots
        ax = axes[idx]
        x_dqn, dqn_mean, dqn_var = moving_average_with_variance(np.array(dqn_rewards), window_size=50)
        ax.plot(x_dqn, dqn_mean, color='blue', label='DQN')
        ax.fill_between(x_dqn, dqn_var[0], dqn_var[1], alpha=0.15, color='blue')
        
        x_ddqn, ddqn_mean, ddqn_var = moving_average_with_variance(np.array(ddqn_rewards), window_size=50)
        ax.plot(x_ddqn, ddqn_mean, color='orange', label='DDQN')
        ax.fill_between(x_ddqn, ddqn_var[0], ddqn_var[1], alpha=0.15, color='orange')
        
        ax.set_title(f"{param_name} = {val}")
        ax.set_xlabel('Episode')
        if idx == 0:
            ax.set_ylabel('Total Reward (Moving Avg)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Baseline configuration across all experiments
    baseline_kwargs = {
        'num_episodes': 300,        # Set to 300-500 to keep reasonable compute time
        'learning_rate': 1e-3, 
        'target_update_freq': 100, 
        'learning_freq': 4, 
        'eps_decay_steps': 10000 
    }

    # Experiment 1: SGD Learning Rate
    run_and_plot_experiment(
        experiment_name="Experiment 1: SGD Learning Rate",
        param_name="learning_rate",
        param_values=[1e-2, 1e-3, 1e-4],
        baseline_kwargs=baseline_kwargs
    )

    # Experiment 2: Target Update Frequency
    run_and_plot_experiment(
        experiment_name="Experiment 2: Target Update Frequency",
        param_name="target_update_freq",
        param_values=[10, 100, 200],
        baseline_kwargs=baseline_kwargs
    )

    # Experiment 3: Learning Frequency
    run_and_plot_experiment(
        experiment_name="Experiment 3: Learning Frequency",
        param_name="learning_freq",
        param_values=[1, 4, 8],
        baseline_kwargs=baseline_kwargs
    )

    # Experiment 4: Epsilon Decay Rates (nr_iterations)
    run_and_plot_experiment(
        experiment_name="Experiment 4: Epsilon Decay Rates",
        param_name="eps_decay_steps",
        param_values=[5000, 10000, 20000],
        baseline_kwargs=baseline_kwargs
    )