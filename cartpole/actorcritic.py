import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# Configuration parameters
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()

num_inputs = 4
num_actions = 2
num_hidden = 128

# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(ActorCritic, self).__init__()
        self.common = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        common = self.relu(self.common(x))
        action_probs = torch.softmax(self.actor(common), dim=-1)
        critic_value = self.critic(common)
        return action_probs, critic_value

model = ActorCritic(num_inputs, num_hidden, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
huber_loss = nn.HuberLoss(reduction='mean')

# Training state
action_probs_history = []
critic_value_history = []
action_losses_history = []
critic_losses_history = []
losses_history = []
rewards_history = []
running_reward = 0
episode_count = 0

env = gym.make("CartPole-v1")

def play():
    """Run agent in environment with rendering"""
    env = gym.make("CartPole-v1", render_mode='human')
    state, _ = env.reset()
    for time in range(500):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
        state, reward, terminated, _, _ = env.step(action)
        if terminated:
            print(f"Score: {time}")
            break

while True:  # Run until solved
    state, _ = env.reset()
    episode_reward = 0
    action_probs_history.clear()
    critic_value_history.clear()
    rewards_history.clear()
    
    for timestep in range(1, max_steps_per_episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Forward pass
        action_probs, critic_value = model(state_tensor)
        critic_value_history.append(critic_value[0, 0].detach())
        
        # Sample action
        dist = Categorical(action_probs)
        action = dist.sample().item()
        action_probs_history.append(dist.log_prob(torch.tensor(action).to(device)))
        
        # Environment step
        state, reward, terminated, _, _ = env.step(action)
        rewards_history.append(reward)
        episode_reward += reward
        
        if terminated:
            break
    
    # Update running reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    
    # Calculate returns
    returns = []
    discounted_sum = 0
    for r in reversed(rewards_history):
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    
    # Normalize returns
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    # Calculate losses
    actor_losses = []
    critic_losses = []
    
    critic_value_history = torch.stack(critic_value_history)
    
    for log_prob, value, ret in zip(action_probs_history, critic_value_history, returns):
        advantage = ret - value.item()
        actor_losses.append(-log_prob * advantage)
        critic_losses.append(huber_loss(value.unsqueeze(0), ret.unsqueeze(0)))
    
    # Visualization every episode
    if episode_count % 1 == 0:
        steps = range(len(action_probs_history))
        plt.style.use('dark_background')
        
        # Critic plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, [v.item() for v in critic_value_history], label="Belohnungsprognose Critic")
        plt.plot(steps, returns.cpu().numpy(), label="Gewichtete Zukunftsbelohnung")
        plt.plot(steps, [c.item() for c in critic_losses], label="Verlust Critic")
        plt.title("Belohnungsprognose Critic")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"critic_cartpole_e{episode_count}.svg")
        plt.close()
        
        # Actor plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, [p.item() for p in action_probs_history], label="Entscheidungssicherheit Actor")
        plt.plot(steps, [a.item() for a in actor_losses], label="Verlust Actor")
        plt.title("Entscheidungssicherheit Actor")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"actor_cartpole_e{episode_count}.svg")
        plt.close()
    
    # Backpropagation
    actor_losses_sum = sum(actor_losses)
    critic_losses_sum = sum(critic_losses)
    total_loss = actor_losses_sum + critic_losses_sum
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Save loss history
    losses_history.append((actor_losses_sum.item(), critic_losses_sum.item()))
    action_losses_history.append(actor_losses_sum.item())
    critic_losses_history.append(critic_losses_sum.item())
    
    episode_count += 1
    
    if episode_count % 50 == 0:
        print(f"Episode {episode_count}: Running reward: {running_reward:.2f}")
    
    if running_reward >= 195:
        print(f"Solved after {episode_count} episodes!")
        # Save loss plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(action_losses_history)), action_losses_history, label="Actor Loss")
        plt.plot(range(len(critic_losses_history)), critic_losses_history, label="Critic Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig("losses_cartpole.svg")
        plt.show()
        break
