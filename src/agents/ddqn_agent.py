import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    """Deep Q-Network for V2I communication optimization"""
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        # Three-layer feedforward network
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    """Double Deep Q-Network Agent for V2I Base Station Selection"""
    
    def __init__(self, state_size=3, action_size=3, learning_rate=0.001):
        """
        Initialize DDQN Agent
        Args:
            state_size: Dimension of state space (reachable_rf, reachable_thz, ad_action)
            action_size: Number of actions (e.g., 3: stay, switch_rf, switch_thz)
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99          # Discount factor
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.memory_size = 10000
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DDQN using device: {self.device}")
        
        # Networks (Policy and Target)
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Training tracking
        self.update_target_every = 100
        self.steps = 0
        self.losses = []
        
        print(f"✓ DDQN Agent initialized")
    
    def state_to_tensor(self, state):
        """Convert state dictionary to tensor"""
        state_array = np.array([
            state['reachable_rf'],
            state['reachable_thz'],
            state['ad_action']
        ], dtype=np.float32)
        return torch.FloatTensor(state_array).to(self.device)
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        Args:
            state: Current V2I state
            training: If False, use greedy policy (no exploration)
        """
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Unpack batch
        states = torch.stack([self.state_to_tensor(s) for s, _, _, _, _ in batch])
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).unsqueeze(1).to(self.device)
        next_states = torch.stack([self.state_to_tensor(ns) for _, _, _, ns, _ in batch])
        dones = torch.FloatTensor([d for _, _, _, _, d in batch]).unsqueeze(1).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN: Select action with policy net, evaluate with target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"✓ Model loaded from {filepath}")
