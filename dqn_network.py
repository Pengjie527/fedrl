import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """
    Deep Q-Network with an Embedding layer for discrete states.
    """
    def __init__(self, n_states=750, n_actions=25, embedding_dim=64, hidden_dim=256):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(n_states, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state, device):
        """
        Forward pass through the network.
        Args:
            state (torch.Tensor or np.ndarray or int): A tensor of state indices.
            device (torch.device): The device to run the forward pass on.
        
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        # Ensure state is a LongTensor on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.LongTensor([state] if np.isscalar(state) else state).to(device)
        else:
            state = state.to(device) # Ensure tensor is on the correct device

        # Handle single state integer input by unsqueezing
        if state.ndim == 0:
            state = state.unsqueeze(0)
        
        # Pass state indices through embedding layer
        x = F.relu(self.embedding(state))
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer
        return self.fc3(x) 