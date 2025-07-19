import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """
    Deep Q-Network with an Embedding layer for discrete states.
    """
    def __init__(self, n_states=750, n_actions=25, embedding_dim=64, hidden_dim=256):
        """
        Initializes the DQN network.
        Args:
            n_states (int): The number of discrete states (for the embedding layer).
            n_actions (int): The number of possible actions.
            embedding_dim (int): The size of the state embedding vector.
            hidden_dim (int): The size of the hidden layers.
        """
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(n_states, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        """
        Forward pass through the network.
        Args:
            state (torch.Tensor): A tensor of state indices.
        
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        # Ensure state is a LongTensor for the embedding layer
        if not isinstance(state, torch.Tensor):
            state = torch.LongTensor(state).to(device)
        # Handle single state integer input
        if state.ndim == 0:
            state = state.unsqueeze(0)
        
        # Pass state indices through embedding layer
        x = F.relu(self.embedding(state))
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer
        return self.fc3(x) 