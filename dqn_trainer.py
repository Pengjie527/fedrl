import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dqn_network import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dqn(
    data, 
    n_states, 
    n_actions, 
    discount,
    policy_net,
    target_net,
    optimizer,
    iterations=10000, 
    batch_size=256,
    target_update_freq=500, # This will be ignored, but kept for API compatibility
    tau=0.005, # Soft update parameter
    device
):
    """
    Trains a DQN model using the provided data.
    
    Args:
        data (pd.DataFrame): The training data.
        n_states (int): Number of states.
        n_actions (int): Number of actions.
        discount (float): Discount factor.
        policy_net (DQN): The main DQN model to be trained.
        target_net (DQN): The target network for stable Q-value estimation.
        optimizer (torch.optim.Optimizer): The optimizer for the policy network.
        iterations (int): Total number of training iterations.
        batch_size (int): The size of each training batch.
        target_update_freq (int): Frequency (in iterations) to update the target network.
        
    Returns:
        float: The average loss over the training iterations.
    """
    if 'next_state' not in data.columns:
        data['next_state'] = data.groupby('icustayid')['state'].shift(-1).fillna(-1).astype(int)
    
    transitions = data[data['next_state'] != -1].copy()
    
    if len(transitions) < batch_size:
        print(f"[Warning] Not enough transitions ({len(transitions)}) for batch size {batch_size}. Skipping training.")
        return 0.0

    states = torch.LongTensor(transitions['state'].values).to(device)
    actions = torch.LongTensor(transitions['action'].values).to(device)
    rewards = torch.FloatTensor(transitions['reward'].values).to(device)
    next_states = torch.LongTensor(transitions['next_state'].values).to(device)
    
    total_loss = 0
    
    for i in range(iterations):
        # Sample a random batch of transitions
        indices = torch.randint(0, len(transitions), (batch_size,))
        batch_states = states[indices]
        batch_actions = actions[indices]
        batch_rewards = rewards[indices]
        batch_next_states = next_states[indices]

        # Compute Q(s, a) - the model's prediction
        q_values = policy_net(batch_states, device).gather(1, batch_actions.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_q_values = target_net(batch_next_states, device).max(1)[0]
            # Compute the expected Q values
            expected_q_values = batch_rewards + (discount * next_q_values)

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()

        # Soft update the target network
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
            
    return total_loss / iterations if iterations > 0 else 0.0 