# main.py - DQN Version
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import argparse
import random
from sklearn.model_selection import train_test_split

# 导入我们自己的模块
from utils import build_state_features, discretize_actions, federated_kmeans
from evaluation import ExperimentLogger
from dqn_network import DQN, device
from dqn_trainer import train_dqn
from federated_learning import Client, Server

# ============================================================
# 全局模拟参数
# ============================================================
N_STATES = 750
N_BINS_ACTION = 5
N_ACTIONS = N_BINS_ACTION**2
DISCOUNT_FACTOR = 0.99
DATA_FILE_PATH = "Dataset/MIMICtable.csv"
COL_BIN = ['gender', 'mechvent', 're_admission']
COL_NORM = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1',
            'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb', 'WBC_count',
            'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3',
            'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance']
COL_LOG = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR', 'input_total',
           'input_4hourly', 'output_total', 'output_4hourly']

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_policy_dqn(policy_net, test_data, discount):
    """
    Evaluates a DQN policy on the test set using the conservative matching method.
    """
    policy_net.eval()  # Set the network to evaluation mode
    
    if test_data.empty:
        return 0.0
        
    # Generate the policy (action for each state) from the DQN
    with torch.no_grad():
        all_states = torch.arange(N_STATES, device=device)
        q_values = policy_net(all_states)
        policy = torch.argmax(q_values, axis=1).cpu().numpy()

    total_discounted_reward = 0
    num_episodes = test_data['icustayid'].nunique()

    if num_episodes == 0:
        return 0.0

    for _, episode in test_data.groupby('icustayid'):
        episode_reward = 0
        sorted_episode = episode.sort_values(by='bloc')
        
        current_discount = 1.0
        for _, row in sorted_episode.iterrows():
            state = int(row['state'])
            chosen_a = policy[state]
            
            # Use logged reward if action matches, otherwise a small penalty
            reward = row['reward'] if chosen_a == int(row['action']) else -1
            
            episode_reward += current_discount * reward
            current_discount *= discount
            
        total_discounted_reward += episode_reward
        
    return total_discounted_reward / num_episodes

def generate_and_load_data():
    """
    Checks if the data file exists, generates mock data if not, then loads it.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Data file '{DATA_FILE_PATH}' not found, generating mock data...")
        # (The mock data generation script logic is omitted here for brevity,
        # but it's the same as in your previous version)
        import generate_mock_data
        generate_mock_data.main()
    
    print(f"Loading data from '{DATA_FILE_PATH}'...")
    df = pd.read_csv(DATA_FILE_PATH)
    return df

def preprocess_data(df, seed=42):
    """
    Preprocesses the entire dataset: builds features, discretizes states and actions.
    """
    print("\nStep 1: Building global state features...")
    features, _, _ = build_state_features(df, COL_BIN, COL_NORM, COL_LOG)

    print("Step 2: Performing K-Means to define global state space...")
    np.random.seed(seed)
    sample_for_kmeans = features[np.random.choice(features.shape[0], size=min(10000, features.shape[0]), replace=False)]
    kmeans_model = federated_kmeans([sample_for_kmeans], n_states=N_STATES)
    df['state'] = kmeans_model.predict(features)
    print(f"Mapped all data points to one of {N_STATES} states.")

    print("\nStep 3: Discretizing action space...")
    df['action'] = discretize_actions(df['input_4hourly'].values, df['max_dose_vaso'].values, n_bins=N_BINS_ACTION)
    print(f"Mapped continuous doses to one of {N_ACTIONS} discrete actions.")
    
    print("\nStep 4: Defining reward function (AI Clinician style)...")
    df = df.sort_values(['icustayid', 'bloc']).reset_index(drop=True)
    df['delta_sofa'] = df.groupby('icustayid')['SOFA'].shift(-1) - df['SOFA']
    df['delta_sofa'].fillna(0, inplace=True)

    df['r_delta_sofa'] = -20 * df['delta_sofa']
    df['r_action_pen'] = -0.001 * df['action']
    terminal_mask = df.groupby('icustayid')['bloc'].transform('max') == df['bloc']
    terminal_bonus = df['died_in_hosp'].apply(lambda x: -15000 if x == 1 else 15000)
    df['r_terminal'] = terminal_mask * terminal_bonus
    
    df['reward'] = df['r_delta_sofa'] + df['r_action_pen'] + df['r_terminal']
    
    return df

def run_federated(args, train_data, test_data):
    print("\n================== Starting Federated DQN Experiment ==================")
    logger = ExperimentLogger(log_dir=os.path.join('logs', 'federated'))
    logger.log_config(vars(args))

    print(f"\nCreating {args.n_clients} clients...")
    clients = []
    client_data_indices = np.array_split(train_data.index, args.n_clients)
    
    for i in range(args.n_clients):
        client_df = train_data.loc[client_data_indices[i]].copy()
        client = Client(client_id=i, local_data=client_df, n_states=N_STATES, n_actions=N_ACTIONS)
        clients.append(client)

    server = Server(clients, n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT_FACTOR)

    for round_num in range(1, args.rounds + 1):
        print(f"\n=============== Federated Round {round_num}/{args.rounds} ===============")
        server.train_one_round()
        
        global_net = server.get_global_net()
        global_reward = evaluate_policy_dqn(global_net, test_data, DISCOUNT_FACTOR)
        
        client_losses = server.get_client_losses() 