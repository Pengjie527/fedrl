# This file will be the main entry point to run the federated reinforcement learning simulation. 
# E:\AFLearning\FlerningwithRL\main.py
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
# MDP/RL 相关
# 为了让 Q-learning 更快收敛，我们先把状态／动作空间缩小
N_STATES = 750  # 对标 AI-Clinician 的高粒度状态数
N_BINS_ACTION = 5  # 恢复 5 档位 ⇒ 共 25 个动作
N_ACTIONS = N_BINS_ACTION**2
DISCOUNT_FACTOR = 0.99 # 折扣因子

# 数据相关
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

def generate_and_load_data():
    """
    检查数据文件是否存在，如果不存在，则调用脚本生成模拟数据。
    然后加载数据。
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"数据文件 '{DATA_FILE_PATH}' 不存在，正在生成模拟数据...")
        try:
            import generate_mock_data
            generate_mock_data.main()
            print("模拟数据生成完毕。")
        except (ImportError, FileNotFoundError):
            print("\n错误: 'generate_mock_data.py' 未找到。正在尝试创建...")

            
    print(f"正在从 '{DATA_FILE_PATH}' 加载数据...")
    df = pd.read_csv(DATA_FILE_PATH)
    return df

def preprocess_data(df, seed=42):
    """
    对整个数据集进行预处理：构建特征、离散化状态和动作。
    """
    print("\n步骤1: 构建全局状态特征...")
    features, _, _ = build_state_features(df, COL_BIN, COL_NORM, COL_LOG)

    print("步骤2: 执行K-Means以定义全局状态空间...")
    np.random.seed(seed)
    sample_for_kmeans = features[np.random.choice(features.shape[0], size=min(10000, features.shape[0]), replace=False)]
    best_inertia = np.inf
    for rs in range(5):
        km = federated_kmeans([sample_for_kmeans], n_states=N_STATES)
        if km.inertia_ < best_inertia:
            best_kmeans = km; best_inertia = km.inertia_
    kmeans_model = best_kmeans
    df['state'] = kmeans_model.predict(features)
    print(f"已将所有数据点映射到 {N_STATES} 个状态中的一个。")

    print("\n步骤3: 离散化动作空间...")
    df['action'] = discretize_actions(df['input_4hourly'].values, df['max_dose_vaso'].values, n_bins=N_BINS_ACTION)
    print(f"已将连续剂量映射到 {N_ACTIONS} 个离散动作中的一个。")
    
    print("\n步骤4: 定义奖励函数 (即时 SOFA 变化 + 动作惩罚 + 终末生存奖励)...")

    # 按 (icustayid, bloc) 排序，计算下一时刻 SOFA
    df = df.sort_values(['icustayid', 'bloc']).reset_index(drop=True)
    df['SOFA_next'] = df.groupby('icustayid')['SOFA'].shift(-1)
    # SOFA 变化（下降为负数，表示病情好转）
    df['delta_sofa'] = df['SOFA_next'] - df['SOFA']
    df['delta_sofa'].fillna(0, inplace=True)

    # 新奖励函数：含 SOFA 变化、SOFA 绝对水平、动作惩罚与缩小的终末奖励
    df['r_delta_sofa'] = -20 * df['delta_sofa']            # 放大 SOFA 变化信号
    df['r_action_pen'] = -0.001 * df['action']             # 更小剂量惩罚
    reward = df['r_delta_sofa'] + df['r_action_pen']

    # Episode 最后一步追加生存/死亡终末奖励 (±20)
    terminal_mask = df.groupby('icustayid')['bloc'].transform('max') == df['bloc']
    terminal_bonus = df['died_in_hosp'].apply(lambda x: -15000 if x == 1 else 15000)
    df['r_terminal'] = terminal_mask * terminal_bonus
    reward += df['r_terminal']

    df['reward'] = df['r_delta_sofa'] + df['r_action_pen'] + df['r_terminal']
    
    # 对奖励进行标准化，以稳定DQN训练
    reward_mean = df['reward'].mean()
    reward_std = df['reward'].std()
    df['reward'] = (df['reward'] - reward_mean) / (reward_std + 1e-8)
    print(f"Reward column has been standardized (mean={reward_mean:.2f}, std={reward_std:.2f}).")
    
    return df

def evaluate_policy_dqn(policy_net, test_data, discount):
    """
    Evaluates a DQN policy on the test set.
    """
    policy_net.eval() # Set the network to evaluation mode
    
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

def run_federated(args, train_data, test_data):
    """
    Run the federated learning experiment with DQN.
    """
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
        avg_loss = np.mean(list(client_losses.values()))

        # For simplicity, we evaluate each client's final model on the global test set
        client_rewards = {}
        for client in clients:
            client_net = client.get_policy_net()
            client_rewards[client.client_id] = evaluate_policy_dqn(client_net, test_data, DISCOUNT_FACTOR)

        logger.log_round_data(round_num, global_reward, client_rewards, loss=avg_loss)

    logger.save_results()
    print("================== Federated DQN Experiment Finished ==================")

def run_centralized(args, train_data, test_data):
    """Run the centralized DQN experiment."""
    print("\n================== Starting Centralized DQN Experiment ==================")
    logger = ExperimentLogger(log_dir=os.path.join('logs', 'centralized'))
    logger.log_config(vars(args))

    policy_net = DQN(N_STATES, N_ACTIONS).to(device)
    target_net = DQN(N_STATES, N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n--- Centralized Round {round_num}/{args.rounds}: Training DQN ---")
        
        avg_loss = train_dqn(
            data=train_data.copy(),
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT_FACTOR,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            iterations=10000 # More iterations for centralized
        )

        global_reward = evaluate_policy_dqn(policy_net, test_data, DISCOUNT_FACTOR)
        logger.log_round_data(round_num, global_reward, {}, loss=avg_loss)

    logger.save_results()
    print("================== Centralized DQN Experiment Finished ==================")


def run_local_only(args, train_data, test_data):
    """
    Runs the local-only DQN experiment.
    """
    print("\n================== Starting Local-Only DQN Experiment ==================")
    logger = ExperimentLogger(log_dir=os.path.join('logs', 'local_only'))
    logger.log_config(vars(args))

    client_data_indices = np.array_split(train_data.index, args.n_clients)
    
    all_client_rewards = []
    all_client_losses = []

    for i in range(args.n_clients):
        print(f"\n--- Training Client {i} ---")
        client_df = train_data.loc[client_data_indices[i]].copy()
        
        # Each client is a full DQN training setup
        policy_net = DQN(N_STATES, N_ACTIONS).to(device)
        target_net = DQN(N_STATES, N_ACTIONS).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

        # Train for a fixed number of iterations, e.g., total rounds * local iterations
        total_iterations = args.rounds * 10000
        
        avg_loss = train_dqn(
            data=client_df,
            n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT_FACTOR,
            policy_net=policy_net, target_net=target_net, optimizer=optimizer,
            iterations=total_iterations
        )
        
        reward = evaluate_policy_dqn(policy_net, test_data, DISCOUNT_FACTOR)
        all_client_rewards.append(reward)
        all_client_losses.append(avg_loss)
        print(f"--- Client {i} finished. Reward: {reward:.4f}, Avg Loss: {avg_loss:.4f} ---")
    
    # Log final results (as if it's one round)
    avg_reward = np.mean(all_client_rewards) if all_client_rewards else 0
    avg_loss = np.mean(all_client_losses) if all_client_losses else 0
    client_rewards_dict = {i: r for i, r in enumerate(all_client_rewards)}
    
    # Log data for each "round" to make the plot comparable
    for round_num in range(1, args.rounds + 1):
        logger.log_round_data(round_num, avg_reward, client_rewards_dict, loss=avg_loss)

    logger.save_results()
    print("================== Local-Only DQN Experiment Finished ==================")

def main():
    """
    主函数，驱动整个联邦强化学习模拟流程。
    """
    parser = argparse.ArgumentParser(description="运行联邦/集中式/本地强化学习实验")
    parser.add_argument('--mode', type=str, default='federated', choices=['federated', 'centralized', 'local_only'],
                        help='选择实验模式')
    parser.add_argument('--n_clients', type=int, default=3, help='客户端数量 (仅用于 federated 和 local_only 模式)')
    parser.add_argument('--rounds', type=int, default=10, help='联邦学习或模拟的轮次')
    parser.add_argument('--test_size', type=float, default=0.2, help='用于全局测试集的数据比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证可复现')
    
    args = parser.parse_args()
    
    # 0. 设定统一随机种子
    set_global_seed(args.seed)

    # 1. 加载和预处理数据
    raw_data = generate_and_load_data()
    processed_data = preprocess_data(raw_data, args.seed)

    # 2. 划分训练集和测试集
    #    为保证可复现性和公平性，我们基于 'icustayid' 进行划分
    all_patient_ids = processed_data['icustayid'].unique()
    train_ids, test_ids = train_test_split(all_patient_ids, test_size=args.test_size, random_state=42)
    
    train_data = processed_data[processed_data['icustayid'].isin(train_ids)].copy()
    test_data = processed_data[processed_data['icustayid'].isin(test_ids)].copy()
    
    print(f"\n数据划分完毕: {len(train_data)} 条训练数据, {len(test_data)} 条测试数据。")

    # 3. 根据模式运行实验
    if args.mode == 'federated':
        run_federated(args, train_data, test_data)
    elif args.mode == 'centralized':
        run_centralized(args, train_data, test_data)
    elif args.mode == 'local_only':
        run_local_only(args, train_data, test_data)
if __name__ == '__main__':
    main()

