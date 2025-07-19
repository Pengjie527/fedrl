# E:\AFLearning\FlerningwithRL\rl_core.py
import numpy as np

# ============================================================
# 强化学习核心：策略迭代
# ============================================================

def policy_iteration(T, R, discount=0.99, max_iter=100):
    """
    使用策略迭代算法求解马尔可夫决策过程（MDP）。

    此函数通过交替进行策略评估和策略改进来找到最优策略。

    参数:
    T : numpy.ndarray
        转移概率张量，维度为 (S, A, S')，其中 S 是状态数，A 是动作数。
        T[s, a, s_prime] 是在状态 s 执行动作 a 后转移到状态 s_prime 的概率。
    R : numpy.ndarray
        奖励函数，维度为 (S, A)。
        R[s, a] 是在状态 s 执行动作 a 后获得的即时奖励。
    discount : float, optional
        折扣因子 (gamma)，介于 0 和 1 之间，默认为 0.99。
    max_iter : int, optional
        最大迭代次数，默认为 100。

    返回:
    policy : numpy.ndarray
        最优策略，一个长度为 S 的一维数组，policy[s] 表示在状态 s 下的最优动作。
    V : numpy.ndarray
        最优值函数，一个长度为 S 的一维数组，V[s] 表示在状态 s 的期望回报。
    Q : numpy.ndarray
        最优动作值函数 (Q-function)，维度为 (S, A)。
        Q[s, a] 表示在状态 s 执行动作 a 后的期望回报。
    """
    n_states, n_actions, _ = T.shape
    
    # 1. 初始化一个随机策略
    policy = np.random.randint(0, n_actions, size=n_states)
    
    for i in range(max_iter):
        policy_stable = True
        
        # 2. 策略评估 (Policy Evaluation)
        #    对于给定的策略 π，计算其值函数 V_π
        #    这通过解贝尔曼期望方程的线性系统来实现: V_π = R_π + γ * P_π * V_π
        
        # 获取当前策略下的转移概率矩阵 P_π(s, s') 和奖励向量 R_π(s)
        P_pi = np.array([T[s, policy[s], :] for s in range(n_states)])
        R_pi = np.array([R[s, policy[s]] for s in range(n_states)])
        
        # 解线性方程组 (I - γ * P_π) * V_π = R_π
        try:
            V = np.linalg.solve(np.eye(n_states) - discount * P_pi, R_pi)
        except np.linalg.LinAlgError:
            # 如果矩阵是奇异的（通常因为某些状态不可达），使用伪逆
            V = np.linalg.pinv(np.eye(n_states) - discount * P_pi) @ R_pi

        # 3. 策略改进 (Policy Improvement)
        #    基于当前的值函数 V，贪婪地选择能最大化期望回报的动作，以改进策略。
        
        # 首先计算动作值函数 Q(s, a) = R(s, a) + γ * Σ_s' P(s, s', a) * V(s')
        # 使用 einsum 进行高效的张量乘法和求和
        Q = R + discount * np.einsum('ijk,k->ij', T, V)
        
        # 找到每个状态下最优的动作
        new_policy = np.argmax(Q, axis=1)
        
        # 4. 检查策略是否稳定
        #    如果新策略与旧策略完全相同，则说明已找到最优策略，可以提前终止。
        if not np.array_equal(policy, new_policy):
            policy_stable = False
            policy = new_policy
        
        if policy_stable:
            print(f"  [RL Core] Policy stabilized after {i+1} iterations.")
            break
            
    return policy, V, Q

def train_rl_policy(data, n_states, n_actions, discount, learning_rate=0.5, iterations=10000, q_table_init=None):
    """
    使用 Q-learning 算法从离线数据中学习策略。这是一个无模型的算法。

    Args:
        data (pd.DataFrame): 包含 (state, action, reward, icustayid) 的数据。
        n_states (int): 状态总数。
        n_actions (int): 动作总数。
        discount (float): 折扣因子。
        learning_rate (float): 学习率 alpha。
        iterations (int): 训练迭代次数。
        q_table_init (np.ndarray, optional): 初始化的Q表，用于继续训练。默认为None，即从零开始。

    Returns:
        Q (np.ndarray): 学习到的 Q-table，维度 (S, A)。
        policy (np.ndarray): 从Q-table推导出的贪婪策略。
    """
    if q_table_init is not None:
        Q = q_table_init
    else:
        Q = np.zeros((n_states, n_actions))
    
    # 准备数据，我们需要 (s, a, r, s') 元组
    # 我们通过对数据按 icustayid 分组并移动 state 列来找到 next_state
    if 'next_state' not in data.columns:
        data['next_state'] = data.groupby('icustayid')['state'].shift(-1).fillna(-1).astype(int)
    
    # 筛选出有效的转移
    transitions = data[data['next_state'] != -1].copy()
    
    if len(transitions) == 0:
        print("[RL Core] 警告: 数据中没有有效的状态转移，无法训练Q-learning。")
        return Q, np.argmax(Q, axis=1)

    # 为了高效训练，我们将数据转换为numpy数组，并随机抽样
    s_a_r_s_tuples = transitions[['state', 'action', 'reward', 'next_state']].to_numpy().astype(int)

    print(f"  [RL Core] 开始 Q-learning 训练，共 {iterations} 次迭代 (含学习率衰减)…")
    for i in range(iterations):
        # 随机抽取一个经验元组 (s, a, r, s')
        s, a, r, s_prime = s_a_r_s_tuples[np.random.randint(len(s_a_r_s_tuples))]
        
        # 计算当前步的自适应学习率 alpha_t = alpha0 / (1 + decay * t)
        alpha_t = learning_rate / (1 + 1e-4 * i)

        # Q-learning 更新规则
        old_value = Q[s, a]
        future_optimal_value = np.max(Q[s_prime])

        # Bellman 方程
        new_value = old_value + alpha_t * (r + discount * future_optimal_value - old_value)
        Q[s, a] = new_value

    policy = np.argmax(Q, axis=1)
    print("  [RL Core] Q-learning 训练完成。")
    
    return Q, policy

def evaluate_policy(q_table, test_data, discount):
    """
    在测试集上评估一个策略（由Q-table代表）的性能。

    Args:
        q_table (np.ndarray): 待评估的Q-table。
        test_data (pd.DataFrame): 包含 (state, reward, icustayid) 的测试数据。
        discount (float): 折扣因子。

    Returns:
        float: 所有 episode 的平均累积折扣奖励。
    """
    if test_data.empty:
        return 0.0
        
    policy = np.argmax(q_table, axis=1)
    total_discounted_reward = 0
    num_episodes = test_data['icustayid'].nunique()

    if num_episodes == 0:
        return 0.0

    # 按 icustayid (episode) 遍历
    for _, episode in test_data.groupby('icustayid'):
        episode_reward = 0
        
        # 在一个 episode 内按时间步长（bloc）排序
        # 注意：在原始代码中，bloc似乎没有被用来排序，这里我们假设数据是按时间顺序的
        sorted_episode = episode.sort_values(by='bloc' if 'bloc' in episode.columns else 'icustayid')
        
        current_discount = 1.0
        for _, row in sorted_episode.iterrows():
            state = int(row['state'])
            # 如果策略选择的动作与日志中的动作一致，则使用日志奖励；否则给予轻微惩罚
            chosen_a = policy[state]
            if chosen_a == int(row['action']):
                reward = row['reward']
            else:
                reward = -1  # 动作不匹配时给予小负奖励
            
            # 累积折扣奖励
            episode_reward += current_discount * reward
            current_discount *= discount
            
        total_discounted_reward += episode_reward
        
    return total_discounted_reward / num_episodes
 