import torch
import torch.optim as optim
from collections import OrderedDict

# 导入我们自己的模块
from dqn_network import DQN, device
from dqn_trainer import train_dqn

# ============================================================
# 客户端（医院节点）定义 - DQN 版本
# ============================================================
class Client:
    """
    代表一个客户端，例如一家医院。
    它拥有本地的病人数据，并负责使用DQN进行本地训练。
    """
    def __init__(self, client_id, local_data, n_states, n_actions):
        self.client_id = client_id
        self.data = local_data
        
        # 每个客户端现在拥有自己的DQN网络和优化器
        self.policy_net = DQN(n_states, n_actions).to(device)
        self.target_net = DQN(n_states, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

    def local_train(self, global_weights, n_states, n_actions, discount, local_iterations=2000):
        """
        使用全局模型权重作为起点，在本地数据上进行训练。

        Args:
            global_weights (OrderedDict): 从服务器接收的全局模型权重。
            ... (其他参数)
            local_iterations (int): 本地训练的迭代次数。默认为2000。
        
        Returns:
            Tuple[OrderedDict, float]: 本地训练后的模型权重和平均损失。
        """
        # 加载全局权重
        self.policy_net.load_state_dict(global_weights)
        self.target_net.load_state_dict(global_weights) # 保持同步

        print(f"  [Client {self.client_id}] Starting local DQN training...")
        
        avg_loss = train_dqn(
            data=self.data.copy(),
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            policy_net=self.policy_net,
            target_net=self.target_net,
            optimizer=self.optimizer,
            iterations=local_iterations
        )
        
        print(f"  [Client {self.client_id}] Local training finished. Average loss: {avg_loss:.4f}")
        return self.policy_net.state_dict(), avg_loss

    def get_policy_net(self):
        return self.policy_net

# ============================================================
# 中央服务器定义 - DQN 版本
# ============================================================
class Server:
    """
    代表中央服务器。
    负责协调整个联邦学习过程，包括分发和聚合DQN模型权重。
    """
    def __init__(self, clients, n_states, n_actions, discount=0.99):
        self.clients = clients
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        
        # 全局模型现在是一个DQN网络
        self.global_net = DQN(n_states, n_actions).to(device)
        self.client_losses = {} # 记录本轮客户端的loss

    def train_one_round(self):
        """
        执行一轮联邦学习训练。
        """
        local_weights = []
        client_data_sizes = []
        
        # 1. 分发全局模型权重并开始客户端训练
        print("--- Distributing global model and starting client training ---")
        global_state_dict = self.global_net.state_dict()
        
        for client in self.clients:
            updated_weights, avg_loss = client.local_train(
                global_weights=global_state_dict,
                n_states=self.n_states,
                n_actions=self.n_actions,
                discount=self.discount
            )
            local_weights.append(updated_weights)
            client_data_sizes.append(len(client.data))
            self.client_losses[client.client_id] = avg_loss
            
        # 2. 聚合更新 (Federated Averaging of model weights)
        print("\n--- Aggregating client models (FedAvg) ---")
        
        if local_weights:
            total_data_size = sum(client_data_sizes)
            weights = [size / total_data_size for size in client_data_sizes]
            
            # 使用加权平均聚合权重
            avg_state_dict = OrderedDict()
            for key in global_state_dict.keys():
                avg_state_dict[key] = torch.sum(torch.stack([w[key] * weight for w, weight in zip(local_weights, weights)]), dim=0)
            
            self.global_net.load_state_dict(avg_state_dict)
            print("Global model has been updated.")
        else:
            print("Warning: No local models were returned to aggregate.")

        print("\n--- Federal Round Completed ---")

    def get_global_net(self):
        return self.global_net
        
    def get_client_losses(self):
        return self.client_losses
