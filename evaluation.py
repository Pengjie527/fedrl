import os
import json
import pandas as pd
from datetime import datetime
# (绘图逻辑已迁移至 plot_results.py)

class ExperimentLogger:
    def __init__(self, log_dir='logs'):
        """
        初始化实验日志记录器。
        会创建一个以当前时间戳命名的唯一目录来保存本次实验的所有结果。
        """
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.log_path, exist_ok=True)
        
        self.round_data = []
        print(f"实验结果将保存在: {self.log_path}")

    def log_config(self, config):
        """
        保存实验的配置信息（例如，超参数）。
        
        Args:
            config (dict): 包含配置项的字典。
        """
        config_path = os.path.join(self.log_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def log_round_data(self, round_num, global_reward, client_rewards, **kwargs):
        """
        记录一个通信轮次的数据。
        
        Args:
            round_num (int): 当前的通信轮次。
            global_reward (float): 全局模型在测试集上的奖励。
            client_rewards (dict): 一个字典，key为client_id, value为该客户端模型的奖励。
            **kwargs: 其他需要记录的指标，例如 loss。
        """
        log_entry = {
            'round': round_num,
            'global_reward': global_reward,
            'avg_client_reward': sum(client_rewards.values()) / len(client_rewards) if client_rewards else 0,
            'min_client_reward': min(client_rewards.values()) if client_rewards else 0,
            'max_client_reward': max(client_rewards.values()) if client_rewards else 0,
        }
        # 添加每个客户端的奖励，方便后续分析公平性
        for client_id, reward in client_rewards.items():
            log_entry[f'client_{client_id}_reward'] = reward
            
        # 添加其他自定义指标
        log_entry.update(kwargs)
        
        self.round_data.append(log_entry)
        print(f"Round {round_num}: Global Reward = {global_reward:.4f}")

    def save_results(self):
        """
        将所有记录的实验数据保存到CSV文件中。
        """
        if not self.round_data:
            print("没有数据可以保存。")
            return
            
        results_path = os.path.join(self.log_path, 'metrics.csv')
        df = pd.DataFrame(self.round_data)
        df.to_csv(results_path, index=False)
        print(f"实验指标已保存到: {results_path}")

        # Reward 曲线绘图功能已迁移至 plot_results.py

    def get_log_path(self):
        """返回当前实验日志的保存路径。"""
        return self.log_path 