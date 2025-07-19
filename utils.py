import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# ============================================================
# 工具函数集合：特征处理、动作离散化、联邦 KMeans 聚类
# ============================================================

def discretize_actions(iv_fluid: np.ndarray, vaso: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """将连续的输液量(iv_fluid)和血管加压剂剂量(vaso)离散到 n_bins×n_bins 网格。
    返回值为整数动作编号，范围 0〜n_bins**2-1。
    参数说明：
    iv_fluid : 静脉输液量数组
    vaso     : 血管加压剂剂量数组
    n_bins   : 每个维度的离散桶数量，默认 5
    """
    # 仅用非零值计算分位点，避免 0 占比过高导致分箱失衡
    iv_positive = iv_fluid[iv_fluid > 0]
    vaso_positive = vaso[vaso > 0]

    # 计算分位点作为桶边界；若全为 0，则返回全 0 边界
    iv_edges = np.quantile(iv_positive, np.linspace(0, 1, n_bins + 1)[1:-1]) if iv_positive.size else np.zeros(n_bins - 1)
    vaso_edges = np.quantile(vaso_positive, np.linspace(0, 1, n_bins + 1)[1:-1]) if vaso_positive.size else np.zeros(n_bins - 1)

    def bin_single(x, edges):
        """给定边界列表，返回 x 所属的桶编号 (0~n_bins-1)。"""
        return np.searchsorted(edges, x, side="right")

    iv_bin = np.vectorize(lambda x: bin_single(x, iv_edges))(iv_fluid)
    vaso_bin = np.vectorize(lambda x: bin_single(x, vaso_edges))(vaso)

    # 组合成二维网格动作：行=vaso 桶，列=iv 桶
    action_id = vaso_bin * n_bins + iv_bin
    return action_id.astype(int)


def build_state_features(df: pd.DataFrame, bin_cols: list, norm_cols: list, log_cols: list):
    """按照 AI Clinician 逻辑构建状态特征矩阵。
    1) 二值特征减 0.5 使其居中；
    2) 连续特征做 Z-Score；
    3) 对数特征先 log(0.1+x) 再 Z-Score。
    返回：特征矩阵、连续特征标准化器、对数特征标准化器。
    """
    # 二值特征：0/1 -> -0.5 / 0.5
    bin_data = df[bin_cols].values - 0.5

    # 连续特征：标准化
    norm_data = df[norm_cols].values
    scaler_norm = StandardScaler().fit(norm_data)
    norm_z = scaler_norm.transform(norm_data)

    # 对数特征：log 变换后标准化
    log_data = np.log(0.1 + df[log_cols].values)
    scaler_log = StandardScaler().fit(log_data)
    log_z = scaler_log.transform(log_data)

    # 拼接所有特征
    feats = np.hstack([bin_data, norm_z, log_z])
    return feats, scaler_norm, scaler_log


def federated_kmeans(local_feature_sets: list, n_states: int = 750, n_init: int = 10, max_iter: int = 300):
    """极简版联邦 KMeans：各客户端先随机下采样一部分特征，
    服务器端简单拼接后跑一次集中式 KMeans。
    注意：真实生产环境应使用安全聚合或分布式 KMeans，此处仅做示范。
    参数：
    local_feature_sets : 各客户端上传的特征子样本列表
    n_states           : 聚类中心数量，即离散状态数
    n_init / max_iter  : sklearn KMeans 参数
    """
    # 拼接各节点上传的特征子集
    concat_feats = np.vstack(local_feature_sets)

    # 运行 MiniBatchKMeans 以提升速度和扩展性
    kmeans = MiniBatchKMeans(n_clusters=n_states, batch_size=500, n_init=n_init,
                             max_iter=max_iter, random_state=0)
    kmeans.fit(concat_feats)
    return kmeans