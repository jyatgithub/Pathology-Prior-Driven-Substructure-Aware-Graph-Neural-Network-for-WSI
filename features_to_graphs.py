

import numpy as np
import os
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch_geometric.data import Data
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree as sKDTree
from tqdm import tqdm
import pickle
import networkx as nx
from torch_geometric.utils import to_networkx

USE_CUDA = torch.cuda.is_available()


def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v


def toTensor(v, dtype=torch.float, requires_grad=True):
    device = 'cuda:0'
    return (Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad)).to(device)


def connectClusters(Cc, dthresh=3000):
    tess = Delaunay(Cc)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    nx_neighbors = neighbors
    W = np.zeros((Cc.shape[0], Cc.shape[0]))
    for n in nx_neighbors:
        nx_neighbors[n] = np.array(list(nx_neighbors[n]), dtype=int)
        nx_neighbors[n] = nx_neighbors[n][
            KDTree(Cc[nx_neighbors[n], :]).query_ball_point(Cc[n], r=dthresh)
        ]
        W[n, nx_neighbors[n]] = 1.0
        W[nx_neighbors[n], n] = 1.0
    return W


def toGeometric(X, W, y, tt=0):
    return Data(
        x=toTensor(X, requires_grad=False),
        edge_index=(toTensor(W, requires_grad=False) > tt).nonzero().t().contiguous(),
        y=toTensor([y], dtype=torch.long, requires_grad=False)
    )


# ============================================================
# 新增：病理先验驱动的子结构特征计算
# ============================================================

def compute_substructure_features(G_pyg):
    """
    对PyG图的每个节点计算4类病理先验子结构特征：

    feat0 - 三角形参与数：对应肿瘤细胞局部聚集密度
    feat1 - 局部聚集系数：对应邻域整体密集程度（转移灶特征）
    feat2 - 4阶环参与数：对应细胞巢/腺管环形排列结构
    feat3 - 度中心性：对应节点在局部拓扑中的角色（边界 vs 中心）

    返回: np.ndarray, shape (N, 4)，已做min-max归一化
    """
    # 转成networkx无向图
    # 注意：这里的G_pyg还没有移到GPU，coords和edge_index都在CPU
    edge_index_np = G_pyg.edge_index.cpu().numpy()
    num_nodes = G_pyg.x.shape[0]

    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index_np[0], edge_index_np[1]))
    G_nx.add_edges_from(edges)

    sub_feats = np.zeros((num_nodes, 4))

    # ---- feat0：三角形参与数 ----
    # nx.triangles返回每个节点参与的三角形数量
    # 病理含义：高三角形密度 → 肿瘤细胞团紧密聚集
    triangles = nx.triangles(G_nx)
    for node, count in triangles.items():
        sub_feats[node, 0] = count


    # ---- feat1：局部聚集系数 ----
    # 衡量邻居节点之间互相连接的程度
    # 病理含义：高聚集系数 → 转移灶内部紧密连接区域
    clustering = nx.clustering(G_nx)
    for node, coeff in clustering.items():
        sub_feats[node, 1] = coeff

    # ---- feat2：4阶环参与数 ----
    # 近似统计节点参与的长度为4的环的数量
    # 病理含义：环形结构 → 腺管样排列、细胞巢围绕
    F_np = G_pyg.x.cpu().numpy()  # 此时x还是原始语义特征，未拼接子结构

    from sklearn.metrics.pairwise import cosine_distances
    for node in G_nx.nodes():
        neighbors = list(G_nx.neighbors(node))
        if len(neighbors) < 2:
            sub_feats[node, 2] = 0.0
            continue
        neighbor_feats = F_np[neighbors]
        dist_matrix = cosine_distances(neighbor_feats)
        # 取上三角均值作为异质性得分
        n = len(neighbors)
        upper = dist_matrix[np.triu_indices(n, k=1)]
        sub_feats[node, 2] = upper.mean() if len(upper) > 0 else 0.0

    # ---- feat3：度中心性 ----
    # 归一化节点度数，范围[0,1]
    # 病理含义：高度中心性 → 转移灶中心区域；低度中心性 → 边界区域
    degree_centrality = nx.degree_centrality(G_nx)
    for node, dc in degree_centrality.items():
        sub_feats[node, 3] = dc

    sub_feats[:, 1] = -sub_feats[:, 1]  # 聚集系数取反

    sub_feats[:, 2] = -sub_feats[:, 2] 

    # ---- Min-Max归一化 ----
    # 避免不同子结构特征量纲差异过大影响后续GNN训练
    for i in range(sub_feats.shape[1]):
        col = sub_feats[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-8:
            sub_feats[:, i] = (col - col_min) / (col_max - col_min)
        else:
            sub_feats[:, i] = 0.0  # 该特征在此图中全为同一值，置零


    return sub_feats


def augment_graph_with_substructure(G_pyg):
    """
    计算子结构特征并拼接到节点特征上
    原始节点特征: (N, D) → 增强后: (N, D+4)
    同时在G_pyg上记录substructure_dim便于GNN层解析
    """
    sub_feats = compute_substructure_features(G_pyg)
    sub_tensor = torch.tensor(sub_feats, dtype=torch.float)

    # 拼接：原始特征 | 子结构特征
    # G_pyg.x此时在GPU，需要先把sub_tensor移到同一设备
    device = G_pyg.x.device
    sub_tensor = sub_tensor.to(device)
    G_pyg.x = torch.cat([G_pyg.x, sub_tensor], dim=1)

    # 记录子结构特征维度，方便GNN_pr.py中分离特征
    G_pyg.substructure_dim = 4

    return G_pyg


# ============================================================
# 主流程（原始代码保持不变，只在保存前插入子结构计算）
# ============================================================

if __name__ == '__main__':
    # similarity parameters（与原始代码完全一致）
    lambda_d = 3e-3
    lambda_f = 1.0e-3
    lamda_h = 0.8
    distance_thres = 1500
    feature_path = 'path/to/your/feature/files'
    output_path = 'path/to/your/output/directory'  # 输出路径，包含子结构特征的图
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm(os.listdir(feature_path)):
        print(filename)
        ofile = os.path.join(output_path, filename[:-4] + '.pkl')
        if os.path.isfile(ofile):
            continue

        label = int(1)
        if filename.endswith(".npz"):
            d = np.load(
                os.path.join(feature_path, filename),
                allow_pickle=True
            )
            x = d['x_coordinate']
            y = d['y_coordinate']
            F = d['feature']

            # 去除不变特征维度（原始逻辑）
            ridx = (np.max(F, axis=0) - np.min(F, axis=0)) > 1e-4
            F = F[:, ridx]

            C = np.asarray(np.vstack((x, y)).T, dtype=int)
            TC = sKDTree(C)
            I, D = TC.query_radius(
                C, r=6 / lambda_d,
                return_distance=True,
                sort_results=True
            )

            # 计算组合距离矩阵（原始逻辑）
            DX = np.zeros(int(C.shape[0] * (C.shape[0] - 1) / 2))
            idx = 0
            for i in range(C.shape[0] - 1):
                f = np.exp(-lambda_f * np.linalg.norm(F[i] - F[I[i]], axis=1))
                d_spatial = np.exp(-lambda_d * D[i])
                df = 1 - f * d_spatial
                dfi = np.ones(C.shape[0])
                dfi[I[i]] = df
                dfi = dfi[i + 1:]
                DX[idx:idx + len(dfi)] = dfi
                idx = idx + len(dfi)
            d = DX

            # 层次聚类（原始逻辑）
            Z = hierarchy.linkage(d, method='average')
            clusters = fcluster(Z, lamda_h, criterion='distance')
            uc = list(set(clusters))

            C_cluster = []
            F_cluster = []
            for c in uc:
                idx = np.where(clusters == c)
                if C[idx, :].squeeze().size == 2:
                    C_cluster.append(
                        list(np.round(C[idx, :].squeeze()))
                    )
                    F_cluster.append(
                        list(F[idx, :].squeeze())
                    )
                else:
                    C_cluster.append(
                        list(np.round(C[idx, :].squeeze().mean(axis=0)))
                    )
                    F_cluster.append(
                        list(F[idx, :].squeeze().mean(axis=0))
                    )

            C_cluster = np.array(C_cluster)
            F_cluster = np.array(F_cluster)

            # 建图（原始逻辑）
            W = connectClusters(C_cluster, dthresh=distance_thres)
            G = toGeometric(F_cluster, W, y=label)
            G.coords = toTensor(C_cluster, requires_grad=False)

            # ---- 新增：计算并拼接子结构特征 ----
            # 注意：必须在toGeometric之后、pickle保存之前执行
            # 此时G.edge_index已构建完成，可以计算图拓扑特征
            G = augment_graph_with_substructure(G)

            with open(ofile, 'wb') as f:
                pickle.dump(G, f)