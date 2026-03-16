
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from platt import PlattScaling
from utils import *
from torch.utils.data import Sampler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix)


# ============================================================
# 完全不变的部分
# ============================================================

from torch_scatter import scatter_logsumexp

def global_lse_pool(x, batch, tau=5.0):
    return scatter_logsumexp(x * tau, batch, dim=0) / tau

class StratifiedSampler(Sampler):
    """Stratified Sampling - 完全不变"""
    def __init__(self, class_vector, batch_size=10):
        self.batch_size = batch_size
        self.n_splits = int(class_vector.size(0) / self.batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        YY = self.class_vector.numpy()
        idx = np.arange(len(YY))
        return [tidx for _, tidx in skf.split(idx, YY)]

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def calc_roc_auc(target, prediction):
    return roc_auc_score(toNumpy(target), toNumpy(prediction[:, -1]))


def calc_pr(target, prediction):
    return average_precision_score(toNumpy(target), toNumpy(prediction[:, -1]))


# ============================================================
# 新增：子结构感知注意力模块
# ============================================================

class SubstructureAttention(torch.nn.Module):
    """
    用边两端节点的子结构特征生成边级别的注意力权重。

    输入: 边两端节点子结构特征的拼接 (substructure_dim * 2,)
    输出: 标量注意力权重 in (0, 1)

    病理含义:
    - 两个肿瘤聚集节点之间（三角形密度都高）的边权重高
    - 消息传递自动聚焦于转移灶的核心区域
    - 边界节点与内部节点之间的边权重低
    """
    def __init__(self, substructure_dim):
        super(SubstructureAttention, self).__init__()
        self.attn_mlp = Sequential(
            Linear(substructure_dim * 2, 16),
            ReLU(),
            Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, sub_i, sub_j):
        """
        sub_i: (E, substructure_dim) 边起点的子结构特征
        sub_j: (E, substructure_dim) 边终点的子结构特征
        返回:  (E, 1) 注意力权重
        """
        sub_pair = torch.cat([sub_i, sub_j], dim=1)
        return self.attn_mlp(sub_pair)


class SubstructureAwarePooling(torch.nn.Module):
    def __init__(self, node_dim, substructure_dim):
        super(SubstructureAwarePooling, self).__init__()

        self.sub_enhance = torch.nn.Sequential(
            torch.nn.Linear(substructure_dim, node_dim),
            torch.nn.BatchNorm1d(node_dim),
            torch.nn.ReLU()
        )

        self.gate = torch.nn.Sequential(
            torch.nn.Linear(node_dim * 2, node_dim),
            torch.nn.Sigmoid()
        )

        # score计算：输入加入节点表示与图均值的差
        # 让score感知节点相对于整图的偏差程度
        # 病理含义：转移灶节点的特征偏离正常淋巴组织均值越大，
        # 越应该获得高重要性分数
        self.score_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_dim * 2, node_dim),  # 拼接h和(h - graph_mean)
            torch.nn.BatchNorm1d(node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, 1)
        )
        self.mix = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, h, sub, batch):
        # step1: 子结构特征升维
        sub_enhanced = self.sub_enhance(sub)

        # step2: 门控融合
        gate_weight = self.gate(torch.cat([h, sub_enhanced], dim=1))
        h_fused = h + gate_weight * sub_enhanced

        # step3: 图均值和图标准差
        graph_mean = global_mean_pool(h, batch)
        graph_mean_expanded = graph_mean[batch]

        # 新增：图标准差作为归一化因子
        # 把绝对偏差归一化为z-score，避免极端节点垄断score
        h_sq = global_mean_pool(h ** 2, batch)
        graph_std = (h_sq - graph_mean ** 2).clamp(min=1e-8).sqrt()
        graph_std_expanded = graph_std[batch]

        # step4: z-score归一化的偏差信号
        h_deviation = (h - graph_mean_expanded) / (graph_std_expanded + 1e-8)

        # step5: score计算
        score_logit = self.score_mlp(
            torch.cat([h_fused, h_deviation], dim=1)
        )

        # step6: 图内softmax
        from torch_scatter import scatter_softmax
        score = scatter_softmax(
            score_logit,
            batch.unsqueeze(1).expand_as(score_logit),
            dim=0
        )

        # step7: 双路聚合
        weighted_fused = global_add_pool(h_fused * score, batch)
        weighted_orig  = global_mean_pool(h, batch)
        alpha = torch.sigmoid(self.mix)
        return alpha * weighted_fused + (1 - alpha) * weighted_orig                     # (B, node_dim)                  # (B, node_dim)
# ============================================================
# 修改：GNN类
# 只保留EdgeConv，移除GINConv，加入子结构感知
# ============================================================

class GNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[6, 6],
                 pooling='max', dropout=0.0, conv='EdgeConv',
                 gembed=False, substructure_dim=4, **kwargs):
        """
        参数与原始GNN完全一致，新增substructure_dim。

        Parameters
        ----------
        dim_features     : 节点特征总维度（含子结构特征）
        dim_target       : 输出维度
        layers           : 每层隐藏维度列表，默认[6,6]
        pooling          : 池化方式 'max'/'mean'/'add'
        dropout          : dropout率
        conv             : 只支持'EdgeConv'
        gembed           : 是否用图嵌入做分类
        substructure_dim : 子结构特征维度，节点特征的最后这几维
                           被视为子结构特征，默认4
        """
        super(GNN, self).__init__()

        if conv != 'EdgeConv':
            raise NotImplementedError(
                f"Only EdgeConv is supported, got {conv}"
            )

        self.dropout         = dropout
        self.embeddings_dim  = layers
        self.no_layers       = len(self.embeddings_dim)
        self.gembed          = gembed
        self.substructure_dim = substructure_dim
        self.pooling = {
            'max' : global_max_pool,
            'mean': global_mean_pool,
            'add' : global_add_pool
        }[pooling]

        # 真正的语义特征维度（去掉子结构特征部分）
        feat_dim = dim_features - substructure_dim

        # 子结构注意力模块（所有层共享，减少参数量）
        self.sub_attn = SubstructureAttention(substructure_dim)

        # 每层一个子结构残差投影
        # 把子结构特征投影到该层输出维度，做残差加法
        self.sub_proj = torch.nn.ModuleList()
        self.sub_scale = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(0.1))
            for _ in self.embeddings_dim
        ])
        self.sub_aware_pool = SubstructureAwarePooling(
            node_dim=self.embeddings_dim[-1],
            substructure_dim=substructure_dim
        )

        self.nns     = []
        self.convs   = []
        self.linears = []

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                # 第0层：Linear直接作用在语义特征上（不含子结构）
                self.first_h = Sequential(
                    Linear(feat_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU()
                )
                self.linears.append(Linear(out_emb_dim, dim_target))
                self.sub_proj.append(Linear(substructure_dim, out_emb_dim))

            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.linears.append(Linear(out_emb_dim, dim_target))

                # EdgeConv: 输入是 [x_i || x_j - x_i]，维度 2 * input_emb_dim
                subnet = Sequential(
                    Linear(2 * input_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU()
                )
                self.nns.append(subnet)
                self.convs.append(EdgeConv(self.nns[-1], **kwargs))
                self.sub_proj.append(Linear(substructure_dim, out_emb_dim))

        self.nns     = torch.nn.ModuleList(self.nns)
        self.convs   = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)
        # sub_proj已在循环中定义为ModuleList

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ---- 分离语义特征和子结构特征 ----
        feat = x[:, :-self.substructure_dim]   # (N, feat_dim)
        sub  = x[:, -self.substructure_dim:]   # (N, substructure_dim)

        # ---- 预计算边级别子结构注意力权重 ----
        # edge_index[0]: 起点, edge_index[1]: 终点
        sub_i = sub[edge_index[0]]   # (E, substructure_dim)
        sub_j = sub[edge_index[1]]   # (E, substructure_dim)
        attn_weight = self.sub_attn(sub_i, sub_j)  # (E, 1)

        # ---- 把边级别注意力聚合为节点级别 ----
        # 对每个节点，取其所有入边注意力权重的均值
        node_attn = torch.zeros(feat.size(0), 1, device=feat.device)
        node_attn.scatter_add_(
            0,
            edge_index[1].unsqueeze(1).expand(-1, 1),
            attn_weight
        )
        degree = torch.zeros(feat.size(0), device=feat.device)
        degree.scatter_add_(
            0,
            edge_index[1],
            torch.ones(edge_index.size(1), device=feat.device)
        )
        degree     = degree.clamp(min=1).unsqueeze(1)
        node_attn  = node_attn / degree   # (N, 1) 归一化注意力权重

        out = 0
        Z   = 0
        h   = feat   # 当前层的节点表示（只含语义特征）

        for layer in range(self.no_layers):
            if layer == 0:
                # 第0层：线性变换 + 子结构残差
                h = self.first_h(h)
                h = h + self.sub_scale[0] * self.sub_proj[0](sub)          # 子结构残差

                z    = self.linears[layer](h)
                Z   += z
                dout = F.dropout(
                    self.pooling(z, batch),
                    p=self.dropout, training=self.training
                )
                out += dout


            else:

                h_new = self.convs[layer - 1](h, edge_index)

                h_new = h_new * node_attn

                h = h_new + self.sub_scale[layer] * self.sub_proj[layer](sub)

                if not self.gembed:

                    z = self.linears[layer](h)

                    Z += z

                    # 最后一层用子结构感知池化，其余层用原始pooling

                    if layer == self.no_layers - 1:

                        pooled = self.sub_aware_pool(h, sub, batch)

                        dout = F.dropout(

                            self.linears[layer](pooled),

                            p=self.dropout, training=self.training

                        )

                    else:

                        dout = F.dropout(

                            self.pooling(z, batch),

                            p=self.dropout, training=self.training

                        )

                else:

                    dout = F.dropout(

                        self.linears[layer](self.pooling(h, batch)),

                        p=self.dropout, training=self.training

                    )

                out += dout

        return out, Z, h


# ============================================================
# 以下完全不变
# ============================================================

def decision_function(model, loader, device='cpu',
                      outOnly=True, returnNumpy=False):
    if type(loader) is not DataLoader:
        loader = DataLoader(loader)
    if type(device) == type(''):
        device = torch.device(device)
    ZXn = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            data   = data.to(device)
            output, zn, xn = model(data)
            if returnNumpy:
                zn, xn = toNumpy(zn), toNumpy(xn)
            if not outOnly:
                ZXn.append((zn, xn))
            if i == 0:
                Z = output
                Y = data.y
            else:
                Z = torch.cat((Z, output))
                Y = torch.cat((Y, data.y))
    if returnNumpy:
        Z, Y = toNumpy(Z), toNumpy(Y)
    return Z, Y, ZXn


def EnsembleDecisionScoring(Q, train_dataset, test_dataset,
                             device='cpu', k=None):
    Z = 0
    if k is None:
        k = len(Q)
    for i, mdl in enumerate(Q):
        if type(mdl) in [tuple, list]:
            mdl = mdl[0]
        zz, yy, _ = decision_function(mdl, train_dataset, device=device)
        mdl.rescaler = PlattScaling().fit(toNumpy(yy), toNumpy(zz))
        zz, yy, _  = decision_function(mdl, test_dataset, device=device)
        zz, yy     = mdl.rescaler.transform(toNumpy(zz)).ravel(), toNumpy(yy)
        Z += zz
        if i + 1 == k:
            break
    Z = Z / k
    return Z, yy


class NetWrapper:
    def __init__(self, model, loss_function, device='cpu',
                 classification=True):
        self.model          = model
        self.loss_fun       = loss_function
        self.device         = torch.device(device)
        self.classification = classification

    def _pair_train(self, train_loader, optimizer, clipping=None):
        model = self.model.to(self.device)
        model.train()
        loss_all = 0
        acc_all  = 0
        assert self.classification
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output, _, _ = model(data)
            y    = data.y
            loss = 0
            c    = 0
            z    = toTensor([0])
            for i in range(len(y) - 1):
                for j in range(i + 1, len(y)):
                    if y[i] != y[j]:
                        c  += 1
                        dz  = output[i, -1] - output[j, -1]
                        dy  = y[i] - y[j]
                        loss += torch.max(z, 1.0 - dy * dz)
            loss = loss / c
            acc  = loss
            loss.backward()
            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)
            loss_all += loss.item() * num_graphs
            acc_all  += acc.item()  * num_graphs
            if clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clipping
                )
            optimizer.step()
        return (acc_all  / len(train_loader.dataset),
                loss_all / len(train_loader.dataset))

    def classify_graphs(self, loader):
        Z, Y, _ = decision_function(
            self.model, loader, device=self.device
        )
        if not isinstance(Z, tuple):
            Z = (Z,)
        loss    = 0
        auc_val = calc_roc_auc(Y, *Z)
        pr      = calc_pr(Y, *Z)
        return auc_val, loss, pr

    def train(self, train_loader, max_epochs=100,
              optimizer=torch.optim.Adam, scheduler=None,
              clipping=None, validation_loader=None,
              test_loader=None, early_stopping=100,
              return_best=True, log_every=0):
        from collections import deque
        Q            = deque(maxlen=10)
        return_best  = return_best and validation_loader is not None
        val_loss, val_acc = -1, -1
        best_val_acc                 = -1
        test_acc_at_best_val_acc     = -1
        val_pr_at_best_val_acc       = -1
        test_pr_at_best_val_acc      = -1
        test_loss, test_acc          = None, None
        time_per_epoch               = []
        self.history                 = []
        patience                     = early_stopping
        best_epoch                   = np.inf
        iterator                     = tqdm(range(1, max_epochs + 1))

        for epoch in iterator:
            updated = False
            if scheduler is not None:
                scheduler.step(epoch)
            start = time.time()
            train_acc, train_loss = self._pair_train(
                train_loader, optimizer, clipping
            )
            end = time.time() - start
            time_per_epoch.append(end)

            if validation_loader is not None:
                val_acc, val_loss, val_pr = self.classify_graphs(
                    validation_loader
                )
            if test_loader is not None:
                test_acc, test_loss, test_pr = self.classify_graphs(
                    test_loader
                )
            if val_acc > best_val_acc:
                best_val_acc             = val_acc
                test_acc_at_best_val_acc = test_acc
                val_pr_at_best_val_acc   = val_pr
                test_pr_at_best_val_acc  = test_pr
                best_epoch               = epoch
                updated                  = True
                if return_best:
                    best_model = deepcopy(self.model)
                    Q.append((best_model, best_val_acc,
                               test_acc_at_best_val_acc,
                               val_pr_at_best_val_acc,
                               test_pr_at_best_val_acc))
            if not return_best:
                Q.append((deepcopy(self.model), val_acc,
                           test_acc, val_pr, test_pr))

            showresults = False
            if log_every == 0:
                showresults = updated
            elif (epoch - 1) % log_every == 0:
                showresults = True

            if showresults:
                msg = (f'Epoch: {epoch}, '
                       f'TR loss: {train_loss:.4f} '
                       f'TR perf: {train_acc:.4f}, '
                       f'VL perf: {val_acc:.4f} '
                       f'TE perf: {test_acc:.4f}, '
                       f'Best: VL perf: {best_val_acc:.4f} '
                       f'TE perf: {test_acc_at_best_val_acc:.4f} '
                       f'VL pr: {val_pr_at_best_val_acc:.4f} '
                       f'TE pr: {test_pr_at_best_val_acc:.4f}')
                tqdm.write('\n' + msg)
                self.history.append(train_loss)

            if epoch - best_epoch > patience:
                iterator.close()
                break

        if return_best:
            val_acc  = best_val_acc
            test_acc = test_acc_at_best_val_acc
            val_pr   = val_pr_at_best_val_acc
            test_pr  = test_pr_at_best_val_acc

        Q.reverse()
        return (Q, train_loss, train_acc, val_loss,
                np.round(val_acc, 2), test_loss,
                np.round(test_acc, 2), val_pr, test_pr)