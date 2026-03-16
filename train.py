
from GNN import *
from scipy.stats import mannwhitneyu
from glob import glob
import os
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, recall_score,
                             precision_score, balanced_accuracy_score,
                             matthews_corrcoef)


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

set_seed(42)
def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)

def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

def compute_metrics(y_true, y_scores, threshold=0.45):
    y_true   = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred   = (y_scores >= threshold).astype(int)
    return {
        'AUC'     : roc_auc_score(y_true, y_scores),
        'AP'      : average_precision_score(y_true, y_scores),
        'ACC'     : accuracy_score(y_true, y_pred),
        'BalACC'  : balanced_accuracy_score(y_true, y_pred),
        'F1'      : f1_score(y_true, y_pred, zero_division=0),
        'Recall'  : recall_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'MCC'     : matthews_corrcoef(y_true, y_pred),
    }

def print_metrics(metrics, prefix=''):
    print(prefix)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

def get_scores(model, loader, device):
    """从loader中获取模型预测概率和真实标签"""
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _, _ = model(batch)
            # out shape: (N, 1) 或 (N,)，取sigmoid转概率
            prob = torch.sigmoid(out).cpu().numpy().flatten()
            scores.extend(prob.tolist())
            labels.extend(batch.y.cpu().numpy().tolist())
    return np.array(scores), np.array(labels)

def check1_fold_stats(train_dataset, test_dataset, fold_idx):
    import numpy as np
    import torch

    def graph_stats(graphs):
        num_nodes = np.array([g.x.size(0) for g in graphs])
        num_edges = np.array([g.edge_index.size(1) for g in graphs])
        return {
            "nodes_mean": num_nodes.mean(),
            "nodes_std":  num_nodes.std(),
            "edges_mean": num_edges.mean(),
            "edges_std":  num_edges.std(),
        }

    def sub_stats(graphs, k=4):
        sub = torch.cat([g.x[:, -k:] for g in graphs]).cpu().numpy()
        return {
            "sub_mean": np.mean(sub, axis=0),
            "sub_zero_ratio": (sub == 0).mean(axis=0),
        }

    print(f"\n{'='*20} Fold {fold_idx+1} STRUCTURE CHECK {'='*20}")

    tr_g = graph_stats(train_dataset)
    te_g = graph_stats(test_dataset)
    tr_s = sub_stats(train_dataset)
    te_s = sub_stats(test_dataset)

    print("[Graph size]")
    print(f"  Train nodes: {tr_g['nodes_mean']:.1f} ± {tr_g['nodes_std']:.1f}")
    print(f"  Test  nodes: {te_g['nodes_mean']:.1f} ± {te_g['nodes_std']:.1f}")
    print(f"  Train edges: {tr_g['edges_mean']:.1f} ± {tr_g['edges_std']:.1f}")
    print(f"  Test  edges: {te_g['edges_mean']:.1f} ± {te_g['edges_std']:.1f}")

    print("[Substructure features]")
    for i in range(4):
        print(
            f"  Sub{i}: "
            f"train mean={tr_s['sub_mean'][i]:.4f}, "
            f"test mean={te_s['sub_mean'][i]:.4f}, "
            f"train zero={tr_s['sub_zero_ratio'][i]:.2%}, "
            f"test zero={te_s['sub_zero_ratio'][i]:.2%}"
        )

if __name__ == '__main__':

    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 300 # Total number of epochs
    split_fold = 5 # Stratified cross validation
    SUBSTRUCTURE_DIM = 4
    scheduler = None

    # Load clinical data
    clin_data = pd.read_csv('./target.csv')

    slide_ids = clin_data['slide'].astype(str).str.replace('.svs', '', regex=False)
    labels = clin_data['target'].values

    # 建立 slide → label 的字典
    slide2label = dict(zip(slide_ids, labels))
    # Load graphs
    bdir = 'path/to/your/graph/files'  # path to graphs
    graphlist = glob(os.path.join(bdir, "*.pkl"))
    GN = []
    dataset = []
    device = 'cuda:0'

    for graph in tqdm(graphlist):
        # graph: ./graphs/HobI16-053768896760.pkl
        slide_id = os.path.splitext(os.path.basename(graph))[0]

        # 如果这个 slide 在 csv 里没有标签，直接跳过
        if slide_id not in slide2label:
            continue

        label = slide2label[slide_id]

        G = pickleLoad(graph)
        G = G.to(device)

        # 二分类标签
        G.y = torch.tensor([label], dtype=torch.long, device=device)

        GN.append(G.x)
        dataset.append(G)

    # Normalise features
    GN_feat = torch.cat([G.x[:, :-SUBSTRUCTURE_DIM] for G in dataset])
    Gmean = torch.mean(GN_feat, dim=0)  # (D,)
    Gstd = torch.std(GN_feat, dim=0)  # (D,)
    Gstd[Gstd < 1e-8] = 1.0

    # 只归一化语义特征，子结构特征保持不变
    for G in dataset:
        feat = G.x[:, :-SUBSTRUCTURE_DIM]  # (N, D)
        sub = G.x[:, -SUBSTRUCTURE_DIM:]  # (N, 4)
        feat = (feat - Gmean) / Gstd
        G.x = torch.cat([feat, sub], dim=1)

    os.makedirs('path/to/your/saved/models', exist_ok=True)
    torch.save({'Gmean': Gmean.cpu(), 'Gstd': Gstd.cpu()}, 'path/to/your/saved/models/norm_stats.pth')
    SUBSTRUCTURE_DIM = 4
    sub_names = ['三角形数', '聚集系数', '4阶环数', '度中心性']

    all_sub = torch.cat([G.x[:, -SUBSTRUCTURE_DIM:].cpu() for G in dataset]).numpy()

    print("\n" + "=" * 50)
    pos_feats = [[] for _ in range(SUBSTRUCTURE_DIM)]
    neg_feats = [[] for _ in range(SUBSTRUCTURE_DIM)]

    for G in dataset:
        label = int(G.y.item())
        sub = G.x[:, -SUBSTRUCTURE_DIM:].cpu().numpy()
        for i in range(SUBSTRUCTURE_DIM):
            if label == 1:
                pos_feats[i].append(sub[:, i].mean())
            else:
                neg_feats[i].append(sub[:, i].mean())

    print("\n子结构特征判别力分析（WSI级别均值）:")
    print(f"{'特征':<12} {'阳性均值':>10} {'阴性均值':>10} {'p值':>10} {'显著性':>8}")
    print("-" * 55)
    for i, name in enumerate(sub_names):
        pos = np.array(pos_feats[i])
        neg = np.array(neg_feats[i])
        stat, p = mannwhitneyu(pos, neg, alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"{name:<12} {pos.mean():>10.4f} {neg.mean():>10.4f} {p:>10.4f} {sig:>8}")

    Y = [float(G.y) for G in dataset]


    skf = StratifiedKFold(n_splits=split_fold, shuffle=True, random_state=42)  # Stratified cross validation
    fold_val_metrics = []
    fold_test_metrics = []
    Fdata = []
    Vacc, Tacc, Vapr, Tapr, Test_ROC_overall, Test_PR_overall = [], [], [], [], [],  [] # Intialise outputs

    Fdata = []
    for fold_idx, (trvi, test) in enumerate(skf.split(dataset, Y)):
        train, valid = train_test_split(
            trvi,
            test_size=0.10,
            shuffle=True,
            stratify=np.array(Y)[trvi],
            random_state=42
        )  # 10% for validation and 90% for training
        sampler = StratifiedSampler(class_vector=torch.from_numpy(np.array(Y)[train]), batch_size=8)

        train_dataset = [dataset[i] for i in train]
        tr_loader = DataLoader(train_dataset, batch_sampler=sampler)
        valid_dataset = [dataset[i] for i in valid]
        v_loader = DataLoader(valid_dataset, shuffle=False)
        test_dataset = [dataset[i] for i in test]
        tt_loader = DataLoader(test_dataset, shuffle=False)
        check1_fold_stats(train_dataset, test_dataset, fold_idx)

        model = GNN(dim_features=dataset[0].x.shape[1], dim_target=1, layers=[16, 16, 8], dropout=0.5, pooling='mean',
                    conv='EdgeConv', aggr='max' ) # model GNN architecture

        net = NetWrapper(model, loss_function=None, device=device)
        model = model.to(device=net.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        Q, train_loss, train_acc, val_loss, val_acc, tt_loss, tt_acc, val_pr, test_pr = net.train(
            train_loader=tr_loader,
            max_epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            clipping=None,
            validation_loader=v_loader,
            test_loader=tt_loader,
            early_stopping=30,
            return_best=False,
            log_every=10)
        val_scores, val_labels = get_scores(model, v_loader, device)
        test_scores, test_labels = get_scores(model, tt_loader, device)

        val_m = compute_metrics(val_labels, val_scores)
        test_m = compute_metrics(test_labels, test_scores)

        fold_val_metrics.append(val_m)
        fold_test_metrics.append(test_m)

        print_metrics(val_m, prefix=f'\n[Fold {fold_idx + 1}] Validation Metrics:')
        print_metrics(test_m, prefix=f'[Fold {fold_idx + 1}] Test Metrics:')
        print(f"\nfold complete {len(Vacc)}: "
              f"train_acc={train_acc:.4f}, "
              f"val_auc={val_acc:.4f}, "
              f"test_auc={tt_acc:.4f}, "
              f"val_pr={val_pr:.4f}, "
              f"test_pr={test_pr:.4f}")
        model.eval()
        with torch.no_grad():
            pos_attn, neg_attn = [], []
            for G in dataset:
                G_device = G.to(device)
                sub_i = G_device.x[G_device.edge_index[0], -SUBSTRUCTURE_DIM:]
                sub_j = G_device.x[G_device.edge_index[1], -SUBSTRUCTURE_DIM:]
                attn = model.sub_attn(sub_i, sub_j).cpu().numpy().flatten()
                if int(G.y.item()) == 1:
                    pos_attn.append(attn.mean())
                else:
                    neg_attn.append(attn.mean())

        pos_attn = np.array(pos_attn)
        neg_attn = np.array(neg_attn)
        all_attn = np.concatenate([pos_attn, neg_attn])
        print(f"\n[Fold {fold_idx + 1}] 注意力权重诊断:")
        print(f"  阳性WSI: {pos_attn.mean():.4f} +/- {pos_attn.std():.4f}")
        print(f"  阴性WSI: {neg_attn.mean():.4f} +/- {neg_attn.std():.4f}")
        print(f"  整体标准差: {all_attn.std():.4f}  （<0.05说明注意力没有分化）")

        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        Vapr.append(val_pr)
        Tapr.append(test_pr)
        # ---- 保存每折最优模型 ----
        save_dir = 'path/to/your/saved/models'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存Q里验证集最优的模型
        best_mdl = Q[0]
        if type(best_mdl) in [tuple, list]:
            best_mdl = best_mdl[0]
        torch.save(
            best_mdl.state_dict(),
            os.path.join(save_dir, f'fold_{fold_idx+1}_best.pth')
        )
        # 同时保存模型结构参数，方便后续加载
        torch.save(
            {
                'state_dict'      : best_mdl.state_dict(),
                'dim_features'    : dataset[0].x.shape[1],
                'dim_target'      : 1,
                'layers'          : [16, 16, 8],
                'dropout'         : 0.5,
                'pooling'         : 'mean',
                'conv'            : 'EdgeConv',
                'aggr'            : 'max',
                'substructure_dim': SUBSTRUCTURE_DIM,
                'fold_idx'        : fold_idx + 1,
            },
            os.path.join(save_dir, f'fold_{fold_idx+1}_best_full.pth')
        )
        print(f"[Fold {fold_idx+1}] 模型已保存至 {save_dir}/fold_{fold_idx+1}_best_full.pth")

        Fdata.append((Q, test_dataset, train_dataset))
        Fdata.append((Q, test_dataset, train_dataset))

        # ---- 5折平均结果（最终epoch模型）----
    print(f"\n{'=' * 50}")
    print("Average results across 5 folds (final epoch model):")
    print(f"{'=' * 50}")
    metric_keys = fold_test_metrics[0].keys()
    for k in metric_keys:
        vals = [m[k] for m in fold_test_metrics]
        print(f"  {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # 原始格式输出（保留，方便和原始SlideGraph+结果对比）
    print(f"\n{'=' * 50}")
    print('Original format (for comparison with baseline):')
    print("avg Valid AUC=", np.mean(Vacc), "+/-", np.std(Vacc))
    print("avg Test AUC=", np.mean(Tacc), "+/-", np.std(Tacc))
    print("avg Valid PR=", np.mean(Vapr), "+/-", np.std(Vapr))
    print("avg Test PR=", np.mean(Tapr), "+/-", np.std(Tapr))

    # ---- Top-10模型集成结果 ----
    print(f"\n{'=' * 50}")
    print("Ensemble results (top-10 models per fold):")
    print(f"{'=' * 50}")
    ensemble_metrics_list = []
    auroc, aupr = [], []

    for idx in range(len(Fdata)):
        Q, test_dataset, train_dataset = Fdata[idx]
        zz, yy = EnsembleDecisionScoring(
            Q, train_dataset, test_dataset,
            device=net.device, k=10
        )
        m = compute_metrics(yy, zz)
        ensemble_metrics_list.append(m)
        auroc.append(roc_auc_score(yy, zz))
        aupr.append(average_precision_score(yy, zz))
        print_metrics(m, prefix=f'[Fold {idx + 1}] Ensemble Test Metrics:')

    print(f"\n{'=' * 50}")
    print("Final averaged ensemble metrics:")
    print(f"{'=' * 50}")
    for k in metric_keys:
        vals = [m[k] for m in ensemble_metrics_list]
        print(f"  {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print(f"\nOriginal format ensemble:")
    print("avg Test AUC overall=", np.mean(auroc), "+/-", np.std(auroc))
    print("avg Test PR overall=", np.mean(aupr), "+/-", np.std(aupr))
"""
Final averaged ensemble metrics:
==================================================
  AUC: 0.9242 +/- 0.0343
  AP: 0.9021 +/- 0.0460
  ACC: 0.9231 +/- 0.0243
  BalACC: 0.8697 +/- 0.0485
  F1: 0.8396 +/- 0.0636
  Recall: 0.7500 +/- 0.1059
  Precision: 0.9714 +/- 0.0571
  MCC: 0.8070 +/- 0.0636

Original format ensemble:
avg Test AUC overall= 0.9242272347535504 +/- 0.03425596491969579
avg Test PR overall= 0.9021085994206295 +/- 0.046015607376161824



"""
