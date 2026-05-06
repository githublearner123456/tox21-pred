import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from tqdm import tqdm
RDLogger.logger().setLevel(RDLogger.CRITICAL)
#超参数配置
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4
EPOCHS = 150
PATIENCE = 10
GRAD_CLIP = 0.5
OVER_SAMPLE_RATIO = 2.0

#原子特征（15维） 所有的特征都被归一化
def atom_features(atom, min_ring_size_dict):
    atomic_num = atom.GetAtomicNum() / 100.0
    degree = atom.GetDegree() / 10.0
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs() / 10.0
    implicit_valence = atom.GetImplicitValence() / 10.0
    hyb = int(atom.GetHybridization())
    is_aromatic = int(atom.GetIsAromatic())
    is_in_ring = int(atom.IsInRing())
    ring_size = min_ring_size_dict.get(atom.GetIdx(), 0) / 20.0
    in_aromatic_ring = int(is_aromatic and ring_size > 0)
    chiral_tag = int(atom.GetChiralTag())
    try:
        partial_charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
    except:
        partial_charge = 0.0
    partial_charge = partial_charge / 5.0
    is_sp = int(hyb == 1)
    is_sp2 = int(hyb == 2)
    is_sp3 = int(hyb == 3)
    return [atomic_num, degree, formal_charge, total_h, implicit_valence,
            hyb, is_aromatic, is_in_ring, ring_size, in_aromatic_ring,
            chiral_tag, partial_charge, is_sp, is_sp2, is_sp3]
#边特征（3维）
def bond_features(bond):
    bond_type = bond.GetBondTypeAsDouble()
    is_aromatic = int(bond.GetIsAromatic())
    is_in_ring = int(bond.IsInRing())
    return [bond_type, is_aromatic, is_in_ring]
#将smiles转化为图，返回包括节点，边特征 ，边索引，标签的data对象，
def smiles_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    min_ring_size = {}
    for ring in atom_rings:
        sz = len(ring)
        for idx in ring:
            if idx not in min_ring_size or sz < min_ring_size[idx]:
                min_ring_size[idx] = sz
    x = [atom_features(atom, min_ring_size) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    if label is not None:#用于有标签的分子
        label = np.nan_to_num(label.astype(float), nan=-1)#解决NaN
        y = torch.tensor(label, dtype=torch.float)
    else:#用于预测未知且没有毒性标签的分子，避免程序崩溃
        y = torch.full((12,), -1.0, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

#定义GINE模型
class GINE(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=HIDDEN_DIM, output_dim=12,
                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE):
        super().__init__()
        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim)) #self.convs 包含 3 个 GINEConv 层，每个 GINEConv 层内部各有一个独立的 mlp
            self.bns.append(BatchNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    #前向传播
    #lin_in->conv1（GINEConv1+MLP1）+BN1+relu+dropout->conv2（GINEConv2+MLP2）+BN2+relu+dropout->conv3（GINEConv3+MLP3）+BN3+relu+dropout->pooling->dropout->fc
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lin_in(x)
        for conv, bn in zip(self.convs, self.bns): #循环3次
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.fc(x)
    #计算损失值（可附带正样本权重 pos_weight 以缓解类别不平衡）
    def compute_loss(self, logits, target, pos_weight=None):
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        loss = criterion(logits, target)
        mask = (target != -1).float()
        loss = (loss * mask).sum() / mask.sum()
        return loss
    #预测
    def predict_smiles(self, smiles, device, task_names):
        self.eval()
        data = smiles_to_graph(smiles, label=None)
        if data is None:
            return None
        data = data.to(device)
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self(data.x, data.edge_index, data.edge_attr, batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return [(name, prob, "有毒" if prob >= 0.5 else "无毒") for name, prob in zip(task_names, probs)]

# 加权采样器，让训练集batch出现更多的有毒分子
def get_weighted_sampler(train_data, over_sample_ratio=OVER_SAMPLE_RATIO):
    weights = []
    for data in train_data:
        y = data.y.numpy()
        is_pos = np.any((y == 1) & (y != -1))
        w = 1.0 + (over_sample_ratio if is_pos else 0.0)
        weights.append(w)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

#Scaffold split划分数据集，此方法更符合药物筛选流程
def scaffold_split(smiles_list, dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    scaffold_to_indices = {}
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    scaffold_sets = sorted(scaffold_to_indices.values(), key=lambda x: len(x), reverse=True)
    n_total = len(dataset)
    train_target = int(frac_train * n_total)
    val_target   = int(frac_val * n_total)

    train_idx, val_idx, test_idx = [], [], []
    for scaffold in scaffold_sets:
        if len(train_idx) + len(scaffold) <= train_target:
            train_idx.extend(scaffold)
        elif len(val_idx) + len(scaffold) <= val_target:
            val_idx.extend(scaffold)
        else:
            test_idx.extend(scaffold)

    return ([dataset[i] for i in train_idx],
            [dataset[i] for i in val_idx],
            [dataset[i] for i in test_idx])

# --------------------- 评估函数 --------------------
#定义测试集与验证集的平均损失值计算方法，便于后续调用
def evaluate_loss(loader, model, pos_weight, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.view(out.shape)
            loss = model.compute_loss(out, target, pos_weight)
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0
#定义平均AUC的计算方法，便于后续调用
def evaluate_auc(loader, model, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating AUC", leave=False):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.view(out.shape)
            y_true.append(target.cpu().numpy())
            y_pred.append(torch.sigmoid(out).cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    aucs = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] != -1
        if mask.sum() > 1:
            y_pred_i = y_pred[mask, i]
            if np.isnan(y_pred_i).any():
                continue
            try:
                auc = roc_auc_score(y_true[mask, i], y_pred_i)
                aucs.append(auc)
            except:
                continue
    return np.mean(aucs) if aucs else 0.5, aucs
#加入更多的指标评估测试集
def evaluate_detailed(loader, model, device, task_names):
    model.eval()
    y_true, y_pred_proba = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Detailed Eval", leave=False):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.view(out.shape)
            y_true.append(target.cpu().numpy())
            y_pred_proba.append(torch.sigmoid(out).cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred_proba = np.vstack(y_pred_proba)
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    results = []
    for i, name in enumerate(task_names):
        mask = y_true[:, i] != -1
        if mask.sum() == 0:
            continue
        y_true_i = y_true[mask, i]
        y_pred_i = y_pred_class[mask, i]
        y_prob_i = y_pred_proba[mask, i]
        if np.isnan(y_prob_i).any():
            continue
        try:
            auc = roc_auc_score(y_true_i, y_prob_i) if len(np.unique(y_true_i)) > 1 else float('nan')
            acc = accuracy_score(y_true_i, y_pred_i)
            prec = precision_score(y_true_i, y_pred_i, zero_division=0)
            rec = recall_score(y_true_i, y_pred_i, zero_division=0)
            f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true_i, y_pred_i, labels=[0,1]).ravel()
            results.append({
                "task": name, "AUC": auc, "Accuracy": acc,
                "Precision": prec, "Recall": rec, "F1": f1,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn
            })
        except:
            continue
    return results

#定义一个epoch的训练流程
def train_one_epoch(loader, model, optimizer, pos_weight, device, grad_clip, epoch=None):
    model.train()
    total_loss = 0.0
    cnt = 0
    desc = f"Training Epoch {epoch}" if epoch is not None else "Training"
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y.view(out.shape)
        loss = model.compute_loss(out, target, pos_weight)
        if torch.isnan(loss):
            continue
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        cnt += 1
    return total_loss / cnt if cnt > 0 else 0.0

# --------------------- 主程序 --------------------
if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv(r"C:\Users\16544\Desktop\tox21.csv")  # 请修改为实际路径
    smiles_list = df["smiles"].values
    labels = df.drop(["smiles", "mol_id"], axis=1).values
    valid_smiles = []
    dataset = []
    for s, l in zip(smiles_list, labels):
        if isinstance(s, str):
            g = smiles_to_graph(s, l)
            if g is not None:
                dataset.append(g)
                valid_smiles.append(s)
    print(f"有效分子数: {len(dataset)}")

    train_data, val_data, test_data = scaffold_split(valid_smiles, dataset,
                                                     frac_train=0.8, frac_val=0.1, frac_test=0.1)
    print(f"训练集: {len(train_data)} | 验证集: {len(val_data)} | 测试集: {len(test_data)}")

    # 创建 DataLoader（使用加权采样器划分训练集的batch）
    sampler = get_weighted_sampler(train_data, OVER_SAMPLE_RATIO)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    #打印出模型的参数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = train_data[0]
    input_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    model = GINE(input_dim, edge_dim).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 计算正样本权重（缓解类别不平衡）
    y_all = torch.cat([d.y for d in train_data], dim=0)
    pos = (y_all == 1).sum(dim=0).float()
    neg = (y_all == 0).sum(dim=0).float()
    pos_weight = (neg / (pos + 1e-6)).to(device)
    print("pos_weight:", pos_weight.cpu().numpy())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    best_val_auc = 0.0
    counter = 0
    best_model_state = None
    task_names = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                  "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
    #循环150次（epochs=150），分别打印训练集与验证集的平均loss值和平均auc，用scheduler监测与调节学习率，并记录每个epoch的训练情况
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(train_loader, model, optimizer, pos_weight, device, GRAD_CLIP, epoch=epoch)
        if np.isnan(train_loss):
            print(f"Epoch {epoch} training loss NaN, stopping")
            break

        val_loss = evaluate_loss(val_loader, model, pos_weight, device)
        train_avg_auc, _ = evaluate_auc(train_loader, model, device)
        val_avg_auc, _ = evaluate_auc(val_loader, model, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_avg_auc)
        val_aucs.append(val_avg_auc)
        scheduler.step(val_avg_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train AUC: {train_avg_auc:.4f} | Val AUC: {val_avg_auc:.4f} | LR: {current_lr:.5f}")
        #早停（patience为10）
        if val_avg_auc > best_val_auc:
            best_val_auc = val_avg_auc
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    #用最佳的 model 评估测试集
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("No best model, using final model")
    test_loss = evaluate_loss(test_loader, model, pos_weight, device)
    test_auc, _ = evaluate_auc(test_loader, model, device)
    #测试集的平均AUC VS 验证集最好的平均AUC
    print(f"Test Avg Loss: {test_loss:.4f}")
    print(f"Test Avg AUC: {test_auc:.4f}")
    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    #详细打印出每个任务的各种评估指标
    test_details = evaluate_detailed(test_loader, model, device, task_names)
    print("\n========== 测试集每个任务的详细结果 ==========")
    for d in test_details:
        print(f"{d['task']:15s} AUC:{d['AUC']:.4f}  Acc:{d['Accuracy']:.4f}  Prec:{d['Precision']:.4f}  Rec:{d['Recall']:.4f}  F1:{d['F1']:.4f}")

    #可视化（训练集VS验证集；Loss与AUC）
    if train_losses:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(train_aucs, label='Train AUC', marker='o')
        plt.plot(val_aucs, label='Val AUC', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        plt.show()

    #单个分子演示
    test_smiles = "CC(C)(C)C1=CC=C(C=C1)C2=CC=C(C=C2)C(C)(C)C"
    print("\n--- 预测示例 ---")
    pred = model.predict_smiles(test_smiles, device, task_names)
    if pred:
        print(f"SMILES: {test_smiles}")
        print("任务名称        概率     预测结果")
        for name, prob, toxic in pred:
            print(f"{name:15s} {prob:.4f}  ->  {toxic}")
