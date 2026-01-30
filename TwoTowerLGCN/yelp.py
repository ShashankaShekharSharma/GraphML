

import os
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================== Config =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MAX_USERS = 20000     # keep for Kaggle safety
MIN_INTERACTIONS = 1 # <-- cold users allowed

EMBED_DIM = 64
NUM_LAYERS = 2
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 2048

TOPKS = [20, 40]
L2_REG = 1e-4
CLAMP = 10.0
EVAL_BATCH = 256   # lower for safety with many items

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ===================== Load Yelp Reviews =====================
def load_yelp(path):
    review_path = os.path.join(path, "yelp_academic_dataset_review.json")

    users, items = [], []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading Yelp reviews"):
            r = json.loads(line)
            users.append(r["user_id"])
            items.append(r["business_id"])

    df = pd.DataFrame({"u": users, "i": items})

    # remap users (cap users, NOT items)
    user_ids = df.u.unique()[:MAX_USERS]
    u_map = {u: idx for idx, u in enumerate(user_ids)}
    df = df[df.u.isin(u_map)]
    df["u"] = df.u.map(u_map)

    # remap ALL items
    item_ids = df.i.unique()
    i_map = {i: idx for idx, i in enumerate(item_ids)}
    df["i"] = df.i.map(i_map)

    interactions = list(zip(df.u.values, df.i.values))
    return interactions, len(u_map), len(i_map)

# ===================== Cold-safe Train/Test Split =====================
def split_data(interactions):
    user_items = defaultdict(list)
    for u, i in interactions:
        user_items[u].append(i)

    train, test = [], []

    for u, items in user_items.items():
        items = list(set(items))

        # cold user: train only
        if len(items) < 2:
            train += [(u, items[0])]
            continue

        random.shuffle(items)
        split = max(1, int(0.8 * len(items)))
        train += [(u, i) for i in items[:split]]
        test  += [(u, i) for i in items[split:]]

    return train, test

# ===================== Dataset =====================
class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ===================== Two-Tower LightGCN =====================
class TwoTowerLightGCN(nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, EMBED_DIM)
        self.item_emb = nn.Embedding(n_items, EMBED_DIM)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate(self, adj):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        out = x
        for _ in range(NUM_LAYERS):
            out = torch.sparse.mm(adj, out)
            x = x + out
        x = x / (NUM_LAYERS + 1)
        return torch.split(x, [self.user_emb.num_embeddings,
                               self.item_emb.num_embeddings])

# ===================== Graph =====================
def build_adj(interactions, n_users, n_items):
    u = torch.tensor([x[0] for x in interactions])
    i = torch.tensor([x[1] for x in interactions]) + n_users

    idx = torch.cat([torch.stack([u, i]), torch.stack([i, u])], dim=1)
    val = torch.ones(idx.size(1))
    size = n_users + n_items

    adj = torch.sparse_coo_tensor(idx, val, (size, size)).coalesce()
    deg = torch.sparse.sum(adj, 1).to_dense()
    deg_inv = deg.pow(-0.5)
    deg_inv[deg_inv == float("inf")] = 0

    r, c = adj.indices()
    return torch.sparse_coo_tensor(
        adj.indices(),
        adj.values() * deg_inv[r] * deg_inv[c],
        adj.size()
    ).coalesce().to(DEVICE)

# ===================== Training =====================
def train_epoch(model, loader, adj, opt):
    model.train()
    total = 0.0

    for u, p in loader:
        u, p = u.to(DEVICE), p.to(DEVICE)
        u_emb, i_emb = model.propagate(adj)

        n = torch.randint(0, i_emb.size(0), p.size(), device=DEVICE)

        pos = (u_emb[u] * i_emb[p]).sum(1)
        neg = (u_emb[u] * i_emb[n]).sum(1)

        loss = -torch.log(torch.sigmoid(torch.clamp(pos - neg, -CLAMP, CLAMP)) + 1e-8).mean()
        loss += L2_REG * (u_emb[u].norm(2)**2 + i_emb[p].norm(2)**2) / u.size(0)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    return total / len(loader)

# ===================== Full Ranking Evaluation =====================
@torch.no_grad()
def evaluate(model, adj, train_dict, test_dict):
    model.eval()
    u_emb, i_emb = model.propagate(adj)
    u_emb = F.normalize(u_emb, dim=1)
    i_emb = F.normalize(i_emb, dim=1)

    metrics = {k: {"recall": [], "ndcg": []} for k in TOPKS}
    users = list(test_dict.keys())

    for i in tqdm(range(0, len(users), EVAL_BATCH), desc="Full ranking"):
        batch = users[i:i + EVAL_BATCH]
        scores = torch.matmul(u_emb[batch], i_emb.T)

        for r, u in enumerate(batch):
            scores[r, list(train_dict[u])] = -1e9

        for k in TOPKS:
            topk = torch.topk(scores, k, dim=1).indices.cpu().numpy()
            for r, u in enumerate(batch):
                pos = test_dict[u]
                hits = [1 if x in pos else 0 for x in topk[r]]

                metrics[k]["recall"].append(sum(hits) / len(pos))
                metrics[k]["ndcg"].append(
                    sum(hits[j] / np.log2(j + 2) for j in range(len(hits))) /
                    sum(1 / np.log2(j + 2) for j in range(min(len(pos), k)))
                )

    return metrics

# ===================== Main =====================
def main():
    path = "/kaggle/input/yelp-dataset"
    interactions, n_users, n_items = load_yelp(path)

    train_data, test_data = split_data(interactions)

    train_dict, test_dict = defaultdict(set), defaultdict(set)
    for u, i in train_data: train_dict[u].add(i)
    for u, i in test_data:  test_dict[u].add(i)

    adj = build_adj(interactions, n_users, n_items)

    loader = DataLoader(
        TrainDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = TwoTowerLightGCN(n_users, n_items).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCHS):
        loss = train_epoch(model, loader, adj, opt)
        print(f"Epoch {e+1}/{EPOCHS} - Loss: {loss:.4f}")

    metrics = evaluate(model, adj, train_dict, test_dict)

    print("\nFINAL METRICS (FULL RANKING)")
    for k in TOPKS:
        print(f"Recall@{k}: {np.mean(metrics[k]['recall']):.4f}")
        print(f"NDCG@{k}:   {np.mean(metrics[k]['ndcg']):.4f}")

if __name__ == "__main__":
    main()

'''
Recall@20: 0.1272
NDCG@20:   0.0938
Recall@40: 0.1617
NDCG@40:   0.1021
'''
