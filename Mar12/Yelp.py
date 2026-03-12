# =========================================================
# Yelp — Hybrid (ID + Feature) Two-Tower + LightGCN + InfoNCE
# FULL RANKING: Recall@20 / Recall@40 / NDCG@20 / NDCG@40
# =========================================================

import os, json, random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================== CONFIG =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MAX_USERS = 20000
EMBED_DIM = 64
NUM_LAYERS = 3
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 2048

TOPKS = [20, 40]

SSL_WEIGHT = 0.1
SSL_TEMP = 0.2
EDGE_DROPOUT = 0.1
CLAMP = 10.0
EVAL_BATCH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# =========================================================
# LOAD YELP + FEATURES
# =========================================================
def load_yelp_with_features(path):
    review_file = os.path.join(path, "yelp_academic_dataset_review.json")
    user_file   = os.path.join(path, "yelp_academic_dataset_user.json")
    biz_file    = os.path.join(path, "yelp_academic_dataset_business.json")

    users, items = [], []
    for line in tqdm(open(review_file), desc="Loading reviews"):
        r = json.loads(line)
        users.append(r["user_id"])
        items.append(r["business_id"])

    df = pd.DataFrame({"u": users, "i": items})

    # ---- User features ----
    u_feats = {}
    for line in tqdm(open(user_file), desc="Loading users"):
        r = json.loads(line)
        u_feats[r["user_id"]] = [
            r.get("review_count", 0),
            r.get("average_stars", 0),
            r.get("fans", 0)
        ]

    # ---- Item features ----
    i_feats = {}
    for line in tqdm(open(biz_file), desc="Loading businesses"):
        r = json.loads(line)
        i_feats[r["business_id"]] = [
            r.get("stars", 0),
            r.get("review_count", 0),
            int(r.get("is_open", 1))
        ]

    # Cap users
    uids = df.u.unique()[:MAX_USERS]
    u_map = {u: idx for idx, u in enumerate(uids)}
    df = df[df.u.isin(u_map)]
    df["u"] = df.u.map(u_map)

    # Remap items
    iids = df.i.unique()
    i_map = {i: idx for idx, i in enumerate(iids)}
    df["i"] = df.i.map(i_map)

    # Build feature matrices
    user_feat_mat = np.array([u_feats.get(u, [0,0,0]) for u in uids], dtype=np.float32)
    item_feat_mat = np.array([i_feats.get(i, [0,0,0]) for i in iids], dtype=np.float32)

    # ---- Normalize features (z-score) ----
    def normalize(x):
        mean = x.mean(0, keepdims=True)
        std = x.std(0, keepdims=True) + 1e-6
        return (x - mean) / std

    user_feat_mat = normalize(user_feat_mat)
    item_feat_mat = normalize(item_feat_mat)

    interactions = list(zip(df.u.values, df.i.values))
    return interactions, len(u_map), len(i_map), user_feat_mat, item_feat_mat

# =========================================================
# SPLIT (Cold-user safe)
# =========================================================
def split_data(interactions):
    user_items = defaultdict(list)
    for u,i in interactions:
        user_items[u].append(i)

    train, test = [], []
    for u,items in user_items.items():
        items = list(set(items))
        if len(items) < 2:
            train.append((u,items[0]))
            continue
        random.shuffle(items)
        k = max(1,int(0.8*len(items)))
        train += [(u,i) for i in items[:k]]
        test  += [(u,i) for i in items[k:]]
    return train,test

# =========================================================
# DATASET
# =========================================================
class TrainDataset(Dataset):
    def __init__(self,data): self.data=data
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

# =========================================================
# MODEL (Hybrid ID + Feature)
# =========================================================
class HybridLightGCN(nn.Module):
    def __init__(self, n_users, n_items, user_feat_dim, item_feat_dim):
        super().__init__()

        # ID embeddings
        self.user_id_emb = nn.Embedding(n_users, EMBED_DIM)
        self.item_id_emb = nn.Embedding(n_items, EMBED_DIM)

        nn.init.xavier_uniform_(self.user_id_emb.weight)
        nn.init.xavier_uniform_(self.item_id_emb.weight)

        # Feature towers
        self.user_tower = nn.Sequential(
            nn.Linear(user_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, EMBED_DIM)
        )

        self.item_tower = nn.Sequential(
            nn.Linear(item_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, EMBED_DIM)
        )

        self.n_users = n_users
        self.n_items = n_items

    def propagate(self, adj, user_feats, item_feats):
        u0 = self.user_id_emb.weight + self.user_tower(user_feats)
        i0 = self.item_id_emb.weight + self.item_tower(item_feats)

        x = torch.cat([u0, i0], dim=0)
        out = x

        for _ in range(NUM_LAYERS):
            out = torch.sparse.mm(adj, out)
            x = x + out

        x = x / (NUM_LAYERS + 1)
        return torch.split(x, [self.n_users, self.n_items])

# =========================================================
# GRAPH
# =========================================================
def build_adj(interactions, n_users, n_items):
    u = torch.tensor([x[0] for x in interactions])
    i = torch.tensor([x[1] for x in interactions]) + n_users

    idx = torch.cat([torch.stack([u,i]), torch.stack([i,u])], 1)
    val = torch.ones(idx.size(1))
    size = n_users + n_items

    adj = torch.sparse_coo_tensor(idx,val,(size,size)).coalesce()
    deg = torch.sparse.sum(adj,1).to_dense()
    deg_inv = deg.pow(-0.5)
    deg_inv[deg_inv==float("inf")] = 0

    r,c = adj.indices()
    return torch.sparse_coo_tensor(adj.indices(),
                                   adj.values()*deg_inv[r]*deg_inv[c],
                                   adj.size()).coalesce().to(DEVICE)

def drop_edge(adj,p):
    idx,val = adj.indices(),adj.values()
    mask = torch.rand(val.size(0),device=val.device)>p
    return torch.sparse_coo_tensor(idx[:,mask],val[mask],adj.size()).coalesce()

# =========================================================
# TRUE INFO-NCE
# =========================================================
def info_nce(z1, z2, temp):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, labels)

# =========================================================
# TRAIN
# =========================================================
def train_epoch(model, loader, adj, opt, user_feats, item_feats):
    model.train()
    total = 0

    for u,p in loader:
        u,p = u.to(DEVICE), p.to(DEVICE)

        adj2 = drop_edge(adj, EDGE_DROPOUT)

        u1,i1 = model.propagate(adj, user_feats, item_feats)
        u2,i2 = model.propagate(adj2, user_feats, item_feats)

        # ---- BPR ----
        n = torch.randint(0, i1.size(0), p.size(), device=DEVICE)
        pos = (u1[u]*i1[p]).sum(1)
        neg = (u1[u]*i1[n]).sum(1)
        bpr = -torch.log(torch.sigmoid(torch.clamp(pos-neg,-CLAMP,CLAMP))+1e-8).mean()

        # ---- InfoNCE ----
        ssl = info_nce(u1[u],u2[u],SSL_TEMP) + info_nce(i1[p],i2[p],SSL_TEMP)

        loss = bpr + SSL_WEIGHT*ssl

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    return total/len(loader)

# =========================================================
# EVALUATE
# =========================================================
@torch.no_grad()
def evaluate(model, adj, train_dict, test_dict, user_feats, item_feats):
    model.eval()
    u_emb,i_emb = model.propagate(adj, user_feats, item_feats)

    u_emb = F.normalize(u_emb, dim=1)
    i_emb = F.normalize(i_emb, dim=1)

    metrics = {k:{"recall":[],"ndcg":[]} for k in TOPKS}
    users = list(test_dict.keys())

    for i in tqdm(range(0,len(users),EVAL_BATCH), desc="Ranking"):
        batch = users[i:i+EVAL_BATCH]
        scores = torch.matmul(u_emb[batch], i_emb.T)

        for r,u in enumerate(batch):
            scores[r,list(train_dict[u])] = -1e9

        for k in TOPKS:
            topk = torch.topk(scores,k,dim=1).indices.cpu().numpy()
            for r,u in enumerate(batch):
                pos = test_dict[u]
                hits = [1 if x in pos else 0 for x in topk[r]]

                recall = sum(hits)/len(pos)
                dcg = sum(hits[j]/np.log2(j+2) for j in range(len(hits)))
                idcg = sum(1/np.log2(j+2) for j in range(min(len(pos),k)))
                ndcg = dcg/idcg if idcg>0 else 0

                metrics[k]["recall"].append(recall)
                metrics[k]["ndcg"].append(ndcg)

    return metrics

# =========================================================
# MAIN
# =========================================================
def main():
    path="/kaggle/input/yelp-dataset"

    interactions,n_users,n_items,u_feat,i_feat = load_yelp_with_features(path)
    train,test = split_data(interactions)

    train_dict,test_dict = defaultdict(set),defaultdict(set)
    for u,i in train: train_dict[u].add(i)
    for u,i in test:  test_dict[u].add(i)

    adj = build_adj(interactions,n_users,n_items)

    user_feats = torch.tensor(u_feat, device=DEVICE)
    item_feats = torch.tensor(i_feat, device=DEVICE)

    loader = DataLoader(TrainDataset(train), batch_size=BATCH_SIZE, shuffle=True)

    model = HybridLightGCN(n_users,n_items,user_feats.size(1),item_feats.size(1)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCHS):
        loss = train_epoch(model,loader,adj,opt,user_feats,item_feats)
        print(f"Epoch {e+1} Loss {loss:.4f}")

    metrics = evaluate(model,adj,train_dict,test_dict,user_feats,item_feats)

    print("\nFINAL METRICS")
    for k in TOPKS:
        print(f"Recall@{k}: {np.mean(metrics[k]['recall']):.4f}")
        print(f"NDCG@{k}:   {np.mean(metrics[k]['ndcg']):.4f}")

if __name__=="__main__":
    main()
