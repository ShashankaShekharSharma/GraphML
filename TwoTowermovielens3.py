'''User check-ins (time ordered)
        ↓
Temporal LightGCN (collaborative signal)
        ↓
Hybrid Two-Tower neural model
   ├── Item tower: GCN + geo features
   └── User tower: GCN + query-based attention over history
        ↓
Hard-negative BPR training
        ↓
FAISS ANN retrieval

Precision@20: 0.1407
Recall@20:    0.1407
NDCG@20:      0.0753
HitRate@20:   0.1407
'''

import random, math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GOWALLA_PATH = "/kaggle/working/gowalla_csv/checkins.csv"

MAX_USERS, MAX_ITEMS = 3000, 4000
MAX_HIST, RECENT_K = 40, 10
GCN_DIM, GEO_DIM, TOWER_DIM, HIDDEN_DIM = 64, 32, 128, 256
EPOCHS_GCN, EPOCHS_TOWER = 200, 80
LR_GCN, LR_TOWER, WEIGHT_DECAY = 1e-3, 5e-4, 1e-5
N_NEG, BATCH_SIZE, TOPK, SEED = 20, 512, 20, 42
TIME_DECAY_TAU = 7 * 24 * 3600

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df = pd.read_csv(GOWALLA_PATH)
df = df.rename(columns={"location_id": "item_id", "check_in_time": "timestamp", "latitude": "lat", "longitude": "lon"})
df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(np.int64) // 10**9
df = df.sort_values("timestamp")

top_users = df["user_id"].value_counts().head(MAX_USERS).index
top_items = df["item_id"].value_counts().head(MAX_ITEMS).index
df = df[df["user_id"].isin(top_users) & df["item_id"].isin(top_items)]

user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
item2idx = {i: j for j, i in enumerate(df["item_id"].unique())}
df["u"], df["i"] = df["user_id"].map(user2idx), df["item_id"].map(item2idx)
num_users, num_items = len(user2idx), len(item2idx)

user_hist = defaultdict(list)
for r in df.itertuples(index=False):
    user_hist[r.u].append((r.i, r.timestamp))

train_pairs, test_pairs = [], []
for u, seq in user_hist.items():
    items = [i for i, _ in seq[-MAX_HIST:]]
    if len(items) > 1:
        train_pairs += [(u, i) for i in items[:-1]]
        test_pairs.append((u, items[-1]))
    else:
        train_pairs.append((u, items[0]))

user_pos = defaultdict(set)
for u, i in train_pairs: user_pos[u].add(i)

item_coords = np.zeros((num_items, 2), dtype=np.float32)
for r in df.itertuples(index=False): item_coords[r.i] = [r.lat, r.lon]
item_coords /= np.linalg.norm(item_coords, axis=1, keepdims=True) + 1e-9
item_feat = torch.tensor(item_coords, device=DEVICE)

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb, self.item_emb = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, adj):
        emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [emb]
        for _ in range(2):
            emb = torch.sparse.mm(adj, emb)
            embs.append(emb)
        out = torch.stack(embs).mean(0)
        return out[:num_users], out[num_users:]

rows, cols, vals = [], [], []
t_max = df["timestamp"].max()
for r in df.itertuples(index=False):
    decay = math.exp(-(t_max - r.timestamp) / TIME_DECAY_TAU)
    rows += [r.u, num_users + r.i]; cols += [num_users + r.i, r.u]; vals += [decay, decay]

adj = torch.sparse_coo_tensor(torch.tensor([rows, cols], device=DEVICE), torch.tensor(vals, device=DEVICE), (num_users + num_items, num_users + num_items))
adj = torch.sparse.softmax(adj, dim=1)

gcn = LightGCN(num_users, num_items, GCN_DIM).to(DEVICE)
opt_gcn = torch.optim.Adam(gcn.parameters(), lr=LR_GCN)
for _ in range(EPOCHS_GCN):
    u_e, i_e = gcn(adj)
    batch = random.sample(train_pairs, min(3000, len(train_pairs)))
    u = torch.tensor([x[0] for x in batch], device=DEVICE)
    pi = torch.tensor([x[1] for x in batch], device=DEVICE)
    ni = torch.tensor([random.choice(list(set(range(num_items)) - user_pos[uu])) for uu in u.tolist()], device=DEVICE)
    loss = -torch.log(torch.sigmoid((u_e[u] * i_e[pi]).sum(1) - (u_e[u] * i_e[ni]).sum(1)) + 1e-12).mean()
    opt_gcn.zero_grad(); loss.backward(); opt_gcn.step()

gcn_user_emb, gcn_item_emb = gcn(adj)
gcn_user_emb, gcn_item_emb = gcn_user_emb.detach(), gcn_item_emb.detach()



class HybridTwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_proj = nn.Linear(2, GEO_DIM)
        self.item_tower = nn.Sequential(nn.Linear(GCN_DIM + GEO_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, TOWER_DIM))
        self.q_proj, self.k_proj = nn.Linear(GCN_DIM, TOWER_DIM), nn.Linear(TOWER_DIM, TOWER_DIM)
        self.user_tower = nn.Sequential(nn.Linear(GCN_DIM + TOWER_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, TOWER_DIM))

    def encode_items(self):
        geo = F.relu(self.geo_proj(item_feat))
        return F.normalize(self.item_tower(torch.cat([geo, gcn_item_emb], dim=1)), dim=1)

    def encode_users(self, users, item_emb):
        reps = []
        for u in users:
            items = list(user_pos[u])[-RECENT_K:]
            h = item_emb[items]
            q, k = self.q_proj(gcn_user_emb[u]), self.k_proj(h)
            alpha = F.softmax((q * k).sum(1), dim=0)
            reps.append((h * alpha.unsqueeze(1)).sum(0))
        fused = torch.cat([gcn_user_emb[users], torch.stack(reps)], dim=1)
        return F.normalize(self.user_tower(fused), dim=1)

def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos.unsqueeze(1) - neg) + 1e-12).mean()

model = HybridTwoTower().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR_TOWER, weight_decay=WEIGHT_DECAY)



users = list(user_pos.keys())
for _ in range(EPOCHS_TOWER):
    random.shuffle(users)
    for i in range(0, len(users), BATCH_SIZE):
        batch = users[i:i+BATCH_SIZE]
        i_emb = model.encode_items()
        index = faiss.IndexFlatIP(TOWER_DIM)
        index.add(i_emb.detach().cpu().numpy().astype(np.float32))
        ue = model.encode_users(batch, i_emb)
        pos = torch.tensor([random.choice(list(user_pos[u])) for u in batch], device=DEVICE)
        _, hard = index.search(ue.detach().cpu().numpy(), 200)
        neg = torch.tensor([[x for x in h if x not in user_pos[u]][:N_NEG] for u, h in zip(batch, hard)], device=DEVICE)
        loss = bpr_loss((ue * i_emb[pos]).sum(1), (ue.unsqueeze(1) * i_emb[neg]).sum(2))
        opt.zero_grad(); loss.backward(); opt.step()

def ndcg(rank, gt): return 1.0 / np.log2(rank.index(gt) + 2) if gt in rank else 0.0

precisions, recalls, ndcgs, hits = [], [], [], []
model.eval()
with torch.no_grad():
    i_emb = model.encode_items()
    index = faiss.IndexFlatIP(TOWER_DIM)
    index.add(i_emb.cpu().numpy().astype(np.float32))
    for u, gt in test_pairs:
        ue = model.encode_users([u], i_emb)
        _, recs = index.search(ue.cpu().numpy().astype(np.float32), TOPK * 3)
        rank = [r for r in recs[0] if r not in user_pos[u]][:TOPK]
        hit = gt in rank
        precisions.append(hit); recalls.append(hit); hits.append(hit); ndcgs.append(ndcg(rank, gt))


print(f"Precision@{TOPK}: {np.mean(precisions):.4f}")
print(f"Recall@{TOPK}:    {np.mean(recalls):.4f}")
print(f"NDCG@{TOPK}:      {np.mean(ndcgs):.4f}")
print(f"HitRate@{TOPK}:   {np.mean(hits):.4f}")