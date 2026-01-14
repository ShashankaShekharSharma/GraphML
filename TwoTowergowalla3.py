'''
Implicit check-ins
      ↓
 LightGCN (collaborative signal)
      ↓
 Temporal Hybrid Two-Tower
      ↓
 FAISS hard-negative training + retrieval
      ↓
 Single-GT next-location evaluation

Precision@20: 0.1054
Recall@20:    0.1054
NDCG@20:      0.0479
HitRate@20:   0.1054
'''
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GOWALLA_PATH = "/kaggle/working/gowalla_csv/checkins.csv"

MAX_USERS = 3000
MAX_ITEMS = 4000
MAX_HIST  = 40
RECENT_K  = 10
GCN_DIM = 64
GEO_DIM = 32
TOWER_DIM = 128
HIDDEN_DIM = 256
EPOCHS_GCN = 200
EPOCHS_TOWER = 80
LR_GCN = 1e-3
LR_TOWER = 5e-4
WEIGHT_DECAY = 1e-5
N_NEG = 20
BATCH_SIZE = 512
TOPK = 20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv(GOWALLA_PATH)
df = df.rename(columns={
    "location_id": "item_id",
    "check_in_time": "timestamp",
    "latitude": "lat",
    "longitude": "lon"
})
df = df.sort_values("timestamp")

top_users = df["user_id"].value_counts().head(MAX_USERS).index
top_items = df["item_id"].value_counts().head(MAX_ITEMS).index
df = df[df["user_id"].isin(top_users) & df["item_id"].isin(top_items)]

user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
item2idx = {i: j for j, i in enumerate(df["item_id"].unique())}
df["u"] = df["user_id"].map(user2idx)
df["i"] = df["item_id"].map(item2idx)

num_users = len(user2idx)
num_items = len(item2idx)

user_hist = defaultdict(list)
for r in df.itertuples(index=False):
    user_hist[r.u].append(r.i)

for u in user_hist:
    user_hist[u] = user_hist[u][-MAX_HIST:]

train_pairs, test_pairs = [], []
for u, items in user_hist.items():
    if len(items) > 1:
        train_pairs += [(u, i) for i in items[:-1]]
        test_pairs.append((u, items[-1]))
    else:
        train_pairs.append((u, items[0]))

user_pos = defaultdict(set)
for u, i in train_pairs:
    user_pos[u].add(i)

item_coords = np.zeros((num_items, 2), dtype=np.float32)
for r in df.itertuples(index=False):
    item_coords[r.i] = [r.lat, r.lon]

item_coords /= np.linalg.norm(item_coords, axis=1, keepdims=True) + 1e-9
item_feat = torch.tensor(item_coords, device=DEVICE)

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
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

rows, cols = [], []
for u, i in train_pairs:
    rows += [u, num_users + i]
    cols += [num_users + i, u]

adj = torch.sparse_coo_tensor(
    torch.tensor([rows, cols], device=DEVICE),
    torch.ones(len(rows), device=DEVICE),
    (num_users + num_items, num_users + num_items)
)
adj = torch.sparse.softmax(adj, dim=1)

gcn = LightGCN(num_users, num_items, GCN_DIM).to(DEVICE)
opt_gcn = torch.optim.Adam(gcn.parameters(), lr=LR_GCN)

for _ in range(EPOCHS_GCN):
    u_emb, i_emb = gcn(adj)
    batch = random.sample(train_pairs, min(3000, len(train_pairs)))
    u = torch.tensor([x[0] for x in batch], device=DEVICE)
    pi = torch.tensor([x[1] for x in batch], device=DEVICE)
    ni = []
    for uu in u.tolist():
        ni.append(random.choice(list(set(range(num_items)) - user_pos[uu])))
    ni = torch.tensor(ni, device=DEVICE)
    loss = -torch.log(torch.sigmoid((u_emb[u] * i_emb[pi]).sum(1) - (u_emb[u] * i_emb[ni]).sum(1)) + 1e-12).mean()
    opt_gcn.zero_grad()
    loss.backward()
    opt_gcn.step()

gcn_user_emb, gcn_item_emb = gcn(adj)
gcn_user_emb, gcn_item_emb = gcn_user_emb.detach(), gcn_item_emb.detach()

class HybridTwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_proj = nn.Linear(2, GEO_DIM)
        self.item_tower = nn.Sequential(
            nn.Linear(GCN_DIM + GEO_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, TOWER_DIM)
        )
        self.user_tower = nn.Sequential(
            nn.Linear(GCN_DIM + TOWER_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, TOWER_DIM)
        )

    def encode_items(self):
        geo = F.relu(self.geo_proj(item_feat))
        x = torch.cat([geo, gcn_item_emb], dim=1)
        return F.normalize(self.item_tower(x), dim=1)

    def encode_users(self, users, item_emb):
        hists = []
        for u in users:
            items = list(user_pos[u])[-RECENT_K:]
            w = torch.linspace(1.0, 0.2, steps=len(items)).to(DEVICE)
            w = w / w.sum()
            hists.append((item_emb[items] * w.unsqueeze(1)).sum(0))
        hist = torch.stack(hists)
        fused = torch.cat([gcn_user_emb[users], hist], dim=1)
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
        item_emb = model.encode_items()
        index = faiss.IndexFlatIP(TOWER_DIM)
        index.add(item_emb.detach().cpu().numpy().astype(np.float32))
        ue = model.encode_users(batch, item_emb)
        pos = torch.tensor([random.choice(list(user_pos[u])) for u in batch], device=DEVICE)
        _, hard = index.search(ue.detach().cpu().numpy(), 200)
        neg = []
        for u, h in zip(batch, hard):
            h = [x for x in h if x not in user_pos[u]]
            neg.append(random.sample(h, N_NEG))
        neg = torch.tensor(neg, device=DEVICE)
        pos_scores = (ue * item_emb[pos]).sum(1)
        neg_scores = (ue.unsqueeze(1) * item_emb[neg]).sum(2)
        loss = bpr_loss(pos_scores, neg_scores)
        opt.zero_grad()
        loss.backward()
        opt.step()

def ndcg(rank, gt):
    return 1.0 / np.log2(rank.index(gt) + 2) if gt in rank else 0.0

precisions, recalls, ndcgs, hits = [], [], [], []
model.eval()
with torch.no_grad():
    item_emb = model.encode_items()
    index = faiss.IndexFlatIP(TOWER_DIM)
    index.add(item_emb.cpu().numpy().astype(np.float32))
    for u, gt in test_pairs:
        ue = model.encode_users([u], item_emb)
        _, recs = index.search(ue.cpu().numpy().astype(np.float32), TOPK * 3)
        rank = []
        for r in recs[0]:
            if r not in user_pos[u]:
                rank.append(r)
            if len(rank) == TOPK: break
        hit = gt in rank
        precisions.append(hit); recalls.append(hit); hits.append(hit); ndcgs.append(ndcg(rank, gt))


print(f"Precision@{TOPK}: {np.mean(precisions):.4f}")
print(f"Recall@{TOPK}:    {np.mean(recalls):.4f}")
print(f"NDCG@{TOPK}:      {np.mean(ndcgs):.4f}")
print(f"HitRate@{TOPK}:   {np.mean(hits):.4f}")