'''
- LightGCN → learns collaborative user/item embeddings from interaction graph
- Two-Tower hybrid model → fuses:
    user GCN embedding
    mean embedding of user’s historical items
    item GCN embedding
    item geographic features (lat/lon)
- FAISS Inner Product search → fast Top-K retrieval

Precision@20: 0.0611
Recall@20:    0.0611
NDCG@20:      0.0246
HitRate@20:   0.0611
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
GCN_DIM = 64
TOWER_DIM = 128
HIDDEN_DIM = 256
EPOCHS_GCN = 300
EPOCHS_TOWER = 100
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

user_ids = df["user_id"].unique()
item_ids = df["item_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {l: i for i, l in enumerate(item_ids)}

df["u"] = df["user_id"].map(user2idx)
df["i"] = df["item_id"].map(item2idx)

num_users = len(user_ids)
num_items = len(item_ids)

user_hist = defaultdict(list)
for r in df.itertuples(index=False):
    user_hist[r.u].append(r.i)

for u in user_hist:
    if len(user_hist[u]) > MAX_HIST:
        user_hist[u] = user_hist[u][-MAX_HIST:]

train_pairs, test_pairs = [], []
for u, items in user_hist.items():
    if len(items) < 2:
        train_pairs += [(u, i) for i in items]
    else:
        train_pairs += [(u, i) for i in items[:-1]]
        test_pairs.append((u, items[-1]))

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
        for _ in range(3):
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

gcn = LightGCN(num_users, num_items, GCN_DIM).to(DEVICE)
opt_gcn = torch.optim.Adam(gcn.parameters(), lr=LR_GCN)

for _ in range(EPOCHS_GCN):
    u_emb, i_emb = gcn(adj)
    batch = random.sample(train_pairs, min(2000, len(train_pairs)))
    u = torch.tensor([x[0] for x in batch], device=DEVICE)
    pi = torch.tensor([x[1] for x in batch], device=DEVICE)
    ni = torch.randint(0, num_items, (len(batch),), device=DEVICE)
    loss = -torch.log(torch.sigmoid((u_emb[u] * i_emb[pi]).sum(1) - (u_emb[u] * i_emb[ni]).sum(1)) + 1e-12).mean()
    opt_gcn.zero_grad()
    loss.backward()
    opt_gcn.step()

gcn_user_emb, gcn_item_emb = gcn(adj)
gcn_user_emb = gcn_user_emb.detach()
gcn_item_emb = gcn_item_emb.detach()

class HybridTwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_tower = nn.Sequential(
            nn.Linear(GCN_DIM + 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, TOWER_DIM)
        )
        self.user_tower = nn.Sequential(
            nn.Linear(GCN_DIM + TOWER_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, TOWER_DIM)
        )

    def encode_items(self):
        x = torch.cat([item_feat, gcn_item_emb], dim=1)
        return F.normalize(self.item_tower(x), dim=1)

    def encode_users(self, users, item_emb):
        hist = []
        for u in users:
            hist.append(item_emb[list(user_pos[u])].mean(0))
        hist = torch.stack(hist)
        fused = torch.cat([gcn_user_emb[users], hist], dim=1)
        return F.normalize(self.user_tower(fused), dim=1)

def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos.unsqueeze(1) - neg) + 1e-12).mean()

model = HybridTwoTower().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR_TOWER, weight_decay=WEIGHT_DECAY)

users = list(user_pos.keys())
popular_items = torch.tensor(df["i"].value_counts().index.values, device=DEVICE)

for _ in range(EPOCHS_TOWER):
    random.shuffle(users)
    item_emb = model.encode_items().detach()
    for i in range(0, len(users), BATCH_SIZE):
        batch = users[i:i+BATCH_SIZE]
        ue = model.encode_users(batch, item_emb)
        pos = torch.tensor([random.choice(list(user_pos[u])) for u in batch], device=DEVICE)
        neg = popular_items[torch.randint(0, len(popular_items), (len(batch), N_NEG), device=DEVICE)]
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
    item_emb_torch = model.encode_items()
    item_emb_np = item_emb_torch.cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatIP(TOWER_DIM)
    index.add(item_emb_np)
    for u, gt in test_pairs:
        ue = model.encode_users([u], item_emb_torch)
        ue_np = ue.cpu().numpy().astype(np.float32)
        _, recs = index.search(ue_np, TOPK * 3)
        rank = []
        for r in recs[0]:
            if r not in user_pos[u]:
                rank.append(r)
            if len(rank) == TOPK:
                break
        hit = gt in rank
        precisions.append(hit)
        recalls.append(hit)
        hits.append(hit)
        ndcgs.append(ndcg(rank, gt))


print(f"Precision@{TOPK}: {np.mean(precisions):.4f}")
print(f"Recall@{TOPK}:    {np.mean(recalls):.4f}")
print(f"NDCG@{TOPK}:      {np.mean(ndcgs):.4f}")
print(f"HitRate@{TOPK}:   {np.mean(hits):.4f}")