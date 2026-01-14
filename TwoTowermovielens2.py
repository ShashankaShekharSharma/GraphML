'''Explicit ratings  → implicit positives
        ↓
Offline LightGCN (collaborative structure)
        ↓
Text features (TF-IDF + SVD)
        ↓
Hybrid Two-Tower retrieval model
        ↓
Multi-negative BPR training
        ↓
Full-softmax Top-K evaluation


Precision@10: 0.0125
Recall@10:    0.0125
NDCG@10:      0.0055
HitRate@10:   0.0125
'''
import os, random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOVIES_PATH = "/kaggle/working/movielens_csv/movies.csv"
RATINGS_PATH = "/kaggle/working/movielens_csv/ratings.csv"
USERS_PATH = "/kaggle/working/movielens_csv/users.csv"

MAX_USERS, MAX_ITEMS = 2000, 2500
MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS = 5, 5
RATING_THRESHOLD = 3.5
SVD_TARGET_DIM, GCN_DIM, TOWER_DIM, HIDDEN_DIM = 128, 64, 128, 256
EPOCHS_GCN, EPOCHS_TOWER = 10000, 5000
LR_GCN, LR_TOWER = 1e-3, 5e-4
N_NEG, BATCH_SIZE, SEED = 5, 512, 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

movies_df = pd.read_csv(MOVIES_PATH)
ratings_df = pd.read_csv(RATINGS_PATH)
users_df = pd.read_csv(USERS_PATH)

ratings_df = ratings_df[ratings_df["rating"] >= RATING_THRESHOLD]
top_users = ratings_df["user_id"].value_counts().head(MAX_USERS).index
top_items = ratings_df["movie_id"].value_counts().head(MAX_ITEMS).index
ratings_df = ratings_df[ratings_df["user_id"].isin(top_users) & ratings_df["movie_id"].isin(top_items)]
ratings_df = ratings_df[ratings_df.groupby("user_id")["movie_id"].transform("count") >= MIN_USER_INTERACTIONS]
ratings_df = ratings_df[ratings_df.groupby("movie_id")["user_id"].transform("count") >= MIN_ITEM_INTERACTIONS]

user_ids, item_ids = ratings_df["user_id"].unique(), ratings_df["movie_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {m: i for i, m in enumerate(item_ids)}
ratings_df["u"], ratings_df["i"] = ratings_df["user_id"].map(user2idx), ratings_df["movie_id"].map(item2idx)
num_users, num_items = len(user_ids), len(item_ids)

user_hist = defaultdict(list)
for r in ratings_df.sort_values("u").itertuples(index=False):
    user_hist[r.u].append(r.i)

train_pairs, test_pairs = [], []
for u, items in user_hist.items():
    if len(items) < 2: train_pairs += [(u, i) for i in items]
    else: train_pairs += [(u, i) for i in items[:-1]]; test_pairs.append((u, items[-1]))

user_pos = defaultdict(set)
for u, i in train_pairs: user_pos[u].add(i)

movies_df = movies_df[movies_df["movie_id"].isin(item_ids)]
movies_df["i"] = movies_df["movie_id"].map(item2idx)
movies_df = movies_df.sort_values("i")

texts = [f"{r.title or ''} {r.genres.replace('|', ' ') if isinstance(r.genres, str) else ''}" for r in movies_df.itertuples(index=False)]
X = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english").fit_transform(texts)
X_svd = TruncatedSVD(n_components=SVD_TARGET_DIM, random_state=SEED).fit_transform(X).astype(np.float32)
X_svd /= np.linalg.norm(X_svd, axis=1, keepdims=True) + 1e-9
item_text_emb = torch.tensor(X_svd, device=DEVICE)
TEXT_DIM = item_text_emb.shape[1]

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb, self.item_emb = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
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
adj = torch.sparse_coo_tensor(torch.tensor([rows, cols], device=DEVICE), torch.ones(len(rows), device=DEVICE), (num_users + num_items, num_users + num_items))

gcn = LightGCN(num_users, num_items, GCN_DIM).to(DEVICE)
opt_gcn = torch.optim.Adam(gcn.parameters(), lr=LR_GCN)
for _ in range(EPOCHS_GCN):
    u_e, i_e = gcn(adj)
    batch_size = min(2000, len(train_pairs))
    batch = random.sample(train_pairs, batch_size)
    u_idx = torch.tensor([u for u, _ in batch], device=DEVICE)
    pos_i = torch.tensor([i for _, i in batch], device=DEVICE)
    neg_i = torch.randint(0, num_items, (batch_size,), device=DEVICE)
    loss = -torch.log(torch.sigmoid((u_e[u_idx] * i_e[pos_i]).sum(1) - (u_e[u_idx] * i_e[neg_i]).sum(1)) + 1e-12).mean()
    opt_gcn.zero_grad(); loss.backward(); opt_gcn.step()

gcn_user_emb, gcn_item_emb = gcn(adj)
gcn_user_emb, gcn_item_emb = gcn_user_emb.detach(), gcn_item_emb.detach()



class HybridTwoTower(nn.Module):
    def __init__(self, text_dim, gcn_dim):
        super().__init__()
        self.item_tower = nn.Sequential(nn.Linear(text_dim + gcn_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, TOWER_DIM))
        self.user_tower = nn.Sequential(nn.Linear(gcn_dim + TOWER_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, TOWER_DIM))

    def encode_items(self):
        return F.normalize(self.item_tower(torch.cat([item_text_emb, gcn_item_emb], dim=1)), dim=1)

    def encode_users(self, users, item_emb):
        hist = []
        for u in users:
            items = list(user_pos[u])
            hist.append(item_emb[items].mean(0) if items else torch.zeros(TOWER_DIM, device=DEVICE))
        fused = torch.cat([gcn_user_emb[users], torch.stack(hist)], dim=1)
        return F.normalize(self.user_tower(fused), dim=1)

def bpr_multi(pos, neg):
    return -torch.log(torch.sigmoid(pos.unsqueeze(1) - neg) + 1e-12).mean()

model = HybridTwoTower(TEXT_DIM, GCN_DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR_TOWER)
users = list(user_pos.keys())
for _ in range(EPOCHS_TOWER):
    random.shuffle(users)
    for i in range(0, len(users), BATCH_SIZE):
        batch = users[i:i+BATCH_SIZE]
        i_emb = model.encode_items()
        u_emb = model.encode_users(batch, i_emb)
        pos = torch.tensor([random.choice(list(user_pos[u])) for u in batch], device=DEVICE)
        neg = torch.randint(0, num_items, (len(batch), N_NEG), device=DEVICE)
        loss = bpr_multi((u_emb * i_emb[pos]).sum(1), (u_emb.unsqueeze(1) * i_emb[neg]).sum(2))
        opt.zero_grad(); loss.backward(); opt.step()

TOPK, TEST_BATCH = 10, 256
def ndcg_at_k(rank, gt): return 1.0 / np.log2(rank.index(gt) + 2) if gt in rank else 0.0

model.eval()
with torch.no_grad():
    item_emb = model.encode_items()
    precisions, recalls, ndcgs, hits = [], [], [], []
    for i in range(0, len(test_pairs), TEST_BATCH):
        block = test_pairs[i:i+TEST_BATCH]
        users, gts = [u for u, _ in block], [gt for _, gt in block]
        u_emb = model.encode_users(users, item_emb)
        scores = u_emb @ item_emb.t()
        for idx, u in enumerate(users): scores[idx, list(user_pos[u])] = -1e9
        _, topk = torch.topk(scores, TOPK, dim=1)
        topk = topk.cpu().tolist()
        for rank, gt in zip(topk, gts):
            h = int(gt in rank)
            precisions.append(h); recalls.append(h); hits.append(h); ndcgs.append(ndcg_at_k(rank, gt))


print(f"Precision@{TOPK}: {np.mean(precisions):.4f}")
print(f"Recall@{TOPK}:    {np.mean(recalls):.4f}")
print(f"NDCG@{TOPK}:      {np.mean(ndcgs):.4f}")
print(f"HitRate@{TOPK}:   {np.mean(hits):.4f}")