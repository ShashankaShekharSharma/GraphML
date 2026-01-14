'''three-stage hybrid recommender system on MovieLens:
- Implicit feedback construction from ratings
- Offline LightGCN to learn collaborative embeddings
- Hybrid two-tower neural ranker that fuses:
    * collaborative signals (GCN)
    * content signals (movie title + genres)
- Proper multi-item evaluation per user with ranking metrics'''

'''Precision@10: 0.0335
Recall@10:    0.0080
NDCG@10:      0.0322
HitRate@10:   0.2765'''

import random
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

MAX_USERS = 2000
MAX_ITEMS = 2500
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
RATING_THRESHOLD = 4.0

SVD_TARGET_DIM = 128
GCN_DIM = 64
TOWER_DIM = 128
HIDDEN_DIM = 256
EPOCHS_GCN = 300
EPOCHS_TOWER = 80
LR_GCN = 1e-3
LR_TOWER = 5e-4
WEIGHT_DECAY = 1e-4
N_NEG = 20
BATCH_SIZE = 512
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

movies_df = pd.read_csv(MOVIES_PATH)
ratings_df = pd.read_csv(RATINGS_PATH)
ratings_df = ratings_df[ratings_df["rating"] >= RATING_THRESHOLD]

top_users = ratings_df["user_id"].value_counts().head(MAX_USERS).index
top_items = ratings_df["movie_id"].value_counts().head(MAX_ITEMS).index

ratings_df = ratings_df[
    ratings_df["user_id"].isin(top_users) &
    ratings_df["movie_id"].isin(top_items)
]

ratings_df = ratings_df[ratings_df.groupby("user_id")["movie_id"].transform("count") >= MIN_USER_INTERACTIONS]
ratings_df = ratings_df[ratings_df.groupby("movie_id")["user_id"].transform("count") >= MIN_ITEM_INTERACTIONS]

user_ids = ratings_df["user_id"].unique()
item_ids = ratings_df["movie_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {m: i for i, m in enumerate(item_ids)}

ratings_df["u"] = ratings_df["user_id"].map(user2idx)
ratings_df["i"] = ratings_df["movie_id"].map(item2idx)

num_users = len(user_ids)
num_items = len(item_ids)

user_hist = defaultdict(list)
for r in ratings_df.itertuples(index=False):
    user_hist[r.u].append(r.i)

train_pairs, test_pairs = [], []
for u, items in user_hist.items():
    random.shuffle(items)
    split = max(1, int(0.8 * len(items)))
    train_pairs += [(u, i) for i in items[:split]]
    test_pairs += [(u, i) for i in items[split:]]

user_pos = defaultdict(set)
for u, i in train_pairs:
    user_pos[u].add(i)

movies_df = movies_df[movies_df["movie_id"].isin(item_ids)]
movies_df["i"] = movies_df["movie_id"].map(item2idx)
movies_df = movies_df.sort_values("i")

texts = []
for r in movies_df.itertuples(index=False):
    texts.append(f"{r.title} {r.genres.replace('|',' ')}")

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), stop_words="english")
X = tfidf.fit_transform(texts)
svd = TruncatedSVD(n_components=SVD_TARGET_DIM, random_state=SEED)
X = svd.fit_transform(X).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
item_text_emb = torch.tensor(X, device=DEVICE)

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
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=DEVICE)
        out = (torch.stack(embs) * weights.view(-1, 1, 1)).sum(0)
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
    loss = -torch.log(torch.sigmoid((u_emb[u]*i_emb[pi]).sum(1) - (u_emb[u]*i_emb[ni]).sum(1))).mean()
    opt_gcn.zero_grad()
    loss.backward()
    opt_gcn.step()

gcn_user_emb, gcn_item_emb = gcn(adj)
gcn_user_emb, gcn_item_emb = gcn_user_emb.detach(), gcn_item_emb.detach()

class HybridTwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_tower = nn.Sequential(
            nn.Linear(SVD_TARGET_DIM + GCN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, TOWER_DIM)
        )
        self.user_tower = nn.Sequential(
            nn.Linear(GCN_DIM + TOWER_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, TOWER_DIM)
        )

    def encode_items(self):
        return F.normalize(self.item_tower(torch.cat([item_text_emb, gcn_item_emb], 1)), dim=1)

    def encode_users(self, users, item_emb):
        hist = torch.stack([item_emb[list(user_pos[u])].mean(0) for u in users])
        return F.normalize(self.user_tower(torch.cat([gcn_user_emb[users], hist], 1)), dim=1)

def bpr(pos, neg):
    return -torch.log(torch.sigmoid(pos.unsqueeze(1) - neg)).mean()

model = HybridTwoTower().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR_TOWER, weight_decay=WEIGHT_DECAY)

users = list(user_pos.keys())
popular_items = torch.tensor(ratings_df["i"].value_counts().index.values, device=DEVICE)

for _ in range(EPOCHS_TOWER):
    random.shuffle(users)
    item_emb = model.encode_items().detach()
    for i in range(0, len(users), BATCH_SIZE):
        batch = users[i:i+BATCH_SIZE]
        ue = model.encode_users(batch, item_emb)
        pos = torch.tensor([random.choice(list(user_pos[u])) for u in batch], device=DEVICE)
        neg = popular_items[torch.randint(0, len(popular_items), (len(batch), N_NEG), device=DEVICE)]
        loss = bpr((ue * item_emb[pos]).sum(1), (ue.unsqueeze(1) * item_emb[neg]).sum(2))
        opt.zero_grad()
        loss.backward()
        opt.step()

TOPK = 10
def ndcg_at_k(ranked, ground_truth):
    dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(ranked) if item in ground_truth])
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), TOPK)))
    return dcg / idcg if idcg > 0 else 0.0

model.eval()
with torch.no_grad():
    item_emb = model.encode_items()
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_by_user = defaultdict(set)
    for u, i in test_pairs:
        test_by_user[u].add(i)

    for u, gt_items in test_by_user.items():
        user_emb = model.encode_users([u], item_emb)
        scores = (user_emb @ item_emb.T).squeeze()
        scores[list(user_pos[u])] = -1e9
        topk = torch.topk(scores, TOPK).indices.tolist()
        hits_k = len(set(topk) & gt_items)
        precisions.append(hits_k / TOPK)
        recalls.append(hits_k / len(gt_items))
        hits.append(1 if hits_k > 0 else 0)
        ndcgs.append(ndcg_at_k(topk, gt_items))

print(f"Precision@{TOPK}: {np.mean(precisions):.4f}")
print(f"Recall@{TOPK}:    {np.mean(recalls):.4f}")
print(f"NDCG@{TOPK}:      {np.mean(ndcgs):.4f}")
print(f"HitRate@{TOPK}:   {np.mean(hits):.4f}")