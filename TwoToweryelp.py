'''
The model learns user–item representations from interaction graphs (LightGCN), enriches item representations with textual content (TF-IDF + SVD), and then trains a two-tower neural ranking model using multi-negative BPR loss to recommend items.
'''

'''Precision@10: 0.3373
Recall@10:    0.3373
NDCG@10:      0.1577
HitRate@10:   0.3373'''


import os, json, random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "//kaggle/input/yelp-dataset"
REVIEW_PATH = os.path.join(DATA_DIR, "yelp_academic_dataset_review.json")
BUSINESS_PATH = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")

MAX_RAW_REVIEWS = 80_000
MAX_USERS = 2000
MAX_ITEMS = 2500
MAX_REVIEWS_PER_BUSINESS = 40
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
RATING_THRESHOLD = 3.5
SVD_TARGET_DIM = 128
GCN_DIM = 64
TOWER_DIM = 128
HIDDEN_DIM = 256
EPOCHS_GCN = 10000
EPOCHS_TOWER = 10000
LR_GCN = 1e-3
LR_TOWER = 5e-4
N_NEG = 5
BATCH_SIZE = 512
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

business_df = pd.read_json(BUSINESS_PATH, lines=True)

rev_records = []
with open(REVIEW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["stars"] >= RATING_THRESHOLD:
            rev_records.append({
                "user_id": r["user_id"],
                "business_id": r["business_id"],
                "text": r.get("text", "")[:1000]
            })
        if len(rev_records) >= MAX_RAW_REVIEWS:
            break

review_df = pd.DataFrame(rev_records)
review_df = review_df[review_df["business_id"].isin(set(business_df["business_id"]))]

top_users = review_df["user_id"].value_counts().head(MAX_USERS).index
top_items = review_df["business_id"].value_counts().head(MAX_ITEMS).index

review_df = review_df[
    review_df["user_id"].isin(top_users) &
    review_df["business_id"].isin(top_items)
]

review_df = review_df[review_df.groupby("user_id")["business_id"].transform("count") >= MIN_USER_INTERACTIONS]
review_df = review_df[review_df.groupby("business_id")["user_id"].transform("count") >= MIN_ITEM_INTERACTIONS]

user_ids = review_df["user_id"].unique()
item_ids = review_df["business_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {b: i for i, b in enumerate(item_ids)}

review_df["u"] = review_df["user_id"].map(user2idx)
review_df["i"] = review_df["business_id"].map(item2idx)

num_users = len(user_ids)
num_items = len(item_ids)

user_hist = defaultdict(list)
for r in review_df.sort_values("u").itertuples(index=False):
    user_hist[r.u].append(r.i)

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

business_df = business_df[business_df["business_id"].isin(item_ids)]
business_df["i"] = business_df["business_id"].map(item2idx)
business_df = business_df.sort_values("i")

rev_group = defaultdict(list)
for r in review_df.itertuples(index=False):
    rev_group[r.business_id].append(r.text)

texts = []
for row in business_df.itertuples(index=False):
    reviews = " ".join(rev_group[row.business_id][:MAX_REVIEWS_PER_BUSINESS])
    texts.append(" ".join([str(row.name), str(row.categories), str(row.city), reviews]))

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
X = tfidf.fit_transform(texts)
svd = TruncatedSVD(n_components=SVD_TARGET_DIM, random_state=SEED)
X_svd = svd.fit_transform(X).astype(np.float32)
X_svd /= np.linalg.norm(X_svd, axis=1, keepdims=True) + 1e-9

item_text_emb = torch.tensor(X_svd, device=DEVICE)
TEXT_DIM = item_text_emb.shape[1]

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

indices = torch.tensor([rows, cols], device=DEVICE)
values = torch.ones(len(rows), device=DEVICE)
adj = torch.sparse_coo_tensor(indices, values, (num_users + num_items, num_users + num_items))

gcn = LightGCN(num_users, num_items, GCN_DIM).to(DEVICE)
opt_gcn = torch.optim.Adam(gcn.parameters(), lr=LR_GCN)

for _ in range(EPOCHS_GCN):
    u_emb, i_emb = gcn(adj)
    batch_size = min(2000, len(train_pairs))
    batch = random.sample(train_pairs, batch_size)
    u_idx = torch.tensor([u for u, _ in batch], device=DEVICE)
    pos_i = torch.tensor([i for _, i in batch], device=DEVICE)
    neg_i = torch.randint(0, num_items, (batch_size,), device=DEVICE)
    loss = -torch.log(torch.sigmoid((u_emb[u_idx] * i_emb[pos_i]).sum(1) - (u_emb[u_idx] * i_emb[neg_i]).sum(1)) + 1e-12).mean()
    opt_gcn.zero_grad()
    loss.backward()
    opt_gcn.step()

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
        hist = torch.stack(hist)
        return F.normalize(self.user_tower(torch.cat([gcn_user_emb[users], hist], dim=1)), dim=1)

def bpr_multi(pos, neg):
    return -torch.log(torch.sigmoid(pos.unsqueeze(1) - neg) + 1e-12).mean()

model = HybridTwoTower(TEXT_DIM, GCN_DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR_TOWER)
users_list = list(user_pos.keys())

for _ in range(EPOCHS_TOWER):
    random.shuffle(users_list)
    for i in range(0, len(users_list), BATCH_SIZE):
        batch = users_list[i:i+BATCH_SIZE]
        item_emb = model.encode_items()
        user_emb = model.encode_users(batch, item_emb)
        pos = torch.tensor([random.choice(list(user_pos[u])) for u in batch], device=DEVICE)
        neg = torch.randint(0, num_items, (len(batch), N_NEG), device=DEVICE)
        loss = bpr_multi((user_emb * item_emb[pos]).sum(1), (user_emb.unsqueeze(1) * item_emb[neg]).sum(2))
        opt.zero_grad()
        loss.backward()
        opt.step()

TOPK, TEST_BATCH = 10, 256
def ndcg_at_k(rank, gt):
    return 1.0 / np.log2(rank.index(gt) + 2) if gt in rank else 0.0

model.eval()
with torch.no_grad():
    item_emb = model.encode_items()
    precisions, ndcgs = [], []
    for i in range(0, len(test_pairs), TEST_BATCH):
        block = test_pairs[i:i+TEST_BATCH]
        u_batch, gts = [u for u, _ in block], [gt for _, gt in block]
        u_embs = model.encode_users(u_batch, item_emb)
        scores = u_embs @ item_emb.t()
        for idx, u in enumerate(u_batch):
            scores[idx, list(user_pos[u])] = -1e9
        _, topk = torch.topk(scores, TOPK, dim=1)
        topk = topk.cpu().tolist()
        for rank, gt in zip(topk, gts):
            hit = int(gt in rank)
            precisions.append(hit)
            ndcgs.append(ndcg_at_k(rank, gt))

print(f"Precision@{TOPK}: {np.mean(precisions):.4f}\nRecall@{TOPK}:    {np.mean(precisions):.4f}\nNDCG@{TOPK}:      {np.mean(ndcgs):.4f}\nHitRate@{TOPK}:   {np.mean(precisions):.4f}")


