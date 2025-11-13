import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn

meta = pd.read_parquet("artifacts/items_meta_for_rec.parquet")

tmdb2director = dict(zip(meta["tmdbId"], meta["director_id"]))
tmdb2genres = dict(zip(meta["tmdbId"], meta["genre_ids"]))

class SASRecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tmdb2director: dict, tmdb2genres: dict, item2idx: dict, max_len=50):
        self.max_len = max_len
        self.user_seqs = []

        for uid, group in df.groupby("userId"):
            seq = list(group.sort_values("timestamp")["tmdbId"])
            if len(seq) < 2:
                continue
            self.user_seqs.append(seq)

        self.tmdb2director = tmdb2director
        self.tmdb2genres = tmdb2genres
        self.item2idx = item2idx

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, idx):
        seq_tmdb = self.user_seqs[idx][-self.max_len:]
        labels_tmdb = seq_tmdb[1:]
        seq_tmdb = seq_tmdb[:-1]

        pad_len = self.max_len - len(seq_tmdb)
        seq_tmdb = [0]*pad_len + seq_tmdb
        labels_tmdb = [0]*pad_len + labels_tmdb
        mask = [0]*pad_len + [1]*len(labels_tmdb[-(self.max_len - pad_len):])

        # meta lookup (tmdbId 기준)
        dir_seq = [self.tmdb2director.get(t, 0) for t in seq_tmdb]
        genre_seq = [self.tmdb2genres.get(t, [0]) for t in seq_tmdb]

        # id → index 변환 (item2idx)
        seq_idx = [self.item2idx.get(t, 0) for t in seq_tmdb]

        max_g = max(len(g) for g in genre_seq)
        padded_genres = [g + [0]*(max_g - len(g)) for g in genre_seq]

        return (
            torch.LongTensor(seq_idx),
            torch.LongTensor(labels_tmdb),
            torch.LongTensor(dir_seq),
            torch.LongTensor(padded_genres),
            torch.FloatTensor(mask)
        )

class MetaEmbedding(nn.Module):
    """
    director id : 단일 범주형 → Embedding
    genre ids : multi-label → Embedding + mean pooling
    layer norm 으로 scale 정규화
    출력 shape 은 SASRec의 item embedding 과 동일 ([B, L, d])
    """
    def __init__(self, n_directors, n_genres, d_model):
        super().__init__()
        self.director_emb = nn.Embedding(n_directors + 1, d_model, padding_idx=0)
        self.genre_emb = nn.Embedding(n_genres + 1, d_model, padding_idx=0)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, director_ids, genre_ids):
        # director_ids: [B, L]
        # genre_ids: [B, L, G]
        d = self.director_emb(director_ids)                 # [B, L, d]
        g = self.genre_emb(genre_ids)                       # [B, L, G, d]
        mask = (genre_ids != 0).float().unsqueeze(-1)       # [B, L, G, 1]
        g_sum = (g * mask).sum(dim=2)
        g_count = mask.sum(dim=2).clamp(min=1e-6)
        g_mean = g_sum / g_count                            # [B, L, d]
        x = (d + g_mean) / 2
        return self.layer_norm(x)                           # [B, L, d]
    


class SASRec(nn.Module):
    def __init__(self, num_items, n_directors, n_genres,
                 hidden_dim=128, n_heads=2, n_layers=2,
                 max_len=50, dropout=0.2):
        super().__init__()
        # 기본 embedding
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.meta_emb = MetaEmbedding(n_directors, n_genres, hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, n_heads,
                                       hidden_dim * 4, dropout,
                                       batch_first=True)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, seq_items, seq_directors, seq_genres):
        B, L = seq_items.size()
        pos = torch.arange(L, device=seq_items.device).unsqueeze(0).expand(B, L)

        # ID embedding
        item_emb = self.item_emb(seq_items) + self.pos_emb(pos)
        # Meta embedding
        meta_emb = self.meta_emb(seq_directors, seq_genres)

        # 결합: 평균 또는 concat + projection 둘 중 선택
        x = (item_emb + meta_emb) / 2                      # [B, L, d]
        x = self.layer_norm(x)

        # Causal Mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=mask)

        logits = self.output(x)                            # [B, L, num_items]
        return logits



df = pd.read_parquet("artifacts/netflix_ratings_for_sasrec.parquet")
meta = pd.read_parquet("artifacts/items_meta_for_rec.parquet")
item2idx = {tmdb: idx+1 for idx, tmdb in enumerate(meta["tmdbId"].unique())}

tmdb2director = dict(zip(meta["tmdbId"], meta["director_id"]))
tmdb2genres = dict(zip(meta["tmdbId"], meta["genre_ids"]))

ds = SASRecDataset(df.sample(50000), tmdb2director, tmdb2genres, item2idx, max_len=50)

sample_user = ds.user_seqs[0]
print("원본 tmdbId seq:", sample_user[-10:])

seq, labels, dirs, genres, mask = ds[0]
print("item seq:", seq[-10:])
print("director seq:", dirs[-10:])
print("mask sum:", mask.sum())



long_users = [u for u, g in df.groupby("userId") if len(g) > 30]
ds_long = SASRecDataset(df[df["userId"].isin(long_users)], tmdb2director, tmdb2genres, item2idx, max_len=50)

seq, labels, dirs, genres, mask = ds_long[0]
print("seq nonzero:", (seq>0).sum())
print("mask sum:", mask.sum())
print("sample directors:", dirs[-10:])


'''

seq, labels, dirs, genres, mask = ds[0]

print("item seq:", seq[:10])
print("director seq:", dirs[:10])
print("genres seq shape:", genres.shape)
print("mask sum:", mask.sum())



# 선택된 샘플의 userId 직접 확인
uid = list(df.groupby("userId").groups.keys())[0]
print("userId:", uid)
print("sequence length:", len(df[df["userId"]==uid]))

# 매핑된 tmdbId 존재 비율 확인
mapped = df["tmdbId"].isin(meta["tmdbId"])
print(f"매핑된 영화 비율: {mapped.mean()*100:.2f}%")




# 길이 30 이상인 유저 중 하나 보기
long_users = df["userId"].value_counts()
uid = long_users[long_users > 30].index[0]
user_df = df[df["userId"]==uid].sort_values("timestamp")
print(user_df.tail())

seq, labels, dirs, genres, mask = ds[long_users[long_users > 30].index.get_loc(uid)]
print(seq[-10:], dirs[-10:], mask.sum())



df_ids = set(df["tmdbId"].unique())
meta_ids = set(meta["tmdbId"].unique())

print(f"Netflix tmdbId 수: {len(df_ids):,}")
print(f"Meta tmdbId 수: {len(meta_ids):,}")
print(f"교집합: {len(df_ids & meta_ids):,} ({len(df_ids & meta_ids)/len(df_ids)*100:.2f}%)")




print("Netflix tmdbId dtype:", df["tmdbId"].dtype)
print("Meta tmdbId dtype:", meta["tmdbId"].dtype)

sample = list(df["tmdbId"].head(5))
print("Netflix sample:", sample)
for s in sample:
    if s in meta["tmdbId"].values:
        print(f"{s}: meta에 존재")
    else:
        print(f"{s}: meta에 없음 (dtype mismatch 가능성)")
'''