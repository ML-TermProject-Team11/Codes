import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn


meta = pd.read_parquet("artifacts/items_meta_for_rec.parquet")

tmdb2director = dict(zip(meta["tmdbId"], meta["director_id"]))
tmdb2genres = dict(zip(meta["tmdbId"], meta["genre_ids"]))

class SASRecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tmdb2director: dict, tmdb2genres: dict, item2idx: dict, max_len=50, max_num_genres=6):
        self.max_len = max_len
        self.max_num_genres = max_num_genres
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

    def __getitem__(self, idx: int):
        seq_tmdb = self.user_seqs[idx][-self.max_len:]
        labels_tmdb = seq_tmdb[1:]
        seq_tmdb = seq_tmdb[:-1]

        pad_len = self.max_len - len(seq_tmdb)
        seq_tmdb  = [0]*pad_len + seq_tmdb
        labels_tmdb = [0]*pad_len + labels_tmdb
        mask = [0]*pad_len + [1]*len(labels_tmdb[-(self.max_len - pad_len):])

        # meta lookup (tmdbId 기준)
        dir_seq   = [self.tmdb2director.get(t, 0) for t in seq_tmdb]
        genre_seq = [self.tmdb2genres.get(t, [0]) for t in seq_tmdb]

        # 장르 고정 길이 패딩
        padded_genres = []
        for g in genre_seq:
            g = g[:self.max_num_genres]
            g += [0] * (self.max_num_genres - len(g))
            padded_genres.append(g)

        # 인덱싱: seq, labels 모두 item2idx로 변환
        seq_idx    = [self.item2idx.get(t, 0) for t in seq_tmdb]
        labels_idx = [self.item2idx.get(t, 0) for t in labels_tmdb]

        return (
            torch.LongTensor(seq_idx),        # [L]
            torch.LongTensor(labels_idx),     # [L]  ← 수정
            torch.LongTensor(dir_seq),        # [L]
            torch.LongTensor(padded_genres),  # [L, G]
            torch.FloatTensor(mask)           # [L]
        )


class MetaEmbedding(nn.Module):
    """
    director id : 단일 범주형 → Embedding
    genre ids : multi-label → Embedding + mean pooling
    layer norm 으로 scale 정규화
    출력 shape 은 SASRec의 item embedding 과 동일 ([B, L, d])
    """
    def __init__(self, n_directors: int, n_genres: int, d_model):
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
                 hidden_dim=256, n_heads=4, n_layers=4,
                 max_len=100, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.meta_emb = MetaEmbedding(n_directors, n_genres, hidden_dim)

        # feature fusion (concat + projection)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, n_heads, hidden_dim * 4,
                dropout, batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, seq_items, seq_directors, seq_genres):
        B, L = seq_items.size()
        pos = torch.arange(L, device=seq_items.device).unsqueeze(0).expand(B, L)

        item_emb = self.item_emb(seq_items) + self.pos_emb(pos)
        meta_emb = self.meta_emb(seq_directors, seq_genres)

        # concat + projection + normalization
        # x = torch.cat([item_emb, meta_emb], dim=-1)
        # x = self.norm(self.fusion(x))

        x = torch.cat([item_emb, meta_emb], dim=-1)
        x = torch.nn.functional.gelu(self.fusion(x))
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)


        # causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=mask)

        logits = self.output(x)
        return logits, item_emb, meta_emb

