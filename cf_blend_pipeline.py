# cf_blend_pipeline.py
"""
Netflix 학습 → MovieLens 평가
- Memory-based CF (item-item, mean-centered cosine) + 장르 Jaccard 보정
- Model-based CF (Funk-SVD) 학습은 Netflix 중심행렬로 복원한 원평점으로 수행
- SVD fold-in: MovieLens 사용자 벡터/바이어스만 ridge 해법으로 추정
- 두 예측을 0~1 정규화 후 가중합(cf_score) → threshold 이상만 최종 출력
- 최종 컬럼: ["movieId","title","title_tmdb","directors","genres_merged",
              "release_year","original_language","cf_score","popularity","runtime"]
필요 파일 (project/artifacts/):
  netflix_cf_centered.npz
  netflix_user_means.npy
  netflix_item2idx.json
  netflix_user2idx.json
  movielens_cf_centered.npz
  movielens_user_means.npy
  movielens_item2idx.json
  movielens_user2idx.json
  movielens_content.parquet
  idmap_movielens_movie_to_tmdb.parquet
  idmap_netflix_movie_to_tmdb_enhanced_repaired.parquet
  tmdb_items_enriched_update.parquet
선택: crosswalk_tmdb_movielens_netflix.parquet (매핑 보조)
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

# -----------------------------
# I/O 유틸
# -----------------------------
def load_sparse(path: str) -> sparse.csr_matrix:
    m = sparse.load_npz(path)
    if not sparse.isspmatrix_csr(m):
        m = m.tocsr()
    return m

def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        # JSON-like list or "A|B|C"
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                if isinstance(v, list): return v
            except Exception:
                pass
        return [p.strip() for p in s.replace(",", "|").split("|") if p.strip()]
    return [str(x)]

def guess_col(cols, *keys):
    keys = [k.lower() for k in keys]
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keys):
            return c
    return None

# -----------------------------
# 매핑/메타 적재
# -----------------------------
@dataclass
class Crosswalk:
    ml_movie_to_tmdb: Dict[int, int]
    nf_movie_to_tmdb: Dict[int, int]
    tmdb_to_nf_movie: Dict[int, int]
    tmdb_to_ml_movie: Dict[int, int]

@dataclass
class TMDBMeta:
    df: pd.DataFrame               # keyed by tmdb_id
    genres: Dict[int, List[str]]   # tmdb_id -> list[str]
    directors: Dict[int, List[str]]

def load_crosswalk(artifacts_dir: str) -> Crosswalk:
    ml2tmdb = pd.read_parquet(os.path.join(artifacts_dir, "idmap_movielens_movie_to_tmdb.parquet"))
    nf2tmdb = pd.read_parquet(os.path.join(artifacts_dir, "idmap_netflix_movie_to_tmdb_enhanced_repaired.parquet"))

    # column inference
    ml_mid = guess_col(ml2tmdb.columns, "movie") or "movieId"
    ml_tmdb = guess_col(ml2tmdb.columns, "tmdb") or "tmdb_id"
    nf_mid = guess_col(nf2tmdb.columns, "movie") or "movieId"
    nf_tmdb = guess_col(nf2tmdb.columns, "tmdb") or "tmdb_id"

    ml2tm = (ml2tmdb[[ml_mid, ml_tmdb]]
             .dropna()
             .astype({ml_mid: int, ml_tmdb: int})
             .drop_duplicates())
    nf2tm = (nf2tmdb[[nf_mid, nf_tmdb]]
             .dropna()
             .astype({nf_mid: int, nf_tmdb: int})
             .drop_duplicates())

    ml_to_tm = dict(zip(ml2tm[ml_mid], ml2tm[ml_tmdb]))
    nf_to_tm = dict(zip(nf2tm[nf_mid], nf2tm[nf_tmdb]))

    tm_to_nf = {}
    for k, v in nf_to_tm.items():
        tm_to_nf[v] = k
    tm_to_ml = {}
    for k, v in ml_to_tm.items():
        tm_to_ml[v] = k

    return Crosswalk(ml_movie_to_tmdb=ml_to_tm,
                     nf_movie_to_tmdb=nf_to_tm,
                     tmdb_to_nf_movie=tm_to_nf,
                     tmdb_to_ml_movie=tm_to_ml)

def load_tmdb_meta(artifacts_dir: str) -> TMDBMeta:
    path = os.path.join(artifacts_dir, "tmdb_items_enriched_update.parquet")
    df = pd.read_parquet(path)

    # key inference
    tmdb_key = (guess_col(df.columns, "tmdb") 
                or guess_col(df.columns, "id")
                or "tmdb_id")
    df = df.rename(columns={tmdb_key: "tmdb_id"})
    df = df.drop_duplicates("tmdb_id")

    # column guesses
    title_col = guess_col(df.columns, "title") or "title"
    orig_lang = guess_col(df.columns, "original", "language") or "original_language"
    year_col = guess_col(df.columns, "release", "year") or "release_year"
    pop_col = guess_col(df.columns, "popularity") or "popularity"
    runtime_col = guess_col(df.columns, "runtime") or "runtime"

    # directors, genres diverse schema handling
    # directors can be list[str], string, or list[dict{name:..}]
    def extract_names(x):
        arr = ensure_list(x)
        out = []
        for e in arr:
            if isinstance(e, dict):
                n = e.get("name")
                if n: out.append(str(n))
            else:
                out.append(str(e))
        return [s for s in out if s]
    dir_col = None
    for cand in ["directors", "crew_directors", "director_names", "crew"]:
        if cand in df.columns:
            dir_col = cand
            break

    gen_col = None
    for cand in ["genres", "genre_names", "genre_ids", "genres_merged"]:
        if cand in df.columns:
            gen_col = cand
            break

    df["title_tmdb"] = df[title_col] if title_col in df.columns else None
    if orig_lang in df.columns:
        df["original_language"] = df[orig_lang]
    else:
        df["original_language"] = None
    if year_col in df.columns:
        df["release_year"] = df[year_col]
    else:
        df["release_year"] = None
    if pop_col in df.columns:
        df["popularity"] = df[pop_col]
    else:
        df["popularity"] = None
    if runtime_col in df.columns:
        df["runtime"] = df[runtime_col]
    else:
        df["runtime"] = None

    # directors map
    if dir_col is None:
        directors_map = {int(k): [] for k in df["tmdb_id"].values}
    else:
        directors_map = {int(r.tmdb_id): extract_names(r[dir_col]) for _, r in df[["tmdb_id", dir_col]].iterrows()}

    # genres map
    def extract_genres(x):
        arr = ensure_list(x)
        out = []
        for e in arr:
            if isinstance(e, dict):
                n = e.get("name")
                if n: out.append(str(n))
            else:
                out.append(str(e))
        return [s for s in out if s]
    if gen_col is None:
        genres_map = {int(k): [] for k in df["tmdb_id"].values}
    else:
        genres_map = {int(r.tmdb_id): extract_genres(r[gen_col]) for _, r in df[["tmdb_id", gen_col]].iterrows()}

    keep = ["tmdb_id", "title_tmdb", "original_language", "release_year", "popularity", "runtime"]
    return TMDBMeta(df=df[keep], genres=genres_map, directors=directors_map)

# -----------------------------
# CF 유사도(아이템-아이템)
# -----------------------------
def topk_cosine_for_targets(X_csr: sparse.csr_matrix,
                            target_cols: List[int],
                            K: int,
                            min_common: int,
                            shrink_b: float) -> sparse.csr_matrix:
    """
    X: users x items, mean-centered (CSR)
    target_cols: 유사도 열을 만들 NF 아이템 인덱스 리스트
    반환: target 열들만 채워진 item-item topK CSR (float32)
    """
    X_csr = X_csr.astype(np.float32, copy=False)
    Xc = X_csr.tocsc()

    # L2 정규화
    col_norms = np.sqrt(Xc.power(2).sum(axis=0)).A1.astype(np.float32)
    inv = np.zeros_like(col_norms, dtype=np.float32)
    nz = col_norms > 0
    inv[nz] = 1.0 / col_norms[nz]
    Xn = (Xc @ sparse.diags(inv)).astype(np.float32)

    # 공평가자 카운트용 이진 행렬
    Xbin = Xc.copy().astype(np.float32)
    Xbin.data[:] = 1.0

    n_items = Xc.shape[1]
    rows, cols, data = [], [], []

    for j in target_cols:
        # cos(i,j)와 공평가자 수 n_ij를 모두 sparse로 계산
        cos_sv = (Xn.T @ Xn[:, j]).tocoo()                 # i행 인덱스와 값만 존재
        cnt_sv = (Xbin.T @ Xbin[:, j]).tocoo()
        cnt_map = {int(r): float(v) for r, v in zip(cnt_sv.row, cnt_sv.data)}

        pairs = []
        for i, s in zip(cos_sv.row, cos_sv.data):
            if i == j:
                continue
            n_ij = cnt_map.get(int(i), 0.0)
            if n_ij < float(min_common):
                continue
            w = float(s) * (n_ij / (n_ij + float(shrink_b)))
            if w != 0.0:
                pairs.append((int(i), np.float32(w)))

        if not pairs:
            continue
        # |w| 기준 Top-K
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        keep = pairs[:K]
        if keep:
            ii, ww = zip(*keep)
            rows.extend(ii)
            cols.extend([j] * len(keep))
            data.extend(ww)

    S = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n_items, n_items)
    ).tocsr()
    return S


def apply_genre_jaccard_weight(S_csr: sparse.csr_matrix,
                               nf_itemid_to_tmdb: Dict[int, int],
                               tmdb_genres: Dict[int, List[str]],
                               lam: float) -> sparse.csr_matrix:
    """
    Smoothed similarity * (1 + lam * Jaccard(genres))
    """
    S = S_csr.tocoo()
    rows, cols, vals = [], [], []
    for i, j, s in zip(S.row, S.col, S.data):
        tm_i = nf_itemid_to_tmdb.get(i)
        tm_j = nf_itemid_to_tmdb.get(j)
        if tm_i is None or tm_j is None:
            g = 1.0
        else:
            Gi = set(tmdb_genres.get(tm_i, []))
            Gj = set(tmdb_genres.get(tm_j, []))
            if not Gi and not Gj:
                g = 1.0
            else:
                inter = len(Gi & Gj)
                union = len(Gi | Gj)
                jac = 0.0 if union == 0 else inter / union
                g = 1.0 + lam * jac
        rows.append(i); cols.append(j); vals.append(s * g)
    return sparse.coo_matrix((vals, (rows, cols)), shape=S.shape).tocsr()

# -----------------------------
# Funk-SVD (SGD) + fold-in
# -----------------------------
@dataclass
class SVDModel:
    P: np.ndarray  # users x k
    Q: np.ndarray  # items x k
    bu: np.ndarray
    bi: np.ndarray
    mu: float
    k: int

def svd_train_from_centered(R_csr: sparse.csr_matrix,
                            user_means: np.ndarray,
                            k: int = 100,
                            n_epochs: int = 15,
                            lr: float = 0.005,
                            reg: float = 0.01,
                            sample_ratio: float = 1.0,
                            seed: int = 2025) -> SVDModel:
    """
    R: users x items mean-centered; true rating r = R[u,i] + user_means[u] for nonzeros
    Training on all observed entries (or sampled by sample_ratio).
    """
    try:
        rng = np.random.default_rng(seed)
    except AttributeError:
        rng = np.random.RandomState(seed)

    R = R_csr.tocsr()
    R.sum_duplicates()          # 중복 합치기
    R.eliminate_zeros()         # 저장된 0 제거
    n_users, n_items = R.shape

    # CSR 구조에서 직접 인덱스 생성
    indptr = R.indptr           # 길이 = n_users + 1
    i_idx  = R.indices          # 길이 = nnz
    vals   = R.data             # 길이 = nnz
    u_idx  = np.repeat(np.arange(n_users), np.diff(indptr))  # 길이 = nnz

    # 원평점 복원
    r_true = vals + user_means[u_idx]

    N = len(vals)
    if sample_ratio < 1.0:
        m = int(N * sample_ratio)
        sel = rng.choice(np.arange(N), size=m, replace=False)
        u_idx, i_idx, r_true = u_idx[sel], i_idx[sel], r_true[sel]
        N = m

    mu = float(np.mean(r_true))
    P = 0.01 * rng.standard_normal((n_users, k))
    Q = 0.01 * rng.standard_normal((n_items, k))
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)

    order = np.arange(N)
    for ep in range(n_epochs):
        rng.shuffle(order)
        for t in order:
            u = u_idx[t]; i = i_idx[t]; r = r_true[t]
            pred = mu + bu[u] + bi[i] + P[u].dot(Q[i])
            e = r - pred
            # updates
            bu[u] += lr * (e - reg * bu[u])
            bi[i] += lr * (e - reg * bi[i])
            Pu = P[u].copy()
            P[u] += lr * (e * Q[i] - reg * P[u])
            Q[i] += lr * (e * Pu - reg * Q[i])
    return SVDModel(P=P, Q=Q, bu=bu, bi=bi, mu=mu, k=k)

@dataclass
class FoldInUser:
    p: np.ndarray
    bu: float

def fold_in_user_ridge(item_factors: np.ndarray,
                       item_bias: np.ndarray,
                       mu: float,
                       user_ratings: List[Tuple[int, float]],
                       reg: float = 0.1) -> FoldInUser:
    """
    item_factors: Q (items x k)
    user_ratings: list of (item_index, rating) with items present in Q
    Solve for [p_u; b_u] via ridge on targets y = r - mu - b_i
    """
    if len(user_ratings) == 0:
        return FoldInUser(p=np.zeros(item_factors.shape[1]), bu=0.0)
    k = item_factors.shape[1]
    X = []
    y = []
    for i, r in user_ratings:
        q = item_factors[i]
        X.append(np.hstack([q, 1.0]))
        y.append(r - mu - item_bias[i])
    X = np.vstack(X)            # n x (k+1)
    y = np.array(y)             # n
    # Ridge with no reg on bias term
    Reg = np.diag([reg] * k + [0.0])
    A = X.T @ X + Reg
    b = X.T @ y
    w = np.linalg.solve(A, b)
    p = w[:k]
    bu = float(w[-1])
    return FoldInUser(p=p, bu=bu)

def svd_predict(u_fold: FoldInUser,
                q_i: np.ndarray,
                b_i: float,
                mu: float) -> float:
    r = mu + u_fold.bu + b_i + u_fold.p.dot(q_i)
    return float(np.clip(r, 1.0, 5.0))

# -----------------------------
# Memory-CF 예측
# -----------------------------
def memcf_predict_for_pair(u_ml: int,
                           i_nf: int,
                           ml_R_train: sparse.csr_matrix,
                           ml_user_means: np.ndarray,
                           nf_item_sim: sparse.csr_matrix,
                           ml_item_mlidx_to_nfidx: Dict[int, int],
                           rel_thresh: float = 0.5,
                           abs_thresh: Optional[float] = None,
                           Kcap: Optional[int] = None) -> Optional[float]:
    """
    u_ml: MovieLens user index (per movielens_user2idx.json)
    i_nf: target Netflix item index
    ml_R_train: MovieLens mean-centered train matrix
    """
    # user u's rated items in ML train
    row = ml_R_train.getrow(u_ml)
    if row.nnz == 0:
        return None
    # original ratings r = centered + mean
    rated_ml_items = row.indices
    centered_vals = row.data
    mean_u = ml_user_means[u_ml]
    origs = centered_vals + mean_u

    # filter "높게 평가한" 아이템
    keep_mask = np.ones_like(origs, dtype=bool)
    if abs_thresh is not None:
        keep_mask &= (origs >= abs_thresh)
    if rel_thresh is not None:
        keep_mask &= ((origs - mean_u) >= rel_thresh)
    rated_ml_items = rated_ml_items[keep_mask]
    centered_vals = centered_vals[keep_mask]

    if len(rated_ml_items) == 0:
        return None

    # map ML rated items -> NF indices
    neigh_nf = []
    neigh_dev = []
    for j_ml, dev in zip(rated_ml_items, centered_vals):
        j_nf = ml_item_mlidx_to_nfidx.get(j_ml)
        if j_nf is not None:
            neigh_nf.append(j_nf)
            neigh_dev.append(dev)

    if not neigh_nf:
        return None

    # similarity column for i_nf
    col = nf_item_sim.getcol(i_nf)  # neighbors -> i_nf
    if col.nnz == 0:
        return None

    # pick s(i,j) for user-rated neighbors
    col = col.tocoo()
    sim_map = {r: v for r, v in zip(col.row, col.data)}
    s_vals = []
    d_vals = []
    for j_nf, d in zip(neigh_nf, neigh_dev):
        s = sim_map.get(j_nf)
        if s is not None:
            s_vals.append(s)
            d_vals.append(d)
    if not s_vals:
        return None

    # cap by Kcap if provided
    if Kcap is not None and len(s_vals) > Kcap:
        idx = np.argsort(np.abs(s_vals))[::-1][:Kcap]
        s_vals = [s_vals[k] for k in idx]
        d_vals = [d_vals[k] for k in idx]

    num = np.sum(np.array(s_vals) * np.array(d_vals))
    den = np.sum(np.abs(s_vals)) + 1e-12
    pred = ml_user_means[u_ml] + num / den
    return float(np.clip(pred, 1.0, 5.0))

# -----------------------------
# ML 스플릿(테스트쌍만 평가)
# -----------------------------
def make_ml_leave_one_out(R_full: sparse.csr_matrix,
                          seed: int = 2025) -> Tuple[sparse.csr_matrix, List[Tuple[int, int, float]]]:
    """
    R_full: mean-centered ratings; reconstruct rating for split
    return: R_train (centered), test_list[(u,i,r)]
    """
    rng = np.random.default_rng(seed)
    R = R_full.tocsr(copy=True)
    n_users = R.shape[0]
    test = []
    for u in range(n_users):
        row = R.getrow(u)
        if row.nnz < 2:
            continue
        pick = rng.integers(0, row.nnz)
        i = row.indices[pick]
        val = row.data[pick]
        R[u, i] = 0.0  # remove from train
        R.eliminate_zeros()
        test.append((u, i, val))
    return R, test

# -----------------------------
# 파이프라인
# -----------------------------
def run_pipeline(
    artifacts_dir: str,
    out_dir: str,
    # MemCF
    mem_topk: int = 100,
    mem_min_common: int = 5,
    mem_shrink_b: float = 50.0,
    mem_genre_lambda: float = 0.2,
    high_abs: Optional[float] = None,   # e.g., 4.0
    high_rel: float = 0.5,
    # SVD
    svd_k: int = 100,
    svd_epochs: int = 15,
    svd_lr: float = 0.005,
    svd_reg: float = 0.01,
    svd_sample_ratio: float = 1.0,
    foldin_reg: float = 0.1,
    # Blend
    w_mem: float = 0.4,
    w_svd: float = 0.6,
    threshold: float = 0.70,
    seed: int = 2025
):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load mappings
    with open(os.path.join(artifacts_dir, "netflix_item2idx.json"), "r") as f:
        nf_item2idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(os.path.join(artifacts_dir, "netflix_user2idx.json"), "r") as f:
        nf_user2idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(os.path.join(artifacts_dir, "movielens_item2idx.json"), "r") as f:
        ml_item2idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(os.path.join(artifacts_dir, "movielens_user2idx.json"), "r") as f:
        ml_user2idx = {int(k): int(v) for k, v in json.load(f).items()}

    nf_idx2item = {v: k for k, v in nf_item2idx.items()}
    ml_idx2item = {v: k for k, v in ml_item2idx.items()}

    # --- Load matrices
    nf_R_centered = load_sparse(os.path.join(artifacts_dir, "netflix_cf_centered.npz"))
    nf_user_means = np.load(os.path.join(artifacts_dir, "netflix_user_means.npy"))
    ml_R_centered_full = load_sparse(os.path.join(artifacts_dir, "movielens_cf_centered.npz"))
    ml_user_means = np.load(os.path.join(artifacts_dir, "movielens_user_means.npy"))

    nf_R_centered = nf_R_centered.astype(np.float32)
    ml_R_centered_full = ml_R_centered_full.astype(np.float32)

    # --- Crosswalk and TMDB meta
    cw = load_crosswalk(artifacts_dir)
    tm = load_tmdb_meta(artifacts_dir)

    # NF index -> tmdb_id (via NF movieId)
    nf_idx_to_tmdb: Dict[int, int] = {}
    for idx, nf_mid in nf_idx2item.items():
        tmdb_id = cw.nf_movie_to_tmdb.get(nf_mid)
        if tmdb_id is not None:
            nf_idx_to_tmdb[idx] = tmdb_id

    # ML index -> NF index (through tmdb)
    mlidx_to_nfidx: Dict[int, int] = {}
    for ml_idx, ml_mid in ml_idx2item.items():
        tmdb_id = cw.ml_movie_to_tmdb.get(ml_mid)
        if tmdb_id is None:
            continue
        nf_mid = cw.tmdb_to_nf_movie.get(tmdb_id)
        if nf_mid is None:
            continue
        nf_idx = nf_item2idx.get(nf_mid)
        if nf_idx is not None:
            mlidx_to_nfidx[ml_idx] = nf_idx

    target_nf_cols = sorted(set(mlidx_to_nfidx.values()))
    print(f"Target NF columns for similarity: {len(target_nf_cols)}")
    if len(target_nf_cols) == 0:
        raise RuntimeError("ML→NF 매핑된 아이템이 없습니다.")

    # -------------------------
    # 1) Memory-based CF
    # -------------------------
    path_sim = os.path.join(
        out_dir,
        f"item_item_topk_K{mem_topk}_m{mem_min_common}_b{int(mem_shrink_b)}_target{len(target_nf_cols)}.npz"
    )

    if os.path.exists(path_sim):
        print(f"Loading cached similarity: {path_sim}")
        S_topk = sparse.load_npz(path_sim)
    else:
        print("Building item-item similarity for ML-mapped items...")
        S_topk = topk_cosine_for_targets(
            nf_R_centered,
            target_cols=target_nf_cols,
            K=mem_topk,
            min_common=mem_min_common,
            shrink_b=mem_shrink_b
        )
        if mem_genre_lambda and mem_genre_lambda > 0:
            print("Applying genre Jaccard weighting...")
            S_topk = apply_genre_jaccard_weight(
                S_topk,
                nf_itemid_to_tmdb=nf_idx_to_tmdb,
                tmdb_genres=tm.genres,
                lam=mem_genre_lambda
            )
        sparse.save_npz(path_sim, S_topk)
        print(f"Saved similarity cache: {path_sim}")


    # -------------------------
    # 2) SVD on Netflix
    # -------------------------
    path_svd = os.path.join(
        out_dir,
        f"svd_k{svd_k}_ep{svd_epochs}_reg{svd_reg}_sr{svd_sample_ratio}.npz"
    )

    if os.path.exists(path_svd):
        print(f"Loading cached SVD: {path_svd}")
        npz = np.load(path_svd, allow_pickle=True)
        svd = SVDModel(
            P=npz["P"], Q=npz["Q"],
            bu=npz["bu"], bi=npz["bi"],
            mu=float(npz["mu"]), k=int(npz["k"])
        )
    else:
        print("Training Funk-SVD on Netflix...")
        svd = svd_train_from_centered(
            nf_R_centered, nf_user_means,
            k=svd_k, n_epochs=svd_epochs, lr=svd_lr,
            reg=svd_reg, sample_ratio=svd_sample_ratio, seed=seed
        )
        np.savez(
            path_svd,
            P=svd.P, Q=svd.Q, bu=svd.bu, bi=svd.bi, mu=svd.mu, k=svd.k
        )
        print(f"Saved SVD cache: {path_svd}")


    # -------------------------
    # 3) ML split (LOO)
    # -------------------------
    print("Creating MovieLens leave-one-out split...")
    ml_R_train_centered, test_list = make_ml_leave_one_out(ml_R_centered_full, seed=seed)

    # -------------------------
    # 4) Fold-in ML users for SVD
    # -------------------------
    print("Folding in MovieLens users into SVD space...")
    # reconstruct ML train ratings for each user on items that map to NF indices present in Q
    Q = svd.Q; bi = svd.bi; mu = svd.mu
    user_fold_map: Dict[int, FoldInUser] = {}

    ml_R_tr = ml_R_train_centered.tocsr()
    n_users_ml = ml_R_tr.shape[0]
    for u in tqdm(range(n_users_ml)):
        row = ml_R_tr.getrow(u)
        if row.nnz == 0:
            user_fold_map[u] = FoldInUser(p=np.zeros(svd.k), bu=0.0)
            continue
        rated = []
        for j_ml, dev in zip(row.indices, row.data):
            nf_idx = mlidx_to_nfidx.get(j_ml)
            if nf_idx is None:
                continue
            r = dev + ml_user_means[u]
            rated.append((nf_idx, float(r)))
        if not rated:
            user_fold_map[u] = FoldInUser(p=np.zeros(svd.k), bu=0.0)
            continue
        fi = fold_in_user_ridge(Q, bi, mu, rated, reg=foldin_reg)
        user_fold_map[u] = fi

    # -------------------------
    # 5) Predict on ML test pairs
    # -------------------------
    print("Predicting on MovieLens test pairs and blending...")
    preds = []
    for (u_ml, i_ml, dev) in test_list:
        true_r = float(dev + ml_user_means[u_ml])

        # target NF item index
        i_nf = mlidx_to_nfidx.get(i_ml)
        # Memory-CF
        mem_pred = None
        if i_nf is not None:
            mem_pred = memcf_predict_for_pair(
                u_ml=u_ml, i_nf=i_nf, ml_R_train=ml_R_tr, ml_user_means=ml_user_means,
                nf_item_sim=S_topk, ml_item_mlidx_to_nfidx=mlidx_to_nfidx,
                rel_thresh=high_rel, abs_thresh=high_abs, Kcap=mem_topk
            )
        # SVD
        svd_pred = None
        if i_nf is not None:
            fi = user_fold_map.get(u_ml)
            if fi is not None:
                svd_pred = svd_predict(fi, Q[i_nf], bi[i_nf], mu)

        # normalize to [0,1] and blend
        parts = []
        weights = []
        if mem_pred is not None:
            parts.append((mem_pred - 1.0) / 4.0)
            weights.append(w_mem)
        if svd_pred is not None:
            parts.append((svd_pred - 1.0) / 4.0)
            weights.append(w_svd)

        if parts:
            wsum = sum(weights)
            weights = [w / wsum for w in weights]
            cf_score = float(np.clip(np.dot(parts, weights), 0.0, 1.0))
        else:
            cf_score = np.nan

        preds.append({
            "user_idx_ml": u_ml,
            "movie_idx_ml": i_ml,
            "movieId": int(ml_idx2item[i_ml]),
            "true_rating": true_r,
            "mem_pred": mem_pred,
            "svd_pred": svd_pred,
            "cf_score": cf_score
        })

    pred_df = pd.DataFrame(preds)
    pred_df.to_parquet(os.path.join(out_dir, "ml_test_predictions_raw.parquet"), index=False)

    # -------------------------
    # 6) Threshold filter + 메타 조인
    # -------------------------
    # MovieLens title
    ml_content = pd.read_parquet(os.path.join(artifacts_dir, "movielens_content.parquet"))
    ml_movie_col = guess_col(ml_content.columns, "movie") or "movieId"
    title_col = guess_col(ml_content.columns, "title") or "title"
    ml_titles = ml_content[[ml_movie_col, title_col]].drop_duplicates()
    ml_titles.columns = ["movieId", "title"]

    # attach tmdb_id → tmdb meta
    ml_to_tm = pd.DataFrame(list(cw.ml_movie_to_tmdb.items()), columns=["movieId", "tmdb_id"])
    out = pred_df.merge(ml_titles, on="movieId", how="left")
    out = out.merge(ml_to_tm, on="movieId", how="left")
    out = out.merge(tm.df, on="tmdb_id", how="left")

    # directors, genres_merged
    def pack_directors(tmdb_id: float):
        if np.isnan(tmdb_id): return ""
        return "|".join(tm.directors.get(int(tmdb_id), []))
    def pack_genres(tmdb_id: float):
        if np.isnan(tmdb_id): return ""
        return "|".join(tm.genres.get(int(tmdb_id), []))

    out["directors"] = out["tmdb_id"].apply(pack_directors)
    out["genres_merged"] = out["tmdb_id"].apply(pack_genres)

    # keep only cf_score >= threshold
    filtered = out[(~out["cf_score"].isna()) & (out["cf_score"] >= threshold)].copy()

    # final column order
    final_cols = ["movieId", "title", "title_tmdb", "directors",
                  "genres_merged", "release_year", "original_language",
                  "cf_score", "popularity", "runtime"]
    for c in final_cols:
        if c not in filtered.columns:
            filtered[c] = None
    filtered = filtered[final_cols]

    out_path = os.path.join(out_dir, f"ml_cf_blended_threshold_{threshold:.2f}.parquet")
    filtered.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    # 간단 RMSE/MAE 리포트 (threshold 무관, 예측 존재 케이스만)
    eval_df = pred_df.dropna(subset=["mem_pred", "svd_pred", "cf_score"], how="all").copy()
    if not eval_df.empty:
        # blended rating 역변환: r = 1 + 4 * cf_score
        eval_df["blend_pred_rating"] = 1.0 + 4.0 * eval_df["cf_score"]
        def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
        def mae(a,b): return float(np.mean(np.abs(a-b)))
        metrics = {
            "rmse_mem": rmse(eval_df["true_rating"], eval_df["mem_pred"].fillna(eval_df["true_rating"].mean())),
            "mae_mem": mae(eval_df["true_rating"], eval_df["mem_pred"].fillna(eval_df["true_rating"].mean())),
            "rmse_svd": rmse(eval_df["true_rating"], eval_df["svd_pred"].fillna(eval_df["true_rating"].mean())),
            "mae_svd": mae(eval_df["true_rating"], eval_df["svd_pred"].fillna(eval_df["true_rating"].mean())),
            "rmse_blend": rmse(eval_df["true_rating"], eval_df["blend_pred_rating"]),
            "mae_blend": mae(eval_df["true_rating"], eval_df["blend_pred_rating"]),
            "coverage_pred": float(len(eval_df) / len(pred_df)),
        }
    else:
        metrics = {"note": "no predictions available"}

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--out_dir", type=str, default="outputs")

    ap.add_argument("--mem_topk", type=int, default=100)
    ap.add_argument("--mem_min_common", type=int, default=5)
    ap.add_argument("--mem_shrink_b", type=float, default=50.0)
    ap.add_argument("--mem_genre_lambda", type=float, default=0.2)
    ap.add_argument("--high_abs", type=float, default=None)
    ap.add_argument("--high_rel", type=float, default=0.5)

    ap.add_argument("--svd_k", type=int, default=100)
    ap.add_argument("--svd_epochs", type=int, default=15)
    ap.add_argument("--svd_lr", type=float, default=0.005)
    ap.add_argument("--svd_reg", type=float, default=0.01)
    ap.add_argument("--svd_sample_ratio", type=float, default=1.0)
    ap.add_argument("--foldin_reg", type=float, default=0.1)

    ap.add_argument("--w_mem", type=float, default=0.4)
    ap.add_argument("--w_svd", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=2025)

    args = ap.parse_args()
    run_pipeline(**vars(args))
