# ML_termP_preprocess.py
# Netflix 학습 / MovieLens 추론 검증 파이프라인 (TMDb 404 복구 포함 최종판)

import os, re, csv, json, time, logging, random
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 선택: pandas 향후 경고 제어
# pd.set_option('future.no_silent_downcasting', True)

# .env 지원
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========================
# 설정
# =========================
BASE_DIR = Path(".").resolve()
MOVIELENS_DIR = BASE_DIR / "movielens"
NETFLIX_DIR   = BASE_DIR / "netflix"
ARTIFACT_DIR  = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 1_000_000
ENRICH_WITH_TMDB = True
BUILD_CBF = True

# TMDb
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # 환경변수로 넣기
TMDB_BASE = "https://api.themoviedb.org/3"


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocess")

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

# =========================
# MovieLens 파트
# =========================
def iter_movielens_ratings(path: Path, chunksize: int = CHUNK_SIZE):
    return pd.read_csv(
        path,
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
        chunksize=chunksize
    )

def load_movielens_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["genres"] = df["genres"].fillna("(no genres listed)")
    return df

def load_movielens_links(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_movielens_tags(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_movielens_genome(tags_path: Path, scores_path: Path):
    genome_tags = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame()
    genome_scores = pd.read_csv(
        scores_path,
        dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float32}
    ) if scores_path.exists() else pd.DataFrame()
    return genome_tags, genome_scores

def build_movielens_content(movies, links, genome_tags, genome_scores):
    print("[MovieLens] building content table ...")
    content = movies.copy()
    if not links.empty:
        content = content.merge(links, on="movieId", how="left")
    content["genre_list"] = content["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    if not genome_scores.empty and not genome_tags.empty:
        genome_merged = genome_scores.merge(genome_tags, on="tagId", how="left")
        genome_merged.to_parquet(ARTIFACT_DIR / "movielens_genome_long.parquet", index=False)
        print("[MovieLens] saved -> artifacts/movielens_genome_long.parquet")
    content.to_parquet(ARTIFACT_DIR / "movielens_content.parquet", index=False)
    print("[MovieLens] saved -> artifacts/movielens_content.parquet")
    return content

def build_movielens_cf_matrix_from_chunks(ratings_path: Path, chunksize: int = CHUNK_SIZE):
    print("[MovieLens] pass1: collecting unique userIds and movieIds ...")
    user_set, item_set = set(), set()
    for chunk in iter_movielens_ratings(ratings_path, chunksize):
        user_set.update(chunk["userId"].tolist())
        item_set.update(chunk["movieId"].tolist())
    user_ids = [int(u) for u in sorted(user_set)]
    item_ids = [int(m) for m in sorted(item_set)]
    user2idx = {int(u): int(i) for i, u in enumerate(user_ids)}
    item2idx = {int(m): int(i) for i, m in enumerate(item_ids)}
    n_users, n_items = len(user2idx), len(item2idx)
    print(f"[MovieLens] users={n_users}, items={n_items}")

    data_parts, row_parts, col_parts = [], [], []
    print("[MovieLens] pass2: building COO arrays ...")
    for chunk in iter_movielens_ratings(ratings_path, chunksize):
        rows = chunk["userId"].map(user2idx).to_numpy()
        cols = chunk["movieId"].map(item2idx).to_numpy()
        vals = chunk["rating"].to_numpy(dtype=np.float32)
        data_parts.append(vals); row_parts.append(rows); col_parts.append(cols)
    data = np.concatenate(data_parts)
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()

    print("[MovieLens] mean-centering ...")
    user_means = np.zeros(n_users, dtype=np.float32)
    for u in range(n_users):
        s, e = mat.indptr[u], mat.indptr[u+1]
        if s == e: continue
        r = mat.data[s:e]; mu = r.mean()
        mat.data[s:e] = r - mu; user_means[u] = mu

    sparse.save_npz(ARTIFACT_DIR / "movielens_cf_centered.npz", mat)
    np.save(ARTIFACT_DIR / "movielens_user_means.npy", user_means)
    save_json(ARTIFACT_DIR / "movielens_user2idx.json", {str(k): int(v) for k, v in user2idx.items()})
    save_json(ARTIFACT_DIR / "movielens_item2idx.json", {str(k): int(v) for k, v in item2idx.items()})
    print(f"[MovieLens] CF matrix saved. shape={mat.shape}")
    return mat

def build_movielens_sasrec_sequences_from_chunks(
    ratings_path: Path, chunksize: int = CHUNK_SIZE, min_items_per_user: int = 3, max_seq_len: int = 200,
):
    print("[MovieLens] building SASRec sequences (chunk) ...")
    user_history: Dict[int, List[Tuple[int, int]]] = {}
    for chunk in iter_movielens_ratings(ratings_path, chunksize):
        for row in chunk.itertuples(index=False):
            u = int(row.userId); m = int(row.movieId); t = int(row.timestamp)
            user_history.setdefault(u, []).append((t, m))
    all_movie_ids = sorted({m for hist in user_history.values() for _, m in hist})
    item2idx = {int(m): int(i + 1) for i, m in enumerate(all_movie_ids)}  # 0 padding

    user2seq = {}
    for u, hist in user_history.items():
        hist.sort(key=lambda x: x[0])
        seq = [item2idx[m] for (t, m) in hist]
        if len(seq) < min_items_per_user: continue
        if len(seq) > max_seq_len: seq = seq[-max_seq_len:]
        user2seq[int(u)] = seq

    save_json(ARTIFACT_DIR / "movielens_sasrec_sequences.json",
              {"user2seq": {str(k): v for k, v in user2seq.items()},
               "item2idx": {str(k): int(v) for k, v in item2idx.items()}})
    print(f"[MovieLens] SASRec sequences saved. users={len(user2seq)}, items={len(item2idx)}")

# =========================
# Netflix 파트
# =========================
def load_netflix_titles(path: Path) -> pd.DataFrame:
    print(f"[Netflix] loading titles from {path} ...")
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            movie_id = int(row[0])
            year_raw = row[1].strip() if len(row) > 1 else ""
            year = None if year_raw in ("", "NULL", "null", "\\N") else int(year_raw)
            title = ",".join(row[2:]).strip() if len(row) > 2 else ""
            rows.append((movie_id, year, title))
    df = pd.DataFrame(rows, columns=["movieId", "year", "title"])
    df.to_parquet(ARTIFACT_DIR / "netflix_titles.parquet", index=False)
    print("[Netflix] saved -> artifacts/netflix_titles.parquet")
    return df

def parse_netflix_block_lines(lines: Iterable[str], movie_id: Optional[int] = None):
    current_movie = movie_id
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.endswith(":"):
            current_movie = int(line[:-1])
        else:
            cust_id, rating, date_str = line.split(",")
            yield current_movie, int(cust_id), int(rating), date_str

def parse_netflix_mv_file(path: Path):
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    return list(parse_netflix_block_lines(lines))

def parse_netflix_combined_file(path: Path):
    with open(path, "r", encoding="latin-1") as f:
        for movie_id, cust_id, rating, date_str in parse_netflix_block_lines(f):
            yield movie_id, cust_id, rating, date_str

def generate_synthetic_netflix_ratings(n_movies: int = 500, n_users: int = 1000, max_per_movie: int = 200):
    print("[Netflix] generating synthetic ratings ...")
    rng = np.random.default_rng(42)
    for m in range(1, n_movies + 1):
        n_r = rng.integers(20, max_per_movie)
        for _ in range(n_r):
            user = int(rng.integers(1, n_users + 1))
            rating = int(rng.integers(1, 6))
            year = rng.integers(1999, 2006)
            month = int(rng.integers(1, 13))
            day = int(rng.integers(1, 29))
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            yield m, user, rating, date_str

def load_or_mock_netflix_ratings(netflix_dir: Path) -> pd.DataFrame:
    mv_files = sorted(netflix_dir.glob("mv_*.txt"))
    rows = []
    if mv_files:
        print(f"[Netflix] detected per-movie files: {len(mv_files)}개")
        for i, mv_path in enumerate(mv_files, 1):
            parsed = parse_netflix_mv_file(mv_path)
            rows.extend(parsed)
            if i % 1000 == 0:
                print(f"[Netflix] ... parsed {i} files")
    else:
        combined_files = [netflix_dir / f"combined_data_{i}.txt" for i in range(1, 5)]
        real_found = False
        for cf in combined_files:
            if cf.exists():
                real_found = True
                print(f"[Netflix] parsing {cf.name} ...")
                for movie_id, cust_id, rating, date_str in parse_netflix_combined_file(cf):
                    rows.append((movie_id, cust_id, rating, date_str))
        if not real_found:
            print("[Netflix] no real files -> using synthetic")
            for movie_id, cust_id, rating, date_str in generate_synthetic_netflix_ratings():
                rows.append((movie_id, cust_id, rating, date_str))

    df = pd.DataFrame(rows, columns=["movieId", "customerId", "rating", "date"])
    df.to_parquet(ARTIFACT_DIR / "netflix_ratings.parquet", index=False)
    print(f"[Netflix] saved -> artifacts/netflix_ratings.parquet (rows={len(df)})")
    return df

def netflix_to_cf(df: pd.DataFrame):
    print("[Netflix] building CF matrix ...")
    users = df["customerId"].unique()
    items = df["movieId"].unique()
    user2idx = {int(u): int(i) for i, u in enumerate(users)}
    item2idx = {int(m): int(i) for i, m in enumerate(items)}

    rows = df["customerId"].map(user2idx).to_numpy()
    cols = df["movieId"].map(item2idx).to_numpy()
    vals = df["rating"].to_numpy(dtype=np.float32)
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(len(user2idx), len(item2idx))).tocsr()

    user_means = np.zeros(len(user2idx), dtype=np.float32)
    for u in range(len(user2idx)):
        s, e = mat.indptr[u], mat.indptr[u+1]
        if s == e: continue
        r = mat.data[s:e]; mu = r.mean()
        mat.data[s:e] = r - mu; user_means[u] = mu

    sparse.save_npz(ARTIFACT_DIR / "netflix_cf_centered.npz", mat)
    np.save(ARTIFACT_DIR / "netflix_user_means.npy", user_means)
    save_json(ARTIFACT_DIR / "netflix_user2idx.json", {str(k): int(v) for k, v in user2idx.items()})
    save_json(ARTIFACT_DIR / "netflix_item2idx.json", {str(k): int(v) for k, v in item2idx.items()})
    print(f"[Netflix] CF matrix saved. shape={mat.shape}")

def netflix_to_sasrec(df: pd.DataFrame, max_seq_len: int = 200):
    print("[Netflix] building SASRec sequences ...")
    df = df.sort_values(["customerId", "date"]).reset_index(drop=True)
    items = df["movieId"].unique()
    item2idx = {int(m): int(i + 1) for i, m in enumerate(items)}  # 0 padding

    user2seq = {}
    for u, grp in df.groupby("customerId"):
        seq = [item2idx[int(m)] for m in grp["movieId"].tolist()]
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        user2seq[int(u)] = seq

    save_json(ARTIFACT_DIR / "netflix_sasrec_sequences.json",
              {"user2seq": {str(k): v for k, v in user2seq.items()},
               "item2idx": {str(k): int(v) for k, v in item2idx.items()}})
    print(f"[Netflix] SASRec sequences saved. users={len(user2seq)}, items={len(item2idx)}")

# =========================
# TMDb API 유틸 (404 즉시 스킵 + 재시도/백오프)
# =========================
def _norm_title(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.strip()
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()

def _build_session():
    s = requests.Session()
    retries = Retry(
        total=7,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = _build_session()

def _tmdb_get(path, params=None, sleep=0.25):
    if TMDB_API_KEY is None:
        raise RuntimeError("TMDB_API_KEY 환경변수가 설정되어 있지 않습니다.")
    url = f"{TMDB_BASE}/{path.lstrip('/')}"
    q = {"api_key": TMDB_API_KEY}
    if params: q.update(params)

    attempt = 0
    while True:
        attempt += 1
        try:
            r = _SESSION.get(url, params=q, timeout=(5, 45))
            if r.status_code == 404:
                # 존재하지 않는 리소스 → 즉시 None 반환 (재시도 불필요)
                return None
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (2.0 * attempt)
                wait += random.uniform(0, 0.5)
                log.warning(f"[TMDb] 429 rate limit. sleeping {wait:.1f}s")
                time.sleep(wait); continue
            r.raise_for_status()
            if sleep: time.sleep(sleep + random.uniform(0, 0.2))
            return r.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            wait = min(60.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            log.warning(f"[TMDb] timeout (attempt {attempt}). sleep {wait:.1f}s: {e}")
            time.sleep(wait)
            if attempt >= 7: return None
        except requests.RequestException as e:
            wait = min(60.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            log.warning(f"[TMDb] request error (attempt {attempt}). sleep {wait:.1f}s: {e}")
            time.sleep(wait)
            if attempt >= 7: return None

@lru_cache(maxsize=4096)
def tmdb_search_movie(title: str, year: int | None):
    params = {"query": title, "include_adult": False}
    if year: params["year"] = int(year)
    return _tmdb_get("/search/movie", params=params)

@lru_cache(maxsize=4096)
def tmdb_get_details(tmdb_id: int):
    return _tmdb_get(f"/movie/{tmdb_id}", params={"append_to_response": "credits"})

@lru_cache(maxsize=4096)
def tmdb_find_by_imdb(imdb_id: str | int):
    if pd.isna(imdb_id):
        return None
    s = str(int(imdb_id)).rjust(7, "0")
    tt = f"tt{s}"
    res = _tmdb_get(f"/find/{tt}", params={"external_source": "imdb_id"})
    if not res:
        return None
    movies = res.get("movie_results", []) or []
    if movies:
        return int(movies[0]["id"])
    return None

def pick_tmdb_from_search(title: str, year: int | None):
    tnorm = _norm_title(title)
    res = tmdb_search_movie(title, year)
    results = res.get("results", []) if res else []
    if not results:
        res = tmdb_search_movie(title, None)
        results = res.get("results", []) if res else []
        if not results: return None
    for x in results:
        name = _norm_title(x.get("title") or x.get("original_title") or "")
        rel = (x.get("release_date") or "")[:4]
        if name == tnorm and (not year or rel == str(year)):
            return x
    for x in results:
        name = _norm_title(x.get("title") or x.get("original_title") or "")
        if name == tnorm:
            return x
    return results[0] if results else None

# =========================
# TMDb 매핑/보강/교차 시퀀스
# =========================
def build_tmdb_mapping_for_netflix_titles(
    netflix_titles_parquet: Path,
    checkpoint_every: int = 300,
    resume: bool = True,
) -> pd.DataFrame:
    df = pd.read_parquet(netflix_titles_parquet)
    out_path = ARTIFACT_DIR / "idmap_netflix_movie_to_tmdb.parquet"
    tmp_path = ARTIFACT_DIR / "idmap_netflix_movie_to_tmdb.tmp.parquet"

    done_ids = set()
    rows = []

    if resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        rows.extend(prev.itertuples(index=False, name=None))
        done_ids.update(int(r[0]) for r in rows)
        log.info(f"[TMDb] resume: loaded {len(done_ids)} existing mappings")

    ok = sum(1 for r in rows if r[1] is not None)
    fail = sum(1 for r in rows if r[1] is None)
    since_last_ckpt = 0

    for r in df.itertuples(index=False):
        mid = int(getattr(r, "movieId"))
        if mid in done_ids: continue

        title = getattr(r, "title"); year  = getattr(r, "year")
        try:
            pick = pick_tmdb_from_search(title, year if pd.notna(year) else None)
        except Exception as e:
            log.warning(f"[TMDb] search error for movieId={mid} '{title}': {e}")
            pick = None

        if pick is None:
            rows.append((mid, None, year, title)); fail += 1
        else:
            tid = int(pick["id"])
            rows.append((mid, tid, year, title)); ok += 1

        done_ids.add(mid)
        since_last_ckpt += 1

        if (ok + fail) % 100 == 0:
            log.info(f"[TMDb] netflix mapped {ok} ok / {fail} fail")

        if since_last_ckpt >= checkpoint_every:
            pd.DataFrame(rows, columns=["netflix_movieId","tmdbId","year","title"]).to_parquet(tmp_path, index=False)
            tmp_path.replace(out_path); since_last_ckpt = 0

    pd.DataFrame(rows, columns=["netflix_movieId","tmdbId","year","title"]).to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    log.info(f"[TMDb] netflix title→tmdb mapping saved. ok={ok}, fail={fail}")
    return pd.read_parquet(out_path)

def build_tmdb_mapping_for_movielens_links(links_csv: Path, movies_csv: Path) -> pd.DataFrame:
    links = pd.read_csv(links_csv) if links_csv.exists() else pd.DataFrame()
    movies = pd.read_csv(movies_csv)
    df = movies[["movieId", "title"]].copy()
    years = df["title"].str.extract(r"\((\d{4})\)$")
    df["year"] = years[0].astype("Int64")
    df["title_plain"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    if not links.empty and "tmdbId" in links.columns:
        merged = df.merge(links[["movieId", "tmdbId", "imdbId"]], on="movieId", how="left")
    else:
        merged = df.copy(); merged["tmdbId"] = pd.NA; merged["imdbId"] = pd.NA

    need = merged["tmdbId"].isna()
    if need.sum() > 0:
        log.info(f"[TMDb] movielens missing tmdbId: {need.sum()} rows → TMDb search")
        fills = []
        for r in merged.loc[need, ["movieId", "title_plain", "year", "imdbId"]].itertuples(index=False):
            # 우선 IMDb→TMDb
            tid = None
            if pd.notna(getattr(r, "imdbId")):
                try:
                    tid = tmdb_find_by_imdb(getattr(r, "imdbId"))
                except Exception as e:
                    log.warning(f"[TMDb] imdb find fail ML movieId={int(getattr(r,'movieId'))}: {e}")
            # 실패 시 제목/연도 검색
            if tid is None:
                yr = int(getattr(r, "year")) if pd.notna(getattr(r, "year")) else None
                try:
                    pick = pick_tmdb_from_search(getattr(r, "title_plain"), yr)
                    tid = int(pick["id"]) if pick else None
                except Exception as e:
                    log.warning(f"[TMDb] title search fail ML movieId={int(getattr(r,'movieId'))}: {e}")
            fills.append((int(getattr(r, "movieId")), tid))
        fill_df = pd.DataFrame(fills, columns=["movieId", "tmdb_fill"])
        merged = merged.merge(fill_df, on="movieId", how="left")
        merged["tmdbId"] = merged["tmdbId"].fillna(merged["tmdb_fill"])
        merged["tmdbId"] = merged["tmdbId"].astype("Int64")
        merged = merged.drop(columns=["tmdb_fill"])

    out = merged[["movieId", "tmdbId", "imdbId", "title", "year"]].copy()
    out.to_parquet(ARTIFACT_DIR / "idmap_movielens_movie_to_tmdb.parquet", index=False)
    log.info("[TMDb] movielens movieId→tmdbId mapping saved.")
    return out

def enrich_tmdb_metadata(all_tmdb_ids: Iterable[int]) -> pd.DataFrame:
    recs = []; seen = set()
    for tid in all_tmdb_ids:
        if pd.isna(tid): continue
        tid = int(tid)
        if tid in seen: continue
        seen.add(tid)
        d = tmdb_get_details(tid)
        if not d:
            log.warning(f"[TMDb] details missing for tmdbId={tid}")
            continue
        genres = [g["name"] for g in d.get("genres", [])]
        year = (d.get("release_date") or "")[:4]
        runtime = d.get("runtime"); popularity = d.get("popularity")
        title = d.get("title") or d.get("original_title")
        directors = [c.get("name") for c in d.get("credits", {}).get("crew", []) if c.get("job") == "Director"]
        recs.append({
            "tmdbId": tid,
            "title_tmdb": title,
            "release_year_tmdb": year if year else None,
            "genres": genres,
            "directors": directors,
            "runtime": runtime,
            "popularity": popularity,
        })
        if len(seen) % 500 == 0:
            log.info(f"[TMDb] enriched {len(seen)} items")
    df = pd.DataFrame(recs)
    df.to_parquet(ARTIFACT_DIR / "tmdb_items_enriched.parquet", index=False)
    log.info("[TMDb] tmdb_items_enriched saved.")
    return df

def build_crosswalk_and_sequences_tmdb(idmap_netflix: pd.DataFrame, idmap_ml: pd.DataFrame):
    nx = idmap_netflix.dropna(subset=["tmdbId"]).copy()
    ml = idmap_ml.dropna(subset=["tmdbId"]).copy()
    nx["tmdbId"] = nx["tmdbId"].astype(int)
    ml["tmdbId"] = ml["tmdbId"].astype(int)
    cross = nx.merge(ml, on="tmdbId", how="inner", suffixes=("_netflix", "_movielens"))
    cross.to_parquet(ARTIFACT_DIR / "crosswalk_tmdb_movielens_netflix.parquet", index=False)

    with open(ARTIFACT_DIR / "movielens_sasrec_sequences.json", "r") as f:
        ml_seq = json.load(f)
    with open(ARTIFACT_DIR / "netflix_sasrec_sequences.json", "r") as f:
        nx_seq = json.load(f)

    ml_movie_to_tmdb = dict(zip(ml["movieId"].astype(int), ml["tmdbId"].astype(int)))
    nx_movie_to_tmdb = dict(zip(nx["netflix_movieId"].astype(int), nx["tmdbId"].astype(int)))

    ml_item2idx = {int(k): int(v) for k, v in ml_seq["item2idx"].items()}
    ml_idx2item = {v: k for k, v in ml_item2idx.items()}
    nx_item2idx = {int(k): int(v) for k, v in nx_seq["item2idx"].items()}
    nx_idx2item = {v: k for k, v in nx_item2idx.items()}

    ml_user2seq_tmdb = {}
    for u, seq in ml_seq["user2seq"].items():
        tmdb_seq = []
        for idx in seq:
            mv = ml_idx2item.get(int(idx))
            if mv is None: continue
            t = ml_movie_to_tmdb.get(int(mv))
            if t: tmdb_seq.append(int(t))
        if tmdb_seq: ml_user2seq_tmdb[u] = tmdb_seq

    nx_user2seq_tmdb = {}
    for u, seq in nx_seq["user2seq"].items():
        tmdb_seq = []
        for idx in seq:
            mv = nx_idx2item.get(int(idx))
            if mv is None: continue
            t = nx_movie_to_tmdb.get(int(mv))
            if t: tmdb_seq.append(int(t))
        if tmdb_seq: nx_user2seq_tmdb[u] = tmdb_seq

    all_tmdb = set()
    for s in ml_user2seq_tmdb.values(): all_tmdb.update(s)
    for s in nx_user2seq_tmdb.values(): all_tmdb.update(s)
    tmdb2idx = {tid: i + 1 for i, tid in enumerate(sorted(all_tmdb))}

    save_json(ARTIFACT_DIR / "movielens_sasrec_sequences_tmdb.json",
              {"user2seq": ml_user2seq_tmdb, "item2idx_tmdb": {str(k): int(v) for k, v in tmdb2idx.items()}})
    save_json(ARTIFACT_DIR / "netflix_sasrec_sequences_tmdb.json",
              {"user2seq": nx_user2seq_tmdb, "item2idx_tmdb": {str(k): int(v) for k, v in tmdb2idx.items()}})

    log.info(f"[COVERAGE] ML users with tmdb seq: {len(ml_user2seq_tmdb)}")
    log.info(f"[COVERAGE] NX users with tmdb seq: {len(nx_user2seq_tmdb)}")
    log.info(f"[COVERAGE] unique tmdb items: {len(tmdb2idx)}")

def report_tmdb_coverage():
    try:
        nx = pd.read_parquet(ARTIFACT_DIR / "idmap_netflix_movie_to_tmdb_enhanced_repaired.parquet")
    except Exception:
        try:
            nx = pd.read_parquet(ARTIFACT_DIR / "idmap_netflix_movie_to_tmdb_enhanced.parquet")
        except Exception:
            nx = pd.read_parquet(ARTIFACT_DIR / "idmap_netflix_movie_to_tmdb.parquet")
    try:
        ml = pd.read_parquet(ARTIFACT_DIR / "idmap_movielens_movie_to_tmdb.parquet")
    except Exception as e:
        print("[COVERAGE] skipped:", e); return
    nx_cov = nx["tmdbId"].notna().mean() if "tmdbId" in nx.columns else 0.0
    ml_cov = ml["tmdbId"].notna().mean() if "tmdbId" in ml.columns else 0.0
    both = nx.dropna(subset=["tmdbId"]).merge(ml.dropna(subset=["tmdbId"]), on="tmdbId", how="inner")
    print(f"[COVERAGE] Netflix tmdbId coverage: {nx_cov:.3f}")
    print(f"[COVERAGE] MovieLens tmdbId coverage: {ml_cov:.3f}")
    print(f"[COVERAGE] TMDb intersection (titles): {both.shape[0]} pairs")

# =========================
# CBF (TMDb 메타 → TF-IDF)
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer

def _cbf_text_from_meta(row: pd.Series) -> str:
    genres = row.get("genres") or []
    dirs   = row.get("directors") or []
    year   = row.get("release_year_tmdb")
    gtxt = " ".join([str(g).strip().replace(" ", "_") for g in genres])
    dtxt = " ".join([str(d).strip().replace(" ", "_") for d in dirs])
    ytxt = ""
    if isinstance(year, str) and year.isdigit():
        y = int(year); ytxt = f"year_{(y//10)*10}s"
    parts = [gtxt, dtxt, ytxt]
    return " ".join([p for p in parts if p]).strip()

def _build_cbf_features_generic(idmap_df: pd.DataFrame, id_col_name: str, prefix: str):
    meta_path = ARTIFACT_DIR / "tmdb_items_enriched.parquet"
    if not meta_path.exists():
        print(f"[CBF] {meta_path.name} not found. skip CBF for {prefix}.")
        return
    meta = pd.read_parquet(meta_path)
    for c in ["tmdbId", "genres", "directors", "release_year_tmdb"]:
        if c not in meta.columns: meta[c] = None

    m = idmap_df.dropna(subset=["tmdbId"]).copy()
    m["tmdbId"] = m["tmdbId"].astype(int)
    df = m.merge(meta[["tmdbId", "genres", "directors", "release_year_tmdb"]], on="tmdbId", how="inner")
    if df.empty:
        print(f"[CBF] no overlapped items to build TF-IDF ({prefix}).")
        return

    df["cbf_text"] = df.apply(_cbf_text_from_meta, axis=1)
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9_\-]+")
    X = vec.fit_transform(df["cbf_text"].fillna(""))

    sparse.save_npz(ARTIFACT_DIR / f"{prefix}_cbf_tfidf.npz", X)
    with open(ARTIFACT_DIR / f"{prefix}_cbf_vocab.json", "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in vec.vocabulary_.items()}, f, ensure_ascii=False)
    df[[id_col_name, "tmdbId"]].reset_index(drop=True).to_parquet(
        ARTIFACT_DIR / f"{prefix}_item_tmdb_map.parquet", index=False
    )
    print(f"[CBF] saved -> {prefix}_cbf_tfidf.npz (items={X.shape[0]}, dims={X.shape[1]})")

# =========================
# Netflix→MovieLens 제목/연도 매칭(퍼지) + TMDb 붙이기
# =========================
from rapidfuzz import fuzz, process
from unidecode import unidecode

def _normalize_title_for_match(t: str) -> str:
    if not isinstance(t, str): return ""
    t = unidecode(t)
    t = t.lower().strip()
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r":|;|-|–|—", " ", t)
    t = re.sub(r",\s*(the|a|an)$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def build_crosswalk_netflix_to_movielens_via_titles(
    netflix_titles_parquet: Path,
    movielens_movies_csv: Path,
    movielens_links_csv: Path,
    fuzzy_threshold: int = 92,
) -> pd.DataFrame:
    nx = pd.read_parquet(netflix_titles_parquet)  # movieId, year, title
    ml_movies = pd.read_csv(movielens_movies_csv)
    ml_links = pd.read_csv(movielens_links_csv) if movielens_links_csv.exists() else pd.DataFrame(columns=["movieId","tmdbId"])

    ml = ml_movies[["movieId","title"]].copy()
    years = ml["title"].str.extract(r"\((\d{4})\)$")
    ml["year"] = years[0].astype("Int64")
    ml["title_plain"] = ml["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    nx["title_norm"] = nx["title"].apply(_normalize_title_for_match)
    ml["title_norm"] = ml["title_plain"].apply(_normalize_title_for_match)

    exact = nx.merge(
        ml, left_on=["title_norm","year"], right_on=["title_norm","year"], how="left", suffixes=("_nx","_ml")
    )

    matched_rows, pending = [], []
    for r in exact.itertuples(index=False):
        nx_mid   = int(getattr(r, "movieId_nx"))
        nx_title = getattr(r, "title_nx")
        nx_year  = getattr(r, "year")
        ml_mid   = getattr(r, "movieId_ml")
        ml_title = getattr(r, "title_ml")
        ml_year  = getattr(r, "year")
        if pd.notna(ml_mid):
            matched_rows.append((nx_mid, nx_title, nx_year, int(ml_mid), ml_title, int(ml_year) if pd.notna(ml_year) else None))
        else:
            pending.append((nx_mid, nx_title, nx_year, getattr(r, "title_norm")))

    ml_by_year = {}
    for yr, grp in ml.groupby("year"):
        ml_by_year[int(yr) if pd.notna(yr) else None] = grp

    def _candidate_ml(yr):
        cands = []
        for y in [yr, (yr-1 if isinstance(yr, int) else None), (yr+1 if isinstance(yr, int) else None), None]:
            g = ml_by_year.get(y)
            if g is not None and len(g) > 0:
                cands.append(g)
        if cands:
            return pd.concat(cands, ignore_index=True).drop_duplicates("movieId")
        return ml

    for nx_mid, nx_title, nx_year, nx_norm in pending:
        pool = _candidate_ml(int(nx_year)) if pd.notna(nx_year) else ml
        scores = process.extract(
            nx_norm,
            pool["title_norm"].tolist(),
            scorer=fuzz.token_set_ratio,
            limit=5
        )
        best_ml_mid = None; best_ml_title = None; best_ml_year = None; best_score = -1
        for cand_title, score, idx in scores:
            if score < fuzzy_threshold:
                continue
            row = pool.iloc[idx]
            year_penalty = 0
            if pd.notna(nx_year) and pd.notna(row["year"]):
                year_penalty = -abs(int(nx_year) - int(row["year"])) * 2
            length_penalty = -abs(len(nx_norm) - len(row["title_norm"]))
            final_score = score*1.0 + 0.1*length_penalty + 0.5*year_penalty
            if final_score > best_score:
                best_score = final_score
                best_ml_mid = int(row["movieId"])
                best_ml_title = row["title"]
                best_ml_year  = int(row["year"]) if pd.notna(row["year"]) else None
        if best_ml_mid is not None:
            matched_rows.append((nx_mid, nx_title, nx_year, best_ml_mid, best_ml_title, best_ml_year))

    cross = pd.DataFrame(matched_rows, columns=[
        "netflix_movieId","nx_title","nx_year","movielens_movieId","ml_title","ml_year"
    ])

    if not ml_links.empty and "tmdbId" in ml_links.columns:
        cross = cross.merge(
            ml_links[["movieId","tmdbId","imdbId"]].rename(columns={"movieId":"movielens_movieId"}),
            on="movielens_movieId", how="left"
        )
    else:
        cross["tmdbId"] = pd.NA; cross["imdbId"] = pd.NA

    out_path = ARTIFACT_DIR / "crosswalk_netflix_to_movielens.parquet"
    cross.to_parquet(out_path, index=False)
    log.info(f"[XWALK] netflix→movielens crosswalk saved. rows={len(cross)}, tmdb attached={cross['tmdbId'].notna().sum()}")
    return cross

def combine_netflix_idmaps_with_crosswalk(idmap_netflix: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    if idmap_netflix.empty:
        base = pd.DataFrame(columns=["netflix_movieId","tmdbId","year","title"])
    else:
        base = idmap_netflix.copy()

    cx = cross[["netflix_movieId","tmdbId","nx_year","nx_title"]].copy()
    cx = cx.rename(columns={"nx_year":"year_from_cx","nx_title":"title_from_cx"})

    merged = base.merge(cx, on="netflix_movieId", how="outer", suffixes=("","_cx"))
    merged["tmdbId"] = merged["tmdbId"].fillna(merged["tmdbId_cx"])
    merged["title"]  = merged["title"].fillna(merged["title_from_cx"])
    merged["year"]   = merged["year"].fillna(merged["year_from_cx"])

    merged = merged[["netflix_movieId","tmdbId","year","title"]].copy()
    merged["tmdbId"] = merged["tmdbId"].astype("Int64")
    merged.to_parquet(ARTIFACT_DIR / "idmap_netflix_movie_to_tmdb_enhanced.parquet", index=False)
    log.info(f"[XWALK] enhanced netflix idmap saved. rows={len(merged)}, tmdb non-null={merged['tmdbId'].notna().sum()}")
    return merged

# ====== 404 복구: IMDb→TMDb 우선, 실패 시 제목/연도 재검색 ======
def validate_and_repair_tmdb_ids(
    idmap_netflix_enh: pd.DataFrame,
    cross_nx_ml: pd.DataFrame,
    ml_links_csv: Path,
    ml_movies_csv: Path,
) -> pd.DataFrame:
    ml_links = pd.read_csv(ml_links_csv) if ml_links_csv.exists() else pd.DataFrame(columns=["movieId","imdbId"])
    nx_to_ml = dict(zip(cross_nx_ml["netflix_movieId"], cross_nx_ml["movielens_movieId"]))
    ml_imdb = dict(zip(ml_links["movieId"], ml_links["imdbId"]))
    nx_title = dict(zip(idmap_netflix_enh["netflix_movieId"], idmap_netflix_enh["title"]))
    nx_year  = dict(zip(idmap_netflix_enh["netflix_movieId"], idmap_netflix_enh["year"]))

    fixed = 0
    rows = []
    checked_cache = {}

    for r in idmap_netflix_enh.itertuples(index=False):
        nx_mid = int(getattr(r, "netflix_movieId"))
        tid = getattr(r, "tmdbId")
        y   = getattr(r, "year")
        title = getattr(r, "title")

        if pd.isna(tid):
            rows.append((nx_mid, pd.NA, y, title))
            continue

        tid = int(tid)
        ok = checked_cache.get(tid)
        if ok is None:
            d = tmdb_get_details(tid)
            ok = d is not None
            checked_cache[tid] = ok

        if ok:
            rows.append((nx_mid, tid, y, title))
            continue

        # 404 등 무효 → 복구 시도
        new_tid = None

        # (A) imdbId로 /find
        ml_mid = nx_to_ml.get(nx_mid)
        if ml_mid is not None:
            imdb_id = ml_imdb.get(ml_mid)
            if imdb_id is not None and not pd.isna(imdb_id):
                try:
                    new_tid = tmdb_find_by_imdb(imdb_id)
                except Exception as e:
                    log.warning(f"[REPAIR] imdb find fail nx={nx_mid} imdb={imdb_id}: {e}")

        # (B) 제목/연도 재검색
        if new_tid is None:
            try:
                t = nx_title.get(nx_mid, title)
                yr = int(nx_year.get(nx_mid, y)) if pd.notna(nx_year.get(nx_mid, y)) else None
                pick = pick_tmdb_from_search(t, yr)
                if pick: new_tid = int(pick["id"])
            except Exception as e:
                log.warning(f"[REPAIR] title/year search fail nx={nx_mid}: {e}")

        if new_tid is not None:
            fixed += 1
            rows.append((nx_mid, new_tid, y, title))
        else:
            rows.append((nx_mid, pd.NA, y, title))

    out = pd.DataFrame(rows, columns=["netflix_movieId","tmdbId","year","title"])
    out["tmdbId"] = out["tmdbId"].astype("Int64")
    out.to_parquet(ARTIFACT_DIR / "idmap_netflix_movie_to_tmdb_enhanced_repaired.parquet", index=False)
    log.info(f"[REPAIR] netflix tmdbId repaired: {fixed} items")
    return out

# =========================
# main
# =========================
def main():
    # ----- MovieLens -----
    ml_ratings_path = MOVIELENS_DIR / "ratings.csv"
    ml_movies_path  = MOVIELENS_DIR / "movies.csv"
    ml_links_path   = MOVIELENS_DIR / "links.csv"
    ml_tags_path    = MOVIELENS_DIR / "tags.csv"
    ml_genome_tags_path   = MOVIELENS_DIR / "genome-tags.csv"
    ml_genome_scores_path = MOVIELENS_DIR / "genome-scores.csv"

    ml_movies = load_movielens_movies(ml_movies_path)
    ml_links  = load_movielens_links(ml_links_path)
    ml_tags   = load_movielens_tags(ml_tags_path)
    ml_genome_tags, ml_genome_scores = load_movielens_genome(ml_genome_tags_path, ml_genome_scores_path)

    build_movielens_content(ml_movies, ml_links, ml_genome_tags, ml_genome_scores)
    build_movielens_cf_matrix_from_chunks(ml_ratings_path, chunksize=CHUNK_SIZE)
    build_movielens_sasrec_sequences_from_chunks(ml_ratings_path, chunksize=CHUNK_SIZE)

    # ----- Netflix -----
    netflix_titles_path = NETFLIX_DIR / "movie_titles.txt"
    if netflix_titles_path.exists():
        load_netflix_titles(netflix_titles_path)
    else:
        print("[Netflix] movie_titles.txt not found. skip titles.")

    netflix_ratings_df = load_or_mock_netflix_ratings(NETFLIX_DIR)
    netflix_to_cf(netflix_ratings_df)
    netflix_to_sasrec(netflix_ratings_df)

    # ===== TMDb 매핑 & 교차워크 =====
    netflix_titles_parquet = ARTIFACT_DIR / "netflix_titles.parquet"
    if netflix_titles_parquet.exists():
        idmap_netflix = build_tmdb_mapping_for_netflix_titles(netflix_titles_parquet)
    else:
        print("[TMDb] netflix_titles.parquet not found. skip Netflix mapping.")
        idmap_netflix = pd.DataFrame(columns=["netflix_movieId","tmdbId","year","title"])

    idmap_ml = build_tmdb_mapping_for_movielens_links(ml_links_path, ml_movies_path)

    if netflix_titles_parquet.exists():
        cross_nx_ml = build_crosswalk_netflix_to_movielens_via_titles(
            netflix_titles_parquet, ml_movies_path, ml_links_path, fuzzy_threshold=92
        )
        idmap_netflix_enh = combine_netflix_idmaps_with_crosswalk(idmap_netflix, cross_nx_ml)
    else:
        cross_nx_ml = pd.DataFrame()
        idmap_netflix_enh = idmap_netflix

    # ===== 404 복구(IMDb→TMDb 우선, 실패 시 제목/연도) =====
    idmap_netflix_fixed = validate_and_repair_tmdb_ids(
        idmap_netflix_enh, cross_nx_ml, ml_links_path, ml_movies_path
    )

    # ===== TMDb 메타 수집 =====
    if ENRICH_WITH_TMDB:
        pieces = []
        if not idmap_netflix_fixed.empty and "tmdbId" in idmap_netflix_fixed.columns:
            s = idmap_netflix_fixed["tmdbId"].dropna()
            if not s.empty: pieces.append(s.astype("Int64"))
        if not idmap_ml.empty and "tmdbId" in idmap_ml.columns:
            s = idmap_ml["tmdbId"].dropna()
            if not s.empty: pieces.append(s.astype("Int64"))

        if pieces:
            tmdb_ids = pd.concat(pieces, ignore_index=True).dropna().astype(int).tolist()
            if tmdb_ids: enrich_tmdb_metadata(tmdb_ids)
        else:
            log.warning("[TMDb] 수집할 tmdbId가 없습니다.")

    # ===== tmdb 기준 시퀀스 =====
    if not idmap_netflix_fixed.empty:
        build_crosswalk_and_sequences_tmdb(idmap_netflix_fixed, idmap_ml)
    else:
        print("[TMDb] Netflix->tmdb 매핑이 없어 tmdb 시퀀스 재작성 스킵")

    # ===== CBF =====
    if BUILD_CBF and ENRICH_WITH_TMDB:
        try:
            _build_cbf_features_generic(
                idmap_netflix_fixed.rename(columns={"netflix_movieId":"netflix_movieId"}),
                id_col_name="netflix_movieId", prefix="netflix"
            )
            _build_cbf_features_generic(
                idmap_ml.rename(columns={"movieId": "movielens_movieId"}),
                id_col_name="movielens_movieId", prefix="movielens"
            )
        except Exception as e:
            log.warning(f"[CBF] building CBF failed: {e}")

    report_tmdb_coverage()
    print("✅ All preprocessing (NX→ML title-match, TMDb 404 repair, meta & CBF) completed.")

if __name__ == "__main__":
    main()
