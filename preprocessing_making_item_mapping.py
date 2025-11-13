import pandas as pd
import numpy as np

items = pd.read_parquet("artifacts/tmdb_items_enriched_update.parquet")

# 1) 중복 제거
items = items.drop_duplicates(subset=["tmdbId"], keep="first")

# 2) 결측 처리
items["director"] = items["director"].fillna("Unknown")
items["genres_tmdb"] = items["genres_tmdb"].fillna("")

# 3) 장르 파싱 → 인덱스
items["genres_list"] = items["genres_tmdb"].apply(lambda s: [g for g in s.split("|") if g])
all_genres = sorted({g for lst in items["genres_list"] for g in lst})
genre2idx = {g:i+1 for i,g in enumerate(all_genres)}  # 0은 패딩
items["genre_ids"] = items["genres_list"].apply(lambda lst: [genre2idx[g] for g in lst])

# 4) 감독 인덱스
all_directors = sorted(items["director"].unique())
director2idx = {d:i+1 for i,d in enumerate(all_directors)}  # 0은 패딩/미사용
items["director_id"] = items["director"].map(director2idx).astype(int)

# 5) runtime 클리핑 + z-score
runtime = items["runtime"].fillna(items["runtime"].median())
runtime = runtime.clip(lower=40, upper=240)
items["runtime_z"] = (runtime - runtime.mean()) / (runtime.std(ddof=0) + 1e-8)

# 6) popularity 로그 + z-score
pop = items["popularity"].fillna(items["popularity"].median())
pop = np.log1p(pop)
items["popularity_z"] = (pop - pop.mean()) / (pop.std(ddof=0) + 1e-8)

# 7) 선택: 연도 버킷(10년 단위)
items["release_year"] = items["release_year"].fillna(0).astype(int)
items["year_bucket"] = (items["release_year"] // 10) * 10

# 8) 최종 저장(모델 입력용 메타)
cols_out = [
    "tmdbId", "director_id", "genre_ids",
    "runtime_z", "popularity_z", "year_bucket"
]
items[cols_out].to_parquet("artifacts/items_meta_for_rec.parquet", index=False)

# 9) 매핑 사전 저장(재현성)
pd.Series(director2idx).to_json("artifacts/director2idx.json")
pd.Series(genre2idx).to_json("artifacts/genre2idx.json")
