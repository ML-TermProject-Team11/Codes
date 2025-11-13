import pandas as pd

# ---------- 1. 파일 로드 ----------
ratings_path = "artifacts/netflix_ratings.parquet"
idmap_path = "artifacts/idmap_netflix_movie_to_tmdb_enhanced_repaired.parquet"

ratings = pd.read_parquet(ratings_path)
idmap = pd.read_parquet(idmap_path)

print("=== 1. 원본 데이터 ===")
print(f"ratings: {ratings.shape}, idmap: {idmap.shape}")

# ---------- 2. ID 매핑 정제 ----------
idmap = idmap.dropna(subset=["tmdbId"])  # 결측 제거
idmap = idmap.drop_duplicates(subset=["netflix_movieId"], keep="first")
idmap["tmdbId"] = idmap["tmdbId"].astype(int)

print("\n=== 2. 매핑 정제 후 ===")
print(idmap.head())
print(f"유효 tmdbId 비율: {len(idmap) / 17781 * 100:.2f}%")

# ---------- 3. 병합 (Netflix → TMDb) ----------
merged = ratings.merge(
    idmap[["netflix_movieId", "tmdbId"]],
    left_on="movieId",
    right_on="netflix_movieId",
    how="inner"
)

print("\n=== 3. 병합 결과 ===")
print(merged.head())
print(f"병합 후 레코드 수: {len(merged)}")

# ---------- 4. 컬럼명 통일 및 timestamp 변환 ----------
merged = merged.rename(columns={
    "customerId": "userId",
    "date": "timestamp"
})
merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
merged = merged.dropna(subset=["timestamp"])
merged["timestamp"] = merged["timestamp"].astype(int) // 1_000_000_000

sasrec_df = merged[["userId", "tmdbId", "timestamp"]].copy()

# ---------- 5. 정렬 및 짧은 시퀀스 제거 ----------
sasrec_df = sasrec_df.sort_values(["userId", "timestamp"])
user_lengths = sasrec_df["userId"].value_counts()
valid_users = user_lengths[user_lengths >= 2].index
sasrec_df = sasrec_df[sasrec_df["userId"].isin(valid_users)]

print("\n=== 5. 시퀀스 정제 후 ===")
print(f"사용자 수: {sasrec_df['userId'].nunique()}")
print(f"총 레코드 수: {len(sasrec_df)}")
print(sasrec_df.head())

# ---------- 6. 저장 ----------
output_path = "artifacts/netflix_ratings_for_sasrec.parquet"
sasrec_df.to_parquet(output_path, index=False)
print(f"\n✅ 저장 완료: {output_path}")
