import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import base

# ============================================================
# 1. 평가용 Dataset 정의 (Leave-One-Out 방식)
# ------------------------------------------------------------
# 각 사용자 시퀀스에서 마지막 영화를 정답(target)으로 두고,
# 그 이전 시청 이력을 입력 시퀀스로 사용한다.
# padding은 앞쪽에 0을 채워 고정 길이(max_len)로 맞춘다.
# ============================================================

import json
import torch
from torch.utils.data import Dataset, DataLoader

# 준비: 학습 때 쓰던 vocab 그대로
MAX_LEN = 50
MAX_GEN = 6

# ---------------------------------------------------------
# JSON 시퀀스 데이터셋 (MovieLens용)
# ---------------------------------------------------------
class JSONEvalDataset(Dataset):
    def __init__(self, json_path, item2idx, tmdb2director, tmdb2genres,
                 max_len=MAX_LEN, max_gen=MAX_GEN):
        self.item2idx, self.tmdb2director, self.tmdb2genres = item2idx, tmdb2director, tmdb2genres
        self.max_len, self.max_gen = max_len, max_gen
        self.samples = []
        data = json.load(open(json_path, "r", encoding="utf-8"))["user2seq"]
        for _, seq in data.items():
            seq = [int(t) for t in seq]
            idx_seq = [item2idx.get(t, 0) for t in seq]
            idx_seq = [x for x in idx_seq if x != 0]
            if len(idx_seq) < 2: continue
            self.samples.append([int(t) for t in seq])  # tmdb 시퀀스 보관

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        tmdb_seq = self.samples[i]
        hist_tmdb, tgt_tmdb = tmdb_seq[:-1], tmdb_seq[-1]
        hist_tmdb = hist_tmdb[-self.max_len:]
        pad = [0] * (self.max_len - len(hist_tmdb))
        # item ids
        seq_items = pad + [self.item2idx.get(t, 0) for t in hist_tmdb]
        # directors
        seq_dirs  = pad + [self.tmdb2director.get(t, 0) for t in hist_tmdb]
        # genres (고정길이 패딩)
        seq_gens = []
        for t in hist_tmdb:
            g = list(self.tmdb2genres.get(t, []))[:self.max_gen]
            g += [0] * (self.max_gen - len(g))
            seq_gens.append(g)
        if pad:
            seq_gens = [[0]*self.max_gen]*len(pad) + seq_gens
        # target
        tgt = self.item2idx.get(tgt_tmdb, 0)
        seen = list(set(seq_items))  # item id 기준
        return (torch.LongTensor(seq_items),
                torch.LongTensor([tgt]),
                torch.LongTensor(seq_dirs),
                torch.LongTensor(seq_gens),  # [L, G]
                seen)   

# ---------------------------------------------------------
# collate_fn: 길이가 다른 'seen'은 리스트로 유지하고 나머지만 stack
# ---------------------------------------------------------
def eval_collate(batch):
    seqs, tgts, dirs, gens, seens = zip(*batch)
    return (torch.stack(seqs), torch.stack(tgts),
            torch.stack(dirs), torch.stack(gens), seens)


# ============================================================
# 2. Top-K 지표 계산 함수
# ------------------------------------------------------------
# 모델이 예측한 점수(logits)에서 상위 K개 아이템을 뽑고,
# 정답 아이템의 순위를 기준으로 HR@K, NDCG@K 계산.
# ============================================================

def topk_metrics(rank, k):
    """단일 샘플의 순위를 기준으로 HR/NDCG 계산"""
    hr = 1.0 if rank < k else 0.0
    ndcg = 1.0 / np.log2(rank + 2) if rank < k else 0.0
    return hr, ndcg


# ============================================================
# 3. 평가 루프
# ------------------------------------------------------------
# 모델의 forward 출력을 기반으로 HitRate/NDCG를 평균 산출한다.
# - seen mask를 적용하여 사용자가 이미 본 영화는 제외
# - K=(5,10,20) 등 다양한 cut-off 기준을 동시에 계산
# ============================================================

@torch.no_grad()
def evaluate(model, dl, num_items, K=(5,10,20), device="cuda"):
    model.eval()
    sums = {f"HR@{k}":0.0 for k in K} | {f"NDCG@{k}":0.0 for k in K}
    n=0
    for seq, tgt, dseq, gseq, seen in dl:
        seq, tgt = seq.to(device), tgt.to(device).view(-1)
        dseq, gseq = dseq.to(device), gseq.to(device)      # 메타 주입
        out = model(seq, dseq, gseq)
        logits = out[0] if isinstance(out, tuple) else out
        scores = logits[:, -1, :]
        # 마스킹
        for b in range(seq.size(0)):
            scores[b, 0] = -1e9
            for s in seen[b]:
                if s < num_items: scores[b, s] = -1e9
        _, topk = scores.topk(k=max(K), dim=-1)
        for b in range(seq.size(0)):
            n+=1
            arr = topk[b].tolist()
            if int(tgt[b]) in arr:
                rank = arr.index(int(tgt[b]))
                for k in K:
                    if rank < k:
                        sums[f"HR@{k}"]   += 1
                        sums[f"NDCG@{k}"] += 1/np.log2(rank+2)
    for k in K:
        sums[f"HR@{k}"]/= n
        sums[f"NDCG@{k}"]/= n
    return sums

@torch.no_grad()
def evaluate_legacy(model, dl, num_items, K=(5,10,20), device="cuda"):
    model.eval()

    # 누적 결과 저장 딕셔너리
    sums = {f"HR@{k}": 0.0 for k in K}
    sums.update({f"NDCG@{k}": 0.0 for k in K})
    n = 0  # 총 평가 샘플 수

    for seq, tgt, seen in dl:
        seq = seq.to(device)
        tgt = tgt.to(device).view(-1)

        B, L = seq.size()

        # meta feature 비활성화 상태 (감독/장르 없이)
        dirs = torch.zeros_like(seq)  # 감독 id placeholder
        gens = torch.zeros(B, L, 6, dtype=torch.long, device=device)  # 장르 placeholder

        # 모델 forward: [B, L, num_items]
        out = model(seq, dirs, gens)
        if isinstance(out, tuple):
            logits = out[0]     # 첫 번째 요소만 사용
        else:
            logits = out
        scores = logits[:, -1, :]  # 마지막 시점의 예측 결과만 사용

        # pad(0) 및 이미 시청한 아이템 마스킹
        for b in range(B):
            scores[b, 0] = -1e9
            for s in seen[b]:
                if s < num_items:
                    scores[b, s] = -1e9

        # 상위 K개 후보 추출
        _, topk = scores.topk(k=max(K), dim=-1)  # [B, maxK]

        # 각 사용자별 Hit/NDCG 계산
        for b in range(B):
            arr = topk[b].tolist()
            n += 1
            if int(tgt[b]) in arr:
                rank = arr.index(int(tgt[b]))
            else:
                rank = 10**9  # top-K 밖
            for k in K:
                hr, ndcg = topk_metrics(rank, k)
                sums[f"HR@{k}"] += hr
                sums[f"NDCG@{k}"] += ndcg

    # 평균화
    for k in K:
        sums[f"HR@{k}"] /= n
        sums[f"NDCG@{k}"] /= n

    return sums


# ============================================================
# 4. 실행부
# ------------------------------------------------------------
# MovieLens 데이터 로드 → Dataset → DataLoader 구성
# 학습된 SASRec(meta) 모델 로드 후 evaluate() 실행.
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Netflix 학습용 데이터에서 item2idx 생성 (Netflix 학습용 parquet 파일들 그대로 사용)
    import pandas as pd
    df_netflix = pd.read_parquet("artifacts/netflix_ratings_for_sasrec.parquet")
    meta = pd.read_parquet("artifacts/items_meta_for_rec.parquet")      
    
    # 학습 때와 동일한 매핑 생성    
    tmdb2director = dict(zip(meta["tmdbId"], meta["director_id"]))
    tmdb2genres = dict(zip(meta["tmdbId"], meta["genre_ids"]))

    df_netflix["tmdbId"] = df_netflix["tmdbId"].astype(int)
    unique_items = sorted(df_netflix["tmdbId"].unique())
    item2idx = {tmdb: idx + 1 for idx, tmdb in enumerate(unique_items)}
    print(f"[INFO] item2idx 생성 완료: {len(item2idx)} items")

    # MovieLens JSON 시퀀스 로드
    eval_ds = JSONEvalDataset("artifacts/movielens_sasrec_sequences.json",
                            item2idx, tmdb2director, tmdb2genres)
    eval_dl = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=2,
                     collate_fn=eval_collate)
    print(f"[INFO] Evaluation samples: {len(eval_ds)} users")

    # 모델 로드
    from pre import SASRec
    model = SASRec(
        num_items=len(item2idx),
        n_directors=24173,
        n_genres=19,
        hidden_dim=256,
        n_heads=4,
        n_layers=4,
        max_len=100,
    ).to(device)

    paths = base.get_model_pths("trained/save_by_epoch/SASRec_feature_meta_schedular_4", reverse=True)
    for path in paths:
        state_dict = torch.load(path, map_location=device)
        #state_dict = torch.load("trained/trained/save_by_epoch/SASRec_feature_meta_schedular_2/epoch_300_11-09_07-28.pth", map_location=device)
        model.load_state_dict(state_dict)

        print(f"\n[for {path}]")
        # 평가
        results = evaluate(model, eval_dl, num_items=len(item2idx), K=(5,10,20), device=device)
        print("\n=== Evaluation Results (MovieLens JSON) ===")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

