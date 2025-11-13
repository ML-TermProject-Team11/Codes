import torch
import base
import wandb
import os
try:
  if __file__.split('.')[-1] == 'py':
    from tqdm import tqdm
  else:
    from tqdm.notebook import tqdm
except:
  from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from pre import SASRecDataset, SASRec
import pandas as pd
import numpy as np

# --------------------------------------------------
def train_iteration(config: dict):
    # wandb 초기화
    wandb.init(
        project=config['project_name'],
        name=config['run_name'],
        config=config
    )

    device = config['device']

    # --------------------------------------------------
    # 1. 데이터 로드
    print("[INFO] Loading data...")
    df = pd.read_parquet("artifacts/netflix_ratings_for_sasrec.parquet")
    meta = pd.read_parquet("artifacts/items_meta_for_rec.parquet")

    # tmdbId → 감독/장르 매핑 dict
    tmdb2director = dict(zip(meta["tmdbId"], meta["director_id"]))
    tmdb2genres = dict(zip(meta["tmdbId"], meta["genre_ids"]))

    # item2idx는 Netflix 전체 tmdbId 기준으로 인덱싱
    df["tmdbId"] = df["tmdbId"].astype(int)
    unique_items = sorted(df["tmdbId"].unique())
    item2idx = {tmdb: idx + 1 for idx, tmdb in enumerate(unique_items)}

    # 데이터셋/로더
    dataset = SASRecDataset(df, tmdb2director, tmdb2genres, item2idx, max_len=config["max_len"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)

    print(f"[INFO] Loaded {len(dataset):,} user sequences")
    print(f"[INFO] num_items: {len(item2idx):,}, num_directors: {len(set(meta['director_id'])):,}, num_genres: {len(set(np.concatenate(meta['genre_ids'].values))):,}")

    # --------------------------------------------------
    # 2. 모델 준비
    model = SASRec(
        num_items=len(item2idx),
        n_directors=max(meta["director_id"]),
        n_genres=max([g for lst in meta["genre_ids"] for g in lst]),
        hidden_dim=256
    ).to(device)

    # 이전 state_dict 로드
    if config.get("state_dict") is not None:
        model.load_state_dict(config["state_dict"])
        print(f"[INFO] Loaded pretrained weights (start_epoch={config['start_epoch']})")

    # ---------------------- Optimizer / Scheduler ----------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    scheduler: str = config.get("scheduler")
    if scheduler and scheduler.lower() == "onecyclelr":
        # OneCycleLR : 초반 warmup, 중반 고LR, 후반 decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,                 # peak learning rate
            total_steps=config["total_epochs"],
            anneal_strategy='cos', 
            div_factor=10,               # initial lr = max_lr/div_factor
            final_div_factor=10          # final lr = max_lr/final_div_factor
        )

    # 모델 출력 크기와 라벨 범위 정합성 확인
    assert model.output.out_features == (len(item2idx) + 1)
    batch = next(iter(loader))
    _, labels, _, _, _ = batch
    assert labels.max().item() <= len(item2idx)
    assert labels.dtype == torch.long


    # --------------------------------------------------
    # 3. 학습 루프
    lambda_align = 0.2  # 가중치
    for epoch in range(config["start_epoch"], config["total_epochs"] + 1):
        model.train()
        total_loss = 0
        loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{config['total_epochs']}")

        for seq, labels, dirs, genres, mask in loop:
            seq, labels = seq.to(device), labels.to(device)
            dirs, genres = dirs.to(device), genres.to(device)

            optimizer.zero_grad()
            logits, item_emb, meta_emb = model(seq, dirs, genres)

            loss_main = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss_align = torch.mean((item_emb.detach() - meta_emb) ** 2)
            loss = loss_main + lambda_align * loss_align

            loss.backward()
            optimizer.step()

            total_loss += loss_main.item()
            loop.set_postfix(loss=f"{loss_main.item():.2f}")

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}] Avg CE Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "CE_loss": avg_loss,
                "align_loss": loss_align.item(),
                "lr": optimizer.param_groups[0]["lr"]})

        # 10 epoch 단위 저장
        if epoch % config["save_per_epoch"] == 0 or epoch == config["total_epochs"]:
            base.save_model(
                model.state_dict(),
                trained=True,
                save_folder=config["save_folder"],
                model_name=f"epoch_{epoch}_{base.currtime_str()}"
            )
        scheduler.step()

    wandb.finish()
    print("[Training Completed]")

# --------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_name = "SASRec_feature_meta_schedular"
    ver = base.get_version_from(project_name)

    config = {}
    config["device"] = device
    config["start_epoch"] = 1
    config["state_dict"] = None  # base.set_and_torch_load_from('trained/save_by_epoch/SASRec_feature_meta', 0)
    config["project_name"] = project_name
    config["save_folder"] = os.path.join(project_name, f"v{ver}")
    config["save_per_epoch"] = 10
    config["scheduler"] = "onecyclelr"
    config["total_epochs"] = 300
    config["batch_size"] = 256
    config["max_len"] = 100
    config["lr"] = 1e-3
    config["run_name"] = f"(onecycle_align_lr_v{ver}:{config['lr']})"
    config["seed"] = 500

    base.set_cuda_seed_print_device(device, config["seed"])
    base.add_version_from(project_name)
    train_iteration(config)