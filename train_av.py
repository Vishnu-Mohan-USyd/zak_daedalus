#!/usr/bin/env python3
"""Train AVWordResNet from scratch using the shared processed/splits.pt split.

--modality-mix assigns equal thirds of training samples to audio, video, and
paired input. Select checkpoints by mean accuracy across those conditions.
Mixed-training outputs are named av_fused_modality_mix.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from train import spec_augment, stratified_split
from paired_dataset import PairedAVDataset
from model_av import AVWordResNet


# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(SCRIPT_DIR, "processed", "splits.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "av_fused.pt")
CURVE_PNG = os.path.join(SCRIPT_DIR, "analysis", "av_fused_curves.png")
CURVE_CSV = os.path.join(SCRIPT_DIR, "analysis", "av_fused_curves.csv")

BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TEST_SIZE = 0.33
RANDOM_SEED = 42
ALPHA_INIT = 0.2
T_STRIDE = 2          # 100 cached frames → 50 input frames (50 fps)
USE_BF16 = True
USE_COMPILE = True

# Each worker opens the video memmap.
NUM_WORKERS = 4


class _AVAugmentedView(torch.utils.data.Dataset):
    """Add the mel channel dimension and optional training augmentation."""

    def __init__(self, base: PairedAVDataset, indices: np.ndarray, augment: bool):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = int(self.indices[k])
        mel, video, label = self.base[idx]
        # mel: (80, 99) → (1, 80, 99)
        mel = mel.unsqueeze(0)
        if self.augment:
            mel = spec_augment(mel)
            if torch.rand(1).item() < 0.5:
                video = torch.flip(video, dims=[-1])
        return mel, video, label


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def _save_curves(history: dict, num_classes: int, best_val_acc: float,
                 curve_png: str = CURVE_PNG) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="train")
    ax.plot(epochs, history["val_acc"], label="val")
    for condition in ("a", "v", "av"):
        key = f"val_acc_{condition}"
        if key in history:
            ax.plot(epochs, history[key], label=f"val {condition.upper()}", alpha=0.7)
    if "alpha" in history:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["alpha"], color="tab:gray",
                 linestyle="--", label="α")
        ax2.set_ylabel("α (gate)")
        ax2.legend(loc="upper right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy (best val: {best_val_acc:.1%}, {num_classes} classes)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(curve_png), exist_ok=True)
    fig.savefig(curve_png, dpi=130)
    plt.close(fig)


def _write_curves_csv(history: dict, curve_csv: str = CURVE_CSV) -> None:
    os.makedirs(os.path.dirname(curve_csv), exist_ok=True)
    extra_keys = [f"val_{metric}_{condition}" for condition in ("a", "v", "av")
                  for metric in ("loss", "acc") if f"val_{metric}_{condition}" in history]
    with open(curve_csv, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,alpha,epoch_time_s,peak_gpu_gib"
                + "".join(f",{key}" for key in extra_keys) + "\n")
        n = len(history["train_loss"])
        for i in range(n):
            f.write(
                f"{i+1},{history['train_loss'][i]:.6f},"
                f"{history['train_acc'][i]:.6f},"
                f"{history['val_loss'][i]:.6f},"
                f"{history['val_acc'][i]:.6f},"
                f"{history['alpha'][i]:.6f},"
                f"{history['epoch_time_s'][i]:.3f},"
                f"{history['peak_gpu_gib'][i]:.3f}"
                + "".join(f",{history[key][i]:.6f}" for key in extra_keys) + "\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality-mix", action="store_true",
                    help="train on equal thirds A-only, V-only and AV")
    args = ap.parse_args()
    stem = "av_fused_modality_mix" if args.modality_mix else "av_fused"
    model_path = os.path.join(SCRIPT_DIR, "models", f"{stem}.pt")
    curve_png = os.path.join(SCRIPT_DIR, "analysis", f"{stem}_curves.png")
    curve_csv = os.path.join(SCRIPT_DIR, "analysis", f"{stem}_curves.csv")
    conditions = ("a", "v", "av") if args.modality_mix else ("av",)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    base = PairedAVDataset(t_stride=T_STRIDE)
    labels = base.labels
    label_to_idx = base.label_to_idx
    idx_to_label = base.idx_to_label
    config = base.config
    num_classes = len(label_to_idx)

    print(f"Loaded {len(base)} paired samples, {num_classes} classes")
    print(f"Filter: dropped speakers={config['dropped_speakers']}, "
          f"dropped (speaker, group)={config['dropped_speaker_groups']}")

    # Reuse the audio baseline's train/validation split.
    if os.path.exists(SPLITS_PATH):
        s = torch.load(SPLITS_PATH, weights_only=False)
        train_idx, val_idx = s["train_idx"], s["val_idx"]
        print(f"Loaded shared splits from {SPLITS_PATH}")
    else:
        train_idx, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)
        os.makedirs(os.path.dirname(SPLITS_PATH), exist_ok=True)
        torch.save({
            "train_idx": train_idx, "val_idx": val_idx,
            "random_seed": RANDOM_SEED, "test_size": TEST_SIZE,
            "dataset_path": "processed/dataset_av.pt",
            "n_samples": len(train_idx) + len(val_idx),
        }, SPLITS_PATH)
        print(f"Wrote shared splits to {SPLITS_PATH}")

    train_hash = _hash_idx(train_idx)
    val_hash = _hash_idx(val_idx)
    print(f"train_idx sha256: {train_hash}")
    print(f"val_idx   sha256: {val_hash}")

    train_ds = _AVAugmentedView(base, train_idx, augment=True)
    val_ds = _AVAugmentedView(base, val_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    if args.modality_mix:
        print("Training: equal thirds A / V / AV; checkpoint metric: mean A / V / AV accuracy")
    print(f"Checkpoint: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = AVWordResNet(num_classes, alpha_init=ALPHA_INIT).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_visual = sum(p.numel() for p in model.visual.parameters())
    n_gate = sum(p.numel() for p in model.gate.parameters())
    print(f"Model parameters: total={n_total:,}, visual={n_visual:,}, gate={n_gate:,}")
    print(f"α init: {model.gate.alpha.item():.4f}")
    print(f"video t_stride: {T_STRIDE} (effective fps = {100 // T_STRIDE})")
    print(f"bf16 autocast: {USE_BF16}")
    print(f"torch.compile: {USE_COMPILE}")
    autocast_kw = {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": USE_BF16}

    # Save checkpoints from the original module to preserve state_dict keys.
    if USE_COMPILE:
        compiled = torch.compile(model, mode="default")
    else:
        compiled = model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6,
    )

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "alpha": [],
               "epoch_time_s": [], "peak_gpu_gib": []}
    if args.modality_mix:
        for condition in conditions:
            history[f"val_loss_{condition}"] = []
            history[f"val_acc_{condition}"] = []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epoch_times: list[float] = []
    peak_gpu_gib = 0.0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} "
          f"| {'Val Loss':>8} | {'Val Acc':>7} |   α   | {'Time':>5} | {'GPU':>5}")
    print("-" * 84)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        condition_counts = torch.zeros(3, dtype=torch.long, device=device)
        for mel, video, y in train_loader:
            mel = mel.to(device, non_blocking=True)
            video = video.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.modality_mix:
                # 0=A, 1=V, 2=AV. Shuffle assignments; rotate leftover samples
                # so epoch counts differ by at most one.
                modes = (torch.arange(y.size(0), device=device) + train_total + epoch - 1) % 3
                modes = modes[torch.randperm(y.size(0), device=device)]
                condition_counts += torch.bincount(modes, minlength=3)
            optimizer.zero_grad()
            with torch.autocast(**autocast_kw):
                if args.modality_mix:
                    logits = compiled(mel, video, audio_dead=(modes == 1), video_dead=(modes == 0))
                else:
                    logits = compiled(mel, video)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_losses = dict.fromkeys(conditions, 0.0)
        val_correct = dict.fromkeys(conditions, 0)
        val_total = 0
        with torch.no_grad():
            for mel, video, y in val_loader:
                mel = mel.to(device, non_blocking=True)
                video = video.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                for condition in conditions:
                    with torch.autocast(**autocast_kw):
                        if condition == "a":
                            logits = compiled(mel, None)
                        elif condition == "v":
                            logits = compiled(torch.zeros_like(mel), video)
                        else:
                            logits = compiled(mel, video)
                        loss = criterion(logits, y)
                    val_losses[condition] += loss.item() * y.size(0)
                    val_correct[condition] += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_losses = {c: loss / val_total for c, loss in val_losses.items()}
        val_accs = {c: correct / val_total for c, correct in val_correct.items()}
        val_loss = sum(val_losses.values()) / len(conditions)
        val_acc = sum(val_accs.values()) / len(conditions)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["alpha"].append(float(model.gate.alpha.detach()))
        if args.modality_mix:
            for condition in conditions:
                history[f"val_loss_{condition}"].append(val_losses[condition])
                history[f"val_acc_{condition}"].append(val_accs[condition])

        epoch_t = time.time() - t0
        epoch_times.append(epoch_t)
        epoch_peak = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                      if device.type == "cuda" else 0.0)
        peak_gpu_gib = max(peak_gpu_gib, epoch_peak)
        history["epoch_time_s"].append(epoch_t)
        history["peak_gpu_gib"].append(epoch_peak)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} "
              f"| {val_loss:8.4f} | {val_acc:6.1%} "
              f"| {history['alpha'][-1]:5.3f} | {epoch_t:4.1f}s "
              f"| {epoch_peak:4.1f}G")
        if args.modality_mix:
            print("        val " + " / ".join(f"{c.upper()}={val_accs[c]:.2%}" for c in conditions)
                  + f"; train counts A/V/AV={condition_counts.tolist()}", flush=True)

        # Save the highest validation accuracy across the full training run.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "config": config,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "epoch": best_epoch,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_idx_sha256": train_hash,
                "val_idx_sha256": val_hash,
                "alpha_init": ALPHA_INIT,
                "alpha_at_best": history["alpha"][-1],
            }
            if args.modality_mix:
                checkpoint.update({
                    "selection_metric": "mean_A_V_AV_accuracy",
                    "val_acc_by_condition": val_accs,
                    "val_loss_by_condition": val_losses,
                    "training_modality_probabilities": {c: 1 / 3 for c in conditions},
                    "training_modality_counts": dict(zip(conditions, condition_counts.tolist())),
                })
            torch.save(checkpoint, model_path)
        scheduler.step()
        if args.modality_mix:
            _write_curves_csv(history, curve_csv)

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"Best val loss:     {best_val_loss:.4f}")
    print(f"α: init={ALPHA_INIT:.4f}, final={history['alpha'][-1]:.4f}")
    print(f"Mean time/epoch: {np.mean(epoch_times):.1f}s "
          f"(median {np.median(epoch_times):.1f}s)")
    print(f"Peak GPU: {peak_gpu_gib:.2f} GiB")
    print(f"Saved model to: {model_path}")

    _save_curves(history, num_classes, best_val_acc, curve_png)
    _write_curves_csv(history, curve_csv)
    print(f"Saved curves to: {curve_png}")
    print(f"Saved curves CSV to: {curve_csv}")


if __name__ == "__main__":
    main()
