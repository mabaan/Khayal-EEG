#!/usr/bin/env python3
"""
Stage 1 trial-level CV using the DiffE model (Diffusion + Encoder + FC).

This refactors the original `diffe/` code into a single script that
shares the Stage 1 data/trial/metric pipeline (stage1_common) so it
fits the repo and is consistent with EEGNet/Conformer flows.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from ema_pytorch import EMA

from stage1_common import (
    Trial,
    build_trials,
    discover_subjects,
    filter_trials_for_classifier,
    load_full_dataset,
    confusion_from_trials,
    save_results,
    save_loss_plot,
)


# --------------------------------------------------------------------------- #
# DiffE model components (lifted from diffe/models.py)
# --------------------------------------------------------------------------- #


class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=(1, 2), keepdim=True)
        var = weight.var(dim=(1, 2), keepdim=True, unbiased=False)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(
            x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, inc: int, outc: int, kernel_size: int, stride=1, gn=8):
        super().__init__()
        self.same_channels = inc == outc
        self.ks = kernel_size
        self.gn = gn if outc % gn == 0 else 1
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, (self.ks * 1 - 1) // 2),
            nn.GroupNorm(self.gn, outc),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            out = (x + x1) / 2
        else:
            out = x1
        return out


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super().__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)
        return x


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super().__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode="nearest")
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer(x)
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.d1_out = n_feat * 1
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 3

        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.down1 = UnetDown(in_channels, self.d1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.d2_out, self.d3_out, 1, gn=8, factor=2)

        self.up2 = UnetUp(self.d3_out, self.u2_out, 1, gn=8, factor=2)
        self.up3 = UnetUp(self.u2_out + self.d2_out, self.u3_out, 1, gn=8, factor=2)
        self.up4 = UnetUp(self.u3_out + self.d1_out, self.u4_out, 1, gn=8, factor=2)
        self.out = nn.Conv1d(self.u4_out + in_channels, in_channels, 1)

    def forward(self, x, t_emb):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        up1 = self.up2(down3)
        up2 = self.up3(torch.cat([up1 + t_emb, down2], 1))
        up3 = self.up4(torch.cat([up2 + t_emb, down1], 1))
        out = self.out(torch.cat([up3, x], 1))

        down = (down1, down2, down3)
        up = (up1, up2, up3)
        return out, down, up


class Encoder(nn.Module):
    def __init__(self, in_channels, dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.e1_out = dim
        self.e2_out = dim
        self.e3_out = dim

        self.down1 = UnetDown(in_channels, self.e1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.e1_out, self.e2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.e2_out, self.e3_out, 1, gn=8, factor=2)

        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        out = self.avg_pooling(down3).squeeze(-1)
        return out, (down1, down2, down3)


class Decoder(nn.Module):
    def __init__(self, in_channels, n_feat=256, encoder_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.encoder_dim = encoder_dim

        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.up2 = UnetUp(self.encoder_dim, self.u2_out, 1, gn=8, factor=2)
        self.up3 = UnetUp(self.u2_out + self.encoder_dim, self.u3_out, 1, gn=8, factor=2)
        self.up4 = UnetUp(self.u3_out + self.encoder_dim, self.u4_out, 1, gn=8, factor=2)
        self.out = nn.Conv1d(self.u4_out + in_channels, in_channels, 1)

    def forward(self, x, ddpm_out):
        x_hat, down, up, t = ddpm_out
        up1 = self.up2(x)
        up2 = self.up3(torch.cat([up1 + t, down[1]], 1))
        up3 = self.up4(torch.cat([up2 + t, down[0]], 1))
        out = self.out(torch.cat([up3, x_hat], 1))
        return out


class LinearClassifier(nn.Module):
    def __init__(self, encoder_dim, fc_dim, emb_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(encoder_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(fc_dim, emb_dim),
        )

    def forward(self, x):
        return self.fc(x)


class DiffE(nn.Module):
    def __init__(self, encoder, decoder, fc):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc

    def forward(self, x, ddpm_out=None):
        encoder_out, _ = self.encoder(x)
        fc_out = self.fc(encoder_out)
        return None, fc_out


# --------------------------------------------------------------------------- #
# Simplified DDPM placeholder (we keep noise prediction loss only)
# --------------------------------------------------------------------------- #


class DDPM(nn.Module):
    def __init__(self, nn_model: nn.Module, betas: Tuple[float, float], n_T: int = 1000, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.nn_model = nn_model
        self.device = device
        self.n_T = n_T
        self.betas = torch.linspace(betas[0], betas[1], n_T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        t = torch.randint(0, self.n_T, (b,), device=x.device).long()
        noise = torch.randn_like(x)
        a_bar = self.alphas_bar[t].view(-1, 1, 1)
        x_noisy = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise
        t_emb = a_bar.sqrt().log().view(-1, 1, 1)  # simple scalar embed
        x_hat, down, up = self.nn_model(x_noisy, t_emb)
        # Return only the parts the decoder expects (noise unused downstream)
        return x_hat, down, up, t_emb


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _compute_channel_stats(X: torch.Tensor):
    mean = X.mean(dim=(0, 2))
    std = X.std(dim=(0, 2))
    std = torch.clamp(std, min=1e-6)
    return mean, std


def _apply_channel_norm(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (X - mean[None, :, None]) / std[None, :, None]


def window_eeg_tensor(X: torch.Tensor, Y: torch.Tensor, window_size: int, stride: int = None):
    if stride is None:
        stride = window_size
    if X.shape[-1] < window_size:
        raise ValueError("window_size is larger than the time dimension")
    windows, labels = [], []
    for x, y in zip(X, Y):
        n_time = x.shape[-1]
        max_start = n_time - window_size
        for start in range(0, max_start + 1, stride):
            windows.append(x[:, start : start + window_size])
            labels.append(y)
    Xw = torch.stack(windows, dim=0)
    Yw = torch.stack(labels, dim=0)
    return Xw, Yw


def get_dataloader_from_arrays(X_train: torch.Tensor, Y_train: torch.Tensor, X_test: torch.Tensor, Y_test: torch.Tensor, batch_size: int, batch_size2: int, shuffle: bool = True):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size2, shuffle=False)
    return train_loader, test_loader


def predict_trials_avg_logits(encoder: nn.Module, fc: nn.Module, X_test: torch.Tensor, window_size: int, stride: int, device: torch.device, num_classes: int) -> np.ndarray:
    encoder.eval()
    fc.eval()
    preds = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            xi = X_test[i].to(device)
            n_time = xi.shape[-1]
            max_start = n_time - window_size
            windows = []
            for start in range(0, max_start + 1, stride):
                windows.append(xi[:, start : start + window_size])
            windows = torch.stack(windows, dim=0).to(device)
            enc_out, _ = encoder(windows)
            logits = fc(enc_out)
            avg_logits = logits.mean(dim=0)
            preds.append(avg_logits.cpu().numpy())
    return np.stack(preds, axis=0)


# --------------------------------------------------------------------------- #
# Training / CV
# --------------------------------------------------------------------------- #


def train_fold(args, X: torch.Tensor, Y: torch.Tensor, train_indices: np.ndarray, test_indices: np.ndarray, fold_out_dir: Path, fold_name: str, n_classes: int, device: torch.device):
    batch_size = 32
    batch_size2 = 256

    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]

    mean, std = _compute_channel_stats(X_train)
    X_train = _apply_channel_norm(X_train, mean, std)
    X_test = _apply_channel_norm(X_test, mean, std)

    window_size = int(args.window_size)
    X_train_w, Y_train_w = window_eeg_tensor(X_train, Y_train, window_size=window_size, stride=window_size)
    X_val_w, Y_val_w = window_eeg_tensor(X_test, Y_test, window_size=window_size, stride=window_size)

    train_loader, test_loader = get_dataloader_from_arrays(
        X_train_w, Y_train_w, X_test, Y_test, batch_size, batch_size2, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val_w, Y_val_w), batch_size=batch_size2, shuffle=False)

    channels = X.shape[1]
    encoder_dim = args.encoder_dim
    fc_dim = args.fc_dim

    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=n_classes).to(device)
    diffe = DiffE(encoder, None, fc).to(device)

    criterion_class = nn.CrossEntropyLoss()

    base_lr, lr = args.lr_base, args.lr_max
    optim_cls = torch.optim.AdamW(diffe.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10)

    scheduler_cls = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optim_cls,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=args.lr_step,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(args.epochs):
        diffe.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.long().to(device)

            optim_cls.zero_grad()
            _, fc_out = diffe(x)
            loss = criterion_class(fc_out, y)
            loss.backward()
            optim_cls.step()

            scheduler_cls.step()
            fc_ema.update()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        diffe.eval()
        with torch.no_grad():
            vl_total, vl_batches = 0.0, 0
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.long().to(device)
                _, fc_out_v = diffe(xv)
                vl_total += criterion_class(fc_out_v, yv).item()
                vl_batches += 1
            val_loss = vl_total / max(vl_batches, 1)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[{fold_name}] Epoch {epoch + 1}/{args.epochs}: train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

    # inference logits for trial-level voting
    sample_logits = predict_trials_avg_logits(
        diffe.encoder, fc_ema, X_test, window_size=window_size, stride=window_size, device=device, num_classes=n_classes
    )

    if args.save_folds:
        # Save EMA-smoothed FC weights so inference matches CV evaluation
        fc_state_raw = diffe.fc.state_dict()
        try:
            diffe.fc.load_state_dict(fc_ema.ema_model.state_dict())
        except Exception:
            # Fallback: if EMA doesn't expose ema_model, keep raw weights
            pass

        ckpt = {
            "model_state_dict": diffe.state_dict(),
            "n_classes": n_classes,
            "arch": {
                "arch_type": "diffe",
                "encoder_dim": encoder_dim,
                "fc_dim": fc_dim,
                "num_channels": channels,
                "window_size": window_size,
            },
            "channel_mean": mean.cpu().numpy(),
            "channel_std": std.cpu().numpy(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, fold_out_dir / f"{fold_name}.pt")
        save_loss_plot(
            train_losses,
            val_losses,
            fold_out_dir / f"{fold_name}_losses.png",
            title=f"{fold_name} Loss",
        )

        # Restore original FC weights to avoid side-effects if function continues
        try:
            diffe.fc.load_state_dict(fc_state_raw)
        except Exception:
            pass

    return sample_logits


def run_cv(trials: List[Trial], X: np.ndarray, y: np.ndarray, args, n_classes: int, device: torch.device, classifier_label: str, save_folds: bool, fold_out_dir: Path):
    y_trials = np.array([t.class_idx for t in trials])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cm_total = np.zeros((n_classes, n_classes), dtype=np.int64)
    fold_accs: List[float] = []

    # convert once to torch
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(trials)), y_trials), 1):
        test_trials = [trials[i] for i in test_idx]
        train_trials = [trials[i] for i in train_idx]

        train_indices = np.array([i for t in train_trials for i in t.indices], dtype=int)
        test_indices = np.array([i for t in test_trials for i in t.indices], dtype=int)

        sample_logits = train_fold(
            args,
            X_t,
            y_t,
            train_indices,
            test_indices,
            fold_out_dir=fold_out_dir,
            fold_name=f"classifier_{classifier_label}_fold{fold}",
            n_classes=n_classes,
            device=device,
        )

        cm_fold = confusion_from_trials(test_trials, test_indices, sample_logits, n_classes)
        cm_total += cm_fold

        fold_acc = float(np.trace(cm_fold) / cm_fold.sum()) if cm_fold.sum() > 0 else 0.0
        fold_accs.append(fold_acc)
        print(f"Fold {fold}/{args.folds}: trials={len(test_trials)}  acc={fold_acc:.4f}")

    return cm_total, fold_accs


# --------------------------------------------------------------------------- #
# Full classifier training (for Stage 2 non-fold usage)
# --------------------------------------------------------------------------- #


def train_full_classifier(
    trials: List[Trial],
    X: np.ndarray,
    y: np.ndarray,
    args,
    n_classes: int,
    device: torch.device,
    out_path: Path,
) -> None:
    """Train a DiffE classifier on ALL trials and save a single checkpoint."""
    all_indices = np.array([i for t in trials for i in t.indices], dtype=int)
    X_full = torch.from_numpy(X).float()
    y_full = torch.from_numpy(y).long()

    # Channel normalisation computed on the training subset
    X_train = X_full[all_indices]
    mean, std = _compute_channel_stats(X_train)
    X_norm = _apply_channel_norm(X_full, mean, std)

    window_size = int(args.window_size)
    X_train_w, Y_train_w = window_eeg_tensor(
        X_norm[all_indices], y_full[all_indices],
        window_size=window_size, stride=window_size,
    )

    channels = X.shape[1]
    encoder_dim = args.encoder_dim
    fc_dim = args.fc_dim

    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=n_classes).to(device)
    diffe = DiffE(encoder, None, fc).to(device)

    criterion_class = nn.CrossEntropyLoss()
    base_lr, lr = args.lr_base, args.lr_max
    optim_cls = torch.optim.AdamW(diffe.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10)

    scheduler_cls = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optim_cls,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=args.lr_step,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )

    train_loader = DataLoader(
        TensorDataset(X_train_w, Y_train_w),
        batch_size=32,
        shuffle=True,
    )

    train_losses_full: List[float] = []
    for epoch in range(args.epochs):
        diffe.train()
        epoch_loss, n_batches = 0.0, 0
        for x, yb in train_loader:
            x, yb = x.to(device), yb.long().to(device)
            optim_cls.zero_grad()
            _, fc_out = diffe(x)
            loss = criterion_class(fc_out, yb)
            loss.backward()
            optim_cls.step()
            scheduler_cls.step()
            fc_ema.update()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses_full.append(epoch_loss / max(n_batches, 1))
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{args.epochs}  loss={train_losses_full[-1]:.4f}")

    # Save with EMA FC weights
    try:
        diffe.fc.load_state_dict(fc_ema.ema_model.state_dict())
    except Exception:
        pass

    ckpt = {
        "model_state_dict": diffe.state_dict(),
        "n_classes": n_classes,
        "arch": {
            "arch_type": "diffe",
            "encoder_dim": encoder_dim,
            "fc_dim": fc_dim,
            "num_channels": channels,
            "window_size": window_size,
        },
        "channel_mean": mean.cpu().numpy(),
        "channel_std": std.cpu().numpy(),
        "train_losses": train_losses_full,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    save_loss_plot(
        train_losses_full,
        None,
        out_path.with_suffix(".png"),
        title=f"{out_path.stem} Loss",
    )
    print(f"  Saved full classifier to {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 trial-level CV using DiffE")
    ap.add_argument("--root", default="data/Preliminary_Preprocessed")
    ap.add_argument("--subject", default=None, help="Optional subject ID (e.g., S4)")
    ap.add_argument("--output_dir", default="cv_results")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr_base", type=float, default=5e-5)
    ap.add_argument("--lr_max", type=float, default=8e-4)
    ap.add_argument("--lr_step", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_folds", action="store_true",
                    help="If set, save each fold model checkpoint (encoder+fc) for Stage 2 ensembling.")
    ap.add_argument("--fold_out_dir", default=None,
                    help="Directory to save fold models; defaults to output_dir.")

    ap.add_argument("--window_size", type=int, default=352)
    ap.add_argument("--ddpm_dim", type=int, default=128)  # unused in simplified classifier-only training
    ap.add_argument("--encoder_dim", type=int, default=128)
    ap.add_argument("--fc_dim", type=int, default=128)
    ap.add_argument("--n_T", type=int, default=1000)      # unused
    ap.add_argument("--alpha", type=float, default=0.1)   # unused

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subjects = [args.subject] if args.subject else discover_subjects(args.root)

    for subj in subjects:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print("\n" + "=" * 80)
        print(f"SUBJECT: {subj} (DiffE)")
        print("=" * 80)

        X, y, meta, idx_to_label = load_full_dataset(args.root, subj)
        n_classes = int(max(y.max(), len(idx_to_label) - 1) + 1)

        trials, session_map, trial_counts = build_trials(y, meta)
        trials_A = filter_trials_for_classifier(trials, session_map, trial_counts, "A")
        trials_B = filter_trials_for_classifier(trials, session_map, trial_counts, "B")

        print(f"Total trials: {len(trials)} | A: {len(trials_A)} | B: {len(trials_B)}")

        out_dir = Path(args.output_dir) / subj
        fold_dir = Path(args.fold_out_dir) if args.fold_out_dir else out_dir

        print("\nRunning CV for Classifier A (DiffE)...")
        cm_A, fold_accs_A = run_cv(
            trials_A, X, y, args, n_classes, device,
            classifier_label="A",
            save_folds=args.save_folds,
            fold_out_dir=fold_dir,
        )
        save_results(out_dir, "classifier_A", cm_A, fold_accs_A)

        print("\nRunning CV for Classifier B (DiffE)...")
        cm_B, fold_accs_B = run_cv(
            trials_B, X, y, args, n_classes, device,
            classifier_label="B",
            save_folds=args.save_folds,
            fold_out_dir=fold_dir,
        )
        save_results(out_dir, "classifier_B", cm_B, fold_accs_B)

        cm_sum = cm_A + cm_B
        save_results(out_dir, "classifier_sum", cm_sum)

        # Train full classifiers and save (for Stage 2 single-model usage)
        print("\nTraining full Classifier A (DiffE)...")
        train_full_classifier(trials_A, X, y, args, n_classes, device, out_dir / "classifier_A.pt")

        print("Training full Classifier B (DiffE)...")
        train_full_classifier(trials_B, X, y, args, n_classes, device, out_dir / "classifier_B.pt")

        print(f"Done: {subj}")


if __name__ == "__main__":
    main()
