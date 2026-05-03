from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LABELS_PATH


class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=(1, 2), keepdim=True)
        var = weight.var(dim=(1, 2), keepdim=True, unbiased=False)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, inc: int, outc: int, kernel_size: int, stride: int = 1, gn: int = 8):
        super().__init__()
        self.same_channels = inc == outc
        self.ks = kernel_size
        self.gn = gn if outc % gn == 0 else 1
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, (self.ks - 1) // 2),
            nn.GroupNorm(self.gn, outc),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            return (x + x1) / 2
        return x1


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, gn: int = 8, factor: int = 2):
        super().__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, dim: int = 512):
        super().__init__()
        self.down1 = UnetDown(in_channels, dim, 1, gn=8, factor=2)
        self.down2 = UnetDown(dim, dim, 1, gn=8, factor=2)
        self.down3 = UnetDown(dim, dim, 1, gn=8, factor=2)
        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        out = self.avg_pooling(down3).squeeze(-1)
        return out, (down1, down2, down3)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, fc_dim: int, emb_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(fc_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DiffE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Optional[nn.Module], fc: LinearClassifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc

    def forward(self, x: torch.Tensor, ddpm_out: Optional[torch.Tensor] = None) -> tuple[None, torch.Tensor]:
        encoder_out, _ = self.encoder(x)
        fc_out = self.fc(encoder_out)
        return None, fc_out


def _load_labels(labels_path: Path) -> List[Dict[str, Any]]:
    with open(labels_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    labels = payload.get("labels", [])
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"labels.json is missing or invalid at {labels_path}")
    ordered = sorted(labels, key=lambda item: int(item["id"]))
    if len(ordered) != 25:
        raise ValueError(f"labels.json must contain 25 labels, found {len(ordered)}")
    return ordered


def _parse_checkpoint_name(filename: str) -> Dict[str, Optional[str]]:
    match = re.match(r"(?P<subject>S\d+)_classifier_(?P<classifier>[A-Za-z])", filename, re.IGNORECASE)
    if not match:
        return {"subject": None, "classifier": None}
    return {
        "subject": str(match.group("subject")).upper(),
        "classifier": str(match.group("classifier")).upper(),
    }


def validate_checkpoint(model_path: str, labels_path: str | Path = LABELS_PATH) -> Dict[str, Any]:
    path = Path(model_path)
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Model checkpoint must be a .pt or .pth file.")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")

    required_keys = {"model_state_dict", "arch", "n_classes", "channel_mean", "channel_std"}
    missing = sorted(required_keys.difference(checkpoint.keys()))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {', '.join(missing)}")

    arch = checkpoint.get("arch")
    if not isinstance(arch, dict):
        raise ValueError("Checkpoint architecture metadata is missing or invalid.")
    if str(arch.get("arch_type", "")).lower() != "diffe":
        raise ValueError(f"Unsupported checkpoint architecture: {arch.get('arch_type')!r}")

    n_classes = int(checkpoint.get("n_classes", 0))
    if n_classes != 25:
        raise ValueError(f"Checkpoint class count must be 25, found {n_classes}")

    num_channels = int(arch.get("num_channels", 0))
    if num_channels != 19:
        raise ValueError(f"Checkpoint channel count must be 19, found {num_channels}")

    channel_mean = np.asarray(checkpoint.get("channel_mean"), dtype=np.float32)
    channel_std = np.asarray(checkpoint.get("channel_std"), dtype=np.float32)
    if channel_mean.shape != (19,):
        raise ValueError(f"channel_mean must have shape (19,), found {channel_mean.shape}")
    if channel_std.shape != (19,):
        raise ValueError(f"channel_std must have shape (19,), found {channel_std.shape}")

    window_size = int(arch.get("window_size", 0))
    if window_size <= 0:
        raise ValueError("Checkpoint window_size must be a positive integer.")

    labels = _load_labels(Path(labels_path))
    parsed_name = _parse_checkpoint_name(path.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "path": str(path),
        "filename": path.name,
        "arch": "diffe",
        "subject": parsed_name["subject"],
        "classifier": parsed_name["classifier"],
        "n_classes": n_classes,
        "device": device,
        "window_size": window_size,
        "encoder_dim": int(arch.get("encoder_dim", 128)),
        "fc_dim": int(arch.get("fc_dim", 128)),
        "num_channels": num_channels,
        "labels": labels,
        "checkpoint": checkpoint,
    }


class Stage1DiffEAdapter:
    def __init__(self, model_path: str, labels_path: str | Path = LABELS_PATH, device: str | None = None):
        metadata = validate_checkpoint(model_path=model_path, labels_path=labels_path)
        self.metadata = metadata
        self.device = str(device or metadata["device"])
        self.window_size = int(metadata["window_size"])
        self.labels = metadata["labels"]
        self.labels_by_id = {int(item["id"]): item for item in self.labels}
        self.channel_mean = np.asarray(metadata["checkpoint"]["channel_mean"], dtype=np.float32)
        self.channel_std = np.maximum(np.asarray(metadata["checkpoint"]["channel_std"], dtype=np.float32), 1e-6)
        self.model = self._build_model(metadata["checkpoint"]).to(self.device)
        self.model.eval()

    def _build_model(self, checkpoint: Dict[str, Any]) -> DiffE:
        arch = checkpoint["arch"]
        encoder = Encoder(in_channels=int(arch["num_channels"]), dim=int(arch["encoder_dim"]))
        fc = LinearClassifier(int(arch["encoder_dim"]), int(arch["fc_dim"]), emb_dim=int(checkpoint["n_classes"]))
        model = DiffE(encoder, None, fc)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def info(self) -> Dict[str, Any]:
        return {
            "path": self.metadata["path"],
            "filename": self.metadata["filename"],
            "arch": self.metadata["arch"],
            "subject": self.metadata["subject"],
            "classifier": self.metadata["classifier"],
            "n_classes": self.metadata["n_classes"],
            "device": self.device,
            "window_size": self.window_size,
            "validated": True,
        }

    def _window_tensor(self, tensor_19_by_t: np.ndarray) -> np.ndarray:
        if tensor_19_by_t.ndim != 2:
            raise ValueError(f"Stage 1 tensor must be 2D, found shape {tensor_19_by_t.shape}")
        if tensor_19_by_t.shape[0] != 19:
            raise ValueError(f"Stage 1 tensor must have 19 channels, found {tensor_19_by_t.shape[0]}")
        if tensor_19_by_t.shape[1] < self.window_size:
            raise ValueError(
                f"Stage 1 tensor is too short for windowing: {tensor_19_by_t.shape[1]} < {self.window_size}"
            )

        normalized = (tensor_19_by_t.astype(np.float32) - self.channel_mean[:, None]) / self.channel_std[:, None]
        windows = [
            normalized[:, start : start + self.window_size]
            for start in range(0, normalized.shape[1] - self.window_size + 1, self.window_size)
        ]
        if not windows:
            raise ValueError("No Stage 1 windows could be created from the tensor.")
        return np.stack(windows, axis=0).astype(np.float32)

    def predict_slot(self, tensor_19_by_t: np.ndarray, slot: int, top_k: int = 8) -> Dict[str, Any]:
        batch_np = self._window_tensor(tensor_19_by_t)
        batch = torch.from_numpy(batch_np).to(self.device)
        with torch.no_grad():
            _, logits = self.model(batch)
        avg_logits = logits.mean(dim=0)
        probabilities = torch.softmax(avg_logits, dim=0).detach().cpu().numpy().astype(np.float32)

        if probabilities.shape != (25,):
            raise ValueError(f"Stage 1 output must have length 25, found {probabilities.shape}")

        top_indices = np.argsort(-probabilities)[: max(1, int(top_k))]
        top_items = []
        for index in top_indices:
            label = self.labels_by_id[int(index)]
            top_items.append(
                {
                    "label_id": int(index),
                    "word": str(label["word"]),
                    "arabic": str(label["arabic"]),
                    "probability": float(probabilities[int(index)]),
                }
            )

        return {
            "slot": int(slot),
            "probabilities": {str(index): float(prob) for index, prob in enumerate(probabilities.tolist())},
            "top_k": top_items,
        }

    def predict_slots(self, tensors: List[np.ndarray], top_k: int = 8) -> List[Dict[str, Any]]:
        results = []
        for index, tensor in enumerate(tensors, start=1):
            results.append(self.predict_slot(tensor, slot=index, top_k=top_k))
        return results
