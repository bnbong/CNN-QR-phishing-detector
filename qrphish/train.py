"""학습 루프: AdamW + BCEWithLogits + val AUROC 조기종료 (스펙 6절 model 블록)."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from qrphish.evaluate import auroc, pick_threshold, predict_probs

__all__ = ["TrainResult", "set_seed", "pick_device", "train_model"]


def set_seed(seed: int) -> None:
    """torch/numpy/random 시드 고정 + cudnn 결정론."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    best_epoch: int
    best_val_auroc: float
    threshold: float
    epochs_run: int
    history: list[dict] = field(default_factory=list)


def _pos_weight(loader, class_weight: str | None) -> torch.Tensor | None:
    """class_weight="balanced"면 양성 가중치 = n_neg/n_pos."""
    if class_weight != "balanced":
        return None
    ds = loader.dataset
    y = np.asarray(getattr(ds, "y", None))
    if y is None or y.size == 0:
        return None
    idx = getattr(ds, "indices", None)
    if idx is not None:
        y = y[np.asarray(idx, dtype=np.int64)]
    pos = float((y > 0.5).sum())
    neg = float((y <= 0.5).sum())
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 60,
    patience: int = 8,
    class_weight: str | None = "balanced",
    amp: bool = True,
    seed: int = 0,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> TrainResult:
    """val AUROC 기준 조기종료. 종료 시 모델은 **best state**로 복원된다."""
    set_seed(seed)
    device = pick_device(device if isinstance(device, str) else None) if not isinstance(
        device, torch.device
    ) else device
    model.to(device)

    pw = _pos_weight(train_loader, class_weight)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device) if pw is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = bool(amp) and device.type == "cuda"  # AMP는 cuda일 때만
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    best_thr = 0.5
    history: list[dict] = []
    bad = 0

    for epoch in range(max_epochs):
        model.train()
        total, nb = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            nb += 1

        yv, pv = predict_probs(model, val_loader, device=device)
        va = auroc(yv, pv)
        if np.isnan(va):
            # val에 클래스가 하나뿐이면 AUROC가 정의되지 않는다. 조기종료 기준과
            # 임계값 선택이 통째로 무의미해지므로 조용히 넘어가지 않는다.
            raise ValueError(
                f"epoch {epoch}: val AUROC가 NaN이다 "
                f"(val n={len(yv)}, n_pos={int((yv > 0.5).sum())}). "
                "val split에 두 클래스가 모두 있어야 한다."
            )
        history.append({"epoch": epoch, "train_loss": total / max(nb, 1), "val_auroc": va})
        if verbose:
            print(f"[{epoch}] loss={total / max(nb, 1):.4f} val_auroc={va:.4f}")

        if va > best_auc:
            best_auc = va
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_thr = pick_threshold(yv, pv)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(
        best_epoch=best_epoch,
        best_val_auroc=float(best_auc),
        threshold=float(best_thr),
        epochs_run=len(history),
        history=history,
    )
