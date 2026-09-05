"""Grad-CAM과 모듈→비트→문자 기여도 역추적 (스펙 P3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from qrphish.mapping import ModuleProvenance

__all__ = [
    "KIND_NAMES",
    "gradcam",
    "attribute_to_chars",
    "attribute_by_kind",
    "byte_to_char_index",
    "mean_cam",
]

# mapping.ModuleProvenance.kind 코드 (스펙 4절)
KIND_NAMES = {
    0: "function",
    1: "char",
    2: "length_header",
    3: "mode_header",
    4: "pad",
    5: "ec",
    6: "remainder",
}


def gradcam(model, x: torch.Tensor, target_layer) -> np.ndarray:
    """(n,n) float CAM. 입력 해상도로 bilinear 업샘플, 0~1 정규화.

    x는 (1,2,n,n) 또는 (2,n,n). 이진 분류 로짓 하나를 목표로 삼는다.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)
    model.eval()

    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logit = model(x).reshape(-1)[0]
        logit.backward()
    finally:
        h1.remove()
        h2.remove()

    a = acts[0].detach()  # (1,C,h,w)
    g = grads[0].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam_t = F.relu((w * a).sum(dim=1, keepdim=True))
    cam_t = F.interpolate(cam_t, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam_t[0, 0].cpu().numpy().astype(np.float64)
    mx = float(cam.max())
    return cam / mx if mx > 0 else cam


def mean_cam(model, dataset, target_layer, indices=None, max_n: int = 256) -> np.ndarray:
    """샘플 평균 CAM(F6용)."""
    idx = np.arange(len(dataset)) if indices is None else np.asarray(indices)
    idx = idx[:max_n]
    acc = None
    for i in idx:
        x, _ = dataset[int(i)]
        c = gradcam(model, x, target_layer)
        acc = c if acc is None else acc + c
    return acc / max(len(idx), 1) if acc is not None else np.zeros((1, 1))


def byte_to_char_index(url: str) -> np.ndarray:
    """UTF-8 바이트 오프셋 → 파이썬 문자 인덱스 테이블. 길이는 URL의 바이트 수.

    ``ModuleProvenance.char_index``는 **바이트** 인덱스다(멀티바이트 문자는 여러
    바이트에 걸친다). 비ASCII URL에서 그대로 문자 인덱스로 쓰면 귀속이 어긋난다.
    """
    table: list[int] = []
    for i, ch in enumerate(url):
        table.extend([i] * len(ch.encode("utf-8")))
    return np.asarray(table, dtype=np.int64)


def attribute_to_chars(cam: np.ndarray, prov: ModuleProvenance, url: str) -> np.ndarray:
    """(len(url),) — 각 문자에 매핑된 모듈들의 CAM 합 / 모듈 수.

    해당 문자에 매핑된 모듈이 없으면 0. remainder/ec는 여기서 제외된다.
    멀티바이트 문자는 자기 바이트들에 걸린 모듈을 전부 모아 평균을 낸다.
    """
    cam = np.asarray(cam, dtype=np.float64)
    kind = np.asarray(prov.kind)
    ci = np.asarray(prov.char_index)
    n_chars = len(url)
    out = np.zeros(n_chars, dtype=np.float64)
    sel = kind == 1
    if not sel.any():
        return out
    b2c = byte_to_char_index(url)
    vals = cam[sel]
    byte_keys = ci[sel].astype(np.int64)
    ok = (byte_keys >= 0) & (byte_keys < b2c.size)
    vals, byte_keys = vals[ok], byte_keys[ok]
    if byte_keys.size == 0:
        return out
    keys = b2c[byte_keys]
    total = np.bincount(keys, weights=vals, minlength=n_chars)
    count = np.bincount(keys, minlength=n_chars)
    nz = count > 0
    out[nz] = total[nz] / count[nz]
    return out


def attribute_by_kind(cam: np.ndarray, prov: ModuleProvenance) -> dict[str, float]:
    """kind별 CAM 질량 비율(합=1). 잔여 비트는 태깅만 하고 해석에서는 제외한다(스펙 1.10)."""
    cam = np.asarray(cam, dtype=np.float64)
    kind = np.asarray(prov.kind)
    total = float(cam.sum())
    out = {name: 0.0 for name in KIND_NAMES.values()}
    if total <= 0:
        return out
    for code, name in KIND_NAMES.items():
        m = kind == code
        if m.any():
            out[name] = float(cam[m].sum() / total)
    return out
