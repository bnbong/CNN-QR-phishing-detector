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


# --------------------------------------------------------------------------
# 2차 실험(review_01 "Grad-CAM은 보조 그림으로") 보강
# --------------------------------------------------------------------------

# kind 코드를 논문 표의 6개 영역으로 접는다. mode/length 헤더는 둘 다 URL 본문 밖의
# 결정적 필드라 하나로 묶는 편이 해석이 쉽다.
KIND_GROUPS = {
    "function": (0,),
    "char": (1,),
    "header": (2, 3),
    "pad": (4,),
    "ec": (5,),
    "remainder": (6,),
}


def signed_target(label: int) -> float:
    """phishing(1)이면 +1, benign(0)이면 -1.

    모든 샘플에서 phishing 로짓을 목표로 삼으면 benign CAM은 "benign의 근거"가 아니라
    "이 benign 샘플에서 phishing 로짓을 올리는 곳"이 된다(review_01 핵심 문제 3).
    """
    return 1.0 if int(label) == 1 else -1.0


def gradcam_signed(model, x: torch.Tensor, target_layer, sign: float = 1.0) -> np.ndarray:
    """부호 있는 Grad-CAM. ``sign``을 로짓에 곱해 역전파한다.

    ``sign=+1``은 기존 :func:`gradcam`과 동일하다(하위 호환).
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
        (float(sign) * logit).backward()
    finally:
        h1.remove()
        h2.remove()

    a = acts[0].detach()
    g = grads[0].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam_t = F.relu((w * a).sum(dim=1, keepdim=True))
    cam_t = F.interpolate(cam_t, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam_t[0, 0].cpu().numpy().astype(np.float64)
    mx = float(cam.max())
    return cam / mx if mx > 0 else cam


def integrated_gradients(
    model, x: torch.Tensor, baseline: float | torch.Tensor = 0.0, steps: int = 32,
    sign: float = 1.0,
) -> np.ndarray:
    """signed Integrated Gradients. ``(2, n, n)`` float64 기여도.

    완결성(completeness): ``attr.sum() ≈ sign * (f(x) - f(baseline))``. 리만 중점합으로
    근사하므로 ``steps``가 커질수록 오차가 줄어든다.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device).double()
    base = (
        torch.full_like(x, float(baseline))
        if not isinstance(baseline, torch.Tensor)
        else baseline.to(device).double().reshape(x.shape)
    )
    model.eval().double()
    try:
        diff = x - base
        total = torch.zeros_like(x)
        # 중점법: alpha = (k + 0.5) / steps
        for k in range(steps):
            alpha = (k + 0.5) / steps
            xi = (base + alpha * diff).requires_grad_(True)
            model.zero_grad(set_to_none=True)
            out = float(sign) * model(xi).reshape(-1)[0]
            (grad,) = torch.autograd.grad(out, xi)
            total = total + grad.detach()
        attr = (diff * total / steps)[0]
    finally:
        model.float()
    return attr.cpu().numpy().astype(np.float64)


def ig_logit_gap(model, x: torch.Tensor, baseline: float = 0.0, sign: float = 1.0) -> float:
    """``sign * (f(x) - f(baseline))``. IG 완결성 검증용."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device).double()
    model.eval().double()
    try:
        with torch.no_grad():
            fx = float(model(x).reshape(-1)[0])
            fb = float(model(torch.full_like(x, float(baseline))).reshape(-1)[0])
    finally:
        model.float()
    return float(sign) * (fx - fb)


def area_fraction_by_kind(prov: ModuleProvenance) -> dict[str, float]:
    """kind 그룹별 **면적 분수**. 모든 그룹 합 = 1."""
    kind = np.asarray(prov.kind)
    total = float(kind.size)
    return {
        name: float(np.isin(kind, codes).sum() / total) if total else 0.0
        for name, codes in KIND_GROUPS.items()
    }


def mass_fraction_by_kind(cam: np.ndarray, prov: ModuleProvenance) -> dict[str, float]:
    """kind 그룹별 attribution **질량 분수**. 모든 그룹 합 = 1(질량>0일 때)."""
    cam = np.abs(np.asarray(cam, dtype=np.float64))
    kind = np.asarray(prov.kind)
    total = float(cam.sum())
    out = {name: 0.0 for name in KIND_GROUPS}
    if total <= 0:
        return out
    for name, codes in KIND_GROUPS.items():
        m = np.isin(kind, codes)
        if m.any():
            out[name] = float(cam[m].sum() / total)
    return out


def enrichment_by_kind(cam: np.ndarray, prov: ModuleProvenance) -> dict[str, float]:
    """``질량 분수 / 면적 분수`` (review_01 핵심 문제 4).

    균일 attribution이면 모든 kind에서 정확히 1이 된다. 1보다 크면 그 영역이 면적
    대비 과대표(over-represented)됐다는 뜻이다. 면적이 0인 kind는 ``nan``.
    """
    mass = mass_fraction_by_kind(cam, prov)
    area = area_fraction_by_kind(prov)
    return {
        k: (mass[k] / area[k] if area[k] > 0 else float("nan")) for k in KIND_GROUPS
    }


def random_attribution_enrichment(
    shape: tuple[int, int], prov: ModuleProvenance, seed: int = 0, n_rep: int = 20
) -> dict[str, float]:
    """균일 난수 attribution 맵의 enrichment 평균(기준선). 기대값은 1이다."""
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {k: [] for k in KIND_GROUPS}
    for _ in range(n_rep):
        e = enrichment_by_kind(rng.random(shape), prov)
        for k, v in e.items():
            acc[k].append(v)
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in acc.items()}


def randomized_model_cam(model, x: torch.Tensor, target_layer_name: str = "block3",
                         sign: float = 1.0, seed: int = 0) -> np.ndarray:
    """파라미터를 무작위 재초기화한 사본으로 뽑은 CAM (sanity check).

    원 CAM과 상관이 높으면 CAM이 학습된 가중치가 아니라 구조·입력만 반영한다는
    뜻이다(Adebayo et al. 2018 sanity check).
    """
    import copy as _copy

    from torch import nn

    rnd = _copy.deepcopy(model)
    gen = torch.Generator().manual_seed(int(seed))
    for m in rnd.modules():
        if isinstance(m, nn.Conv2d | nn.Linear):
            with torch.no_grad():
                w = torch.empty_like(m.weight, device="cpu")
                nn.init.kaiming_uniform_(w, a=5**0.5, generator=gen)
                m.weight.copy_(w.to(m.weight.device))
                if m.bias is not None:
                    m.bias.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.reset_running_stats()
    rnd.eval()
    return gradcam_signed(rnd, x, getattr(rnd, target_layer_name), sign=sign)


def cam_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """두 CAM의 피어슨 상관. 한쪽이 상수면 0."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def balanced_sample(y: np.ndarray, indices: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """``indices`` 안에서 클래스별 균형 무작위 표본. 앞쪽 N개 고정 방식을 대체한다."""
    y = np.asarray(y)
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    for c in (0, 1):
        pool = indices[y[indices] == c]
        if pool.size == 0:
            continue
        k = min(int(per_class), int(pool.size))
        picked.append(rng.choice(pool, size=k, replace=False))
    if not picked:
        return np.array([], dtype=np.int64)
    out = np.concatenate(picked)
    return np.sort(out)


__all__ += [
    "KIND_GROUPS",
    "signed_target",
    "gradcam_signed",
    "integrated_gradients",
    "ig_logit_gap",
    "area_fraction_by_kind",
    "mass_fraction_by_kind",
    "enrichment_by_kind",
    "random_attribution_enrichment",
    "randomized_model_cam",
    "cam_correlation",
    "balanced_sample",
]
