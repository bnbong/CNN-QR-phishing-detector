"""probes — 임베딩 추출, 목표 생성, 셔플 기준선, 프로브 유의성(약한 검정)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from qrphish.models import SmallCNN
from qrphish.probes import (
    KEYWORDS,
    PROBE_N_BOOT,
    build_targets,
    embed,
    fit_binary_probe,
    group_shuffle,
    r2_score_fn,
    run_probe_suite,
    standardize,
    top_char_ngrams,
)
from tests.test_explain import Setup, build_trained


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> Setup:
    return build_trained(tmp_path_factory.mktemp("probes"), n_per_class=40, epochs=2)


# ------------------------------------------------------------------- 임베딩
def test_embed_shape_and_matches_head_input(setup: Setup) -> None:
    """임베딩은 GAP 출력이므로 ``head``에 넣으면 forward 로짓과 같아야 한다."""
    z = embed(setup.model, setup.ds, np.arange(8))
    assert z.shape == (8, 128)
    with torch.no_grad():
        xs = torch.stack([setup.ds[i][0] for i in range(8)])
        logit = setup.model(xs).numpy()
        from_embed = (
            setup.model.head(torch.from_numpy(z).float()).squeeze(-1).numpy()
        )
    assert np.allclose(logit, from_embed, atol=1e-4)


def test_embed_rejects_non_conv() -> None:
    class Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Linear(1, 1)

    with pytest.raises(TypeError):
        embed(Dummy(), [], np.arange(0))


# --------------------------------------------------------------------- 목표
def test_top_char_ngrams_is_label_independent(setup: Setup) -> None:
    grams = top_char_ngrams(setup.urls, size=3, top=20)
    assert len(grams) == 20
    assert all(len(g) == 3 for g in grams)
    assert "htt" in grams
    # 결정적: 같은 입력이면 같은 목록
    assert grams == top_char_ngrams(setup.urls, size=3, top=20)


def test_build_targets_covers_spec(setup: Setup) -> None:
    grams = top_char_ngrams(setup.urls, size=3, top=5)
    targets = build_targets(setup.urls, setup.y, grams)
    names = {t.name for t in targets}
    for kw in KEYWORDS:
        assert f"keyword:{kw}" in names
    for cont in ("digit_ratio", "special_ratio", "dot_count", "path_depth", "subdomain_count"):
        assert cont in names
        t = next(t for t in targets if t.name == cont)
        assert t.kind == "continuous"
    for b in ("has_path", "has_query", "has_digit", "has_hyphen", "ip_host", "label:phishing"):
        assert b in names
    # 합성 데이터 설계상 phishing만 숫자를 가진다.
    hd = next(t for t in targets if t.name == "has_digit")
    assert np.array_equal(hd.values, setup.y)


def test_ip_host_target() -> None:
    urls = ["http://192.168.10.24/wp-admin/login.php", "http://example.com/a"]
    t = {x.name: x for x in build_targets(urls, np.array([1, 0]), [])}
    assert list(t["ip_host"].values) == [1, 0]
    assert list(t["has_query"].values) == [0, 0]
    assert list(t["path_depth"].values) == [2, 1]


# ----------------------------------------------------------------- 기준선/지표
def test_group_shuffle_preserves_length_and_breaks_link() -> None:
    groups = np.repeat(np.arange(10), 5)
    values = np.repeat(np.arange(10), 5)  # 그룹당 상수 목표
    out = group_shuffle(values, groups, seed=0)
    assert out.shape == values.shape
    # 그룹째 갈아끼우므로 그룹 안은 여전히 상수여야 한다.
    for g in range(10):
        assert len(set(out[groups == g])) == 1
    assert not np.array_equal(out, values)


def test_r2_score_fn() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pytest.approx(1.0) == r2_score_fn(y, y)
    assert r2_score_fn(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert np.isnan(r2_score_fn(np.ones(4), np.ones(4)))


def test_fit_binary_probe_constant_target() -> None:
    z = np.random.default_rng(0).normal(size=(20, 4))
    pred = fit_binary_probe(z[:10], np.zeros(10), z[10:])
    assert pred.shape == (10,)
    assert np.allclose(pred, 0.0)


# ------------------------------------------------------------------- 통합(약한 검정)
def test_probe_suite_digit_target_beats_shuffle(setup: Setup) -> None:
    """학습된 CNN 임베딩에서 '숫자 포함'이 셔플 기준선보다 잘 예측돼야 한다.

    합성 데이터에서 숫자 포함 = phishing 라벨이라 학습된 임베딩이 이 목표를 선형으로
    담고 있어야 한다. 소표본이므로 유의성이 아니라 **점추정 우위**만 요구하는
    약한 검정이다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    grams = top_char_ngrams(setup.urls[tr], size=3, top=3)
    targets = [
        t for t in build_targets(setup.urls, setup.y, grams)
        if t.name in ("has_digit", "digit_ratio", "label:phishing")
    ]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    torch.manual_seed(123)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)

    res = run_probe_suite(
        z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
        seed=0, n_boot=100, min_positive=5,
    )
    hd = res["has_digit"]
    assert "skipped" not in hd
    assert hd["kind"] == "binary"
    assert hd["score"] > hd["shuffle"]
    assert set(hd) >= {
        "score", "ci", "shuffle", "shuffle_ci", "random_init",
        "accessible", "learned_gain", "accessible_and_gained", "used_in_decision",
        "significant",
    }
    assert res["digit_ratio"]["kind"] == "continuous"


def test_probe_suite_skips_degenerate_targets(setup: Setup) -> None:
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    targets = [t for t in build_targets(setup.urls, setup.y, []) if t.name == "ip_host"]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    res = run_probe_suite(z_tr, z_te, z_tr, z_te, targets, tr, te, setup.groups, n_boot=10)
    # 합성 URL에 IP 호스트가 없으므로 건너뛰어야 한다.
    assert "skipped" in res["ip_host"]


def test_probe_significance_requires_both_baselines(setup: Setup) -> None:
    """유의 = CI 하한 > max(셔플 CI 상한, 무작위 초기화 CI 상한) (코덱스 리뷰 3).

    학습된 임베딩 자리에 **무작위 초기화 임베딩을 그대로** 넣으면 두 쪽이 같으므로
    어떤 목표도 무작위 초기화 기준선을 넘을 수 없고, 따라서 유의로 서면 안 된다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    targets = [t for t in build_targets(setup.urls, setup.y, []) if t.name == "has_digit"]
    torch.manual_seed(321)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)

    res = run_probe_suite(
        zr_tr, zr_te, zr_tr, zr_te, targets, tr, te, setup.groups,
        seed=0, n_boot=100, min_positive=5,
    )
    row = res["has_digit"]
    assert "random_init_ci" in row
    assert row["significant"] is False
    assert row["above_random_init"] is False


def test_probe_ngram_universe_is_fixed_across_seeds(setup: Setup) -> None:
    """목표 유니버스는 시드와 무관하게 같아야 한다 (코덱스 리뷰 3).

    ``top_char_ngrams``는 라벨을 보지 않고 빈도·사전순으로만 정렬하므로, 같은 URL
    집합에서는 몇 번을 불러도 같은 목록을 낸다. 러너는 이 목록을 층에서 한 번만 만들어
    모든 시드에 넘긴다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    a = top_char_ngrams(setup.urls[tr], size=3, top=10)
    b = top_char_ngrams(setup.urls[tr], size=3, top=10)
    assert a == b and len(a) == len(set(a))


# ------------------------------------------------------- 속도 최적화 경로(의미 보존)
def test_standardize_matches_pipeline_scaler(setup: Setup) -> None:
    """시드당 한 번 적합한 스케일러가 파이프라인 안 StandardScaler와 같은 값을 낸다."""
    from sklearn.preprocessing import StandardScaler

    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    a_tr, a_te = standardize(z_tr, z_te)
    sc = StandardScaler().fit(z_tr)
    assert np.allclose(a_tr, sc.transform(z_tr))
    assert np.allclose(a_te, sc.transform(z_te))


def test_fit_binary_probe_scale_flag_equivalent(setup: Setup) -> None:
    """미리 표준화한 뒤 ``scale=False``로 적합해도 파이프라인과 같은 결정값(순위)이 나온다."""
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    t = setup.y[tr]
    a = fit_binary_probe(z_tr, t, z_te)
    s_tr, s_te = standardize(z_tr, z_te)
    b = fit_binary_probe(s_tr, t, s_te, scale=False)
    assert np.allclose(a, b, atol=1e-8)


def test_probe_n_boot_default_is_200() -> None:
    """부트스트랩 기본 반복 수는 200이다(결과 JSON에도 이 값이 기록된다)."""
    assert PROBE_N_BOOT == 200


def test_probe_suite_parallel_matches_serial(setup: Setup) -> None:
    """joblib 병렬 경로는 직렬 경로와 **같은 결과**를 낸다(결정성).

    시드는 목표 인덱스에서 파생하므로 실행 순서·워커 수와 무관하다.
    """
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    grams = top_char_ngrams(setup.urls[tr], size=3, top=6)
    targets = build_targets(setup.urls, setup.y, grams)
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    torch.manual_seed(7)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)

    kw: dict[str, Any] = dict(seed=0, n_boot=40, min_positive=5)
    a = run_probe_suite(z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
                        n_jobs=1, **kw)
    b = run_probe_suite(z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
                        n_jobs=2, **kw)
    assert set(a) == set(b)
    for name, row in a.items():
        assert row == b[name]


def test_probe_significance_rule_is_unchanged(setup: Setup) -> None:
    """세 질문의 판정을 저장된 CI로 재검산한다 (review_03 6절).

    ``accessible``은 셔플 기준선만, ``learned_gain``은 무작위 초기화 기준선만 본다.
    옛 ``significant``는 둘의 AND이며 값이 그대로 유지된다(별칭).
    """
    tr = np.nonzero(setup.split == 0)[0]
    te = np.nonzero(setup.split == 2)[0]
    targets = build_targets(setup.urls, setup.y, top_char_ngrams(setup.urls[tr], size=3, top=4))
    z_tr, z_te = embed(setup.model, setup.ds, tr), embed(setup.model, setup.ds, te)
    torch.manual_seed(11)
    rnd = SmallCNN(in_ch=2).eval()
    zr_tr, zr_te = embed(rnd, setup.ds, tr), embed(rnd, setup.ds, te)
    res = run_probe_suite(z_tr, z_te, zr_tr, zr_te, targets, tr, te, setup.groups,
                          seed=0, n_boot=40, min_positive=5, n_jobs=1)
    checked = 0
    for row in res.values():
        if "skipped" in row:
            continue
        bars = [v for v in (row["shuffle_ci"][1], row["random_init_ci"][1]) if np.isfinite(v)]
        bar = max(bars) if bars else float("nan")
        expect = bool(np.isfinite(row["ci"][0]) and np.isfinite(bar) and row["ci"][0] > bar)
        acc = bool(
            np.isfinite(row["ci"][0])
            and np.isfinite(row["shuffle_ci"][1])
            and row["ci"][0] > row["shuffle_ci"][1]
        )
        gain = bool(
            np.isfinite(row["random_init_ci"][1]) and row["ci"][0] > row["random_init_ci"][1]
        )
        assert row["accessible"] is acc
        assert row["learned_gain"] is gain
        assert row["accessible_and_gained"] is (acc and gain)
        # 옛 AND 규칙과 값이 같아야 한다(이름만 바꿨다).
        assert row["accessible_and_gained"] is expect
        assert row["significant"] is expect
        assert row["above_random_init"] is gain
        # 세 번째 질문은 현재 실험으로 답할 수 없다 — 항상 null.
        assert row["used_in_decision"] is None
        checked += 1
    assert checked > 0


def test_accessible_and_learned_gain_can_disagree() -> None:
    """"접근 가능"과 "학습 이득"은 다른 질문이다 — 한쪽만 참인 경우가 있어야 한다.

    review_03의 예(ngram:jp. 학습 0.909 vs 무작위 초기화 0.849)처럼, 셔플은 크게 넘지만
    무작위 초기화는 넘지 못하는 목표가 실제로 존재한다. 집계 함수가 그 구분을 유지하는지
    본다.
    """
    from qrphish.runner import _probe_aggregate, _probe_summary

    def row(lo: float, shuf_hi: float, rnd_hi: float) -> dict:
        return {
            "kind": "binary", "family": "ngram", "score": lo + 0.05,
            "ci": [lo, lo + 0.1], "shuffle": 0.5, "shuffle_ci": [0.4, shuf_hi],
            "random_init": rnd_hi - 0.05, "random_init_ci": [rnd_hi - 0.1, rnd_hi],
            "accessible": lo > shuf_hi, "learned_gain": lo > rnd_hi,
            "accessible_and_gained": lo > shuf_hi and lo > rnd_hi,
            "above_random_init": lo > rnd_hi,
            "significant": lo > shuf_hi and lo > rnd_hi,
            "used_in_decision": None,
        }

    per_seed = [
        {"targets": {"ngram:jp.": row(0.86, 0.59, 0.91), "ngram:xx": row(0.86, 0.59, 0.70)}}
        for _ in range(2)
    ]
    agg = _probe_aggregate(per_seed, n_seeds=2)
    assert agg["ngram:jp."]["accessible"] is True
    assert agg["ngram:jp."]["learned_gain"] is False
    assert agg["ngram:jp."]["accessible_and_gained"] is False
    assert agg["ngram:xx"]["accessible_and_gained"] is True
    assert agg["ngram:jp."]["used_in_decision"] is None

    summ = _probe_summary(agg)
    assert summ["accessible"]["n"] == 2
    assert summ["learned_gain"]["n"] == 1
    assert summ["accessible_and_gained"]["n"] == 1
    assert [t["target"] for t in summ["accessible_only"]["top10"]] == ["ngram:jp."]
    assert summ["used_in_decision"] is None
    # 하위 호환 별칭이 옛 키를 그대로 채운다.
    assert summ["n_significant"] == 1


def test_probe_aggregate_recovers_accessible_from_old_rows() -> None:
    """``accessible`` 필드가 없는 옛 시드 행도 저장된 CI로 정확히 복원된다."""
    from qrphish.runner import _probe_aggregate

    old = {
        "kind": "binary", "family": "ngram", "score": 0.9,
        "ci": [0.86, 0.95], "shuffle": 0.5, "shuffle_ci": [0.4, 0.59],
        "random_init": 0.85, "random_init_ci": [0.78, 0.91],
        "above_random_init": False, "significant": False,
    }
    agg = _probe_aggregate([{"targets": {"ngram:jp.": old}}], n_seeds=1)
    assert agg["ngram:jp."]["accessible"] is True
    assert agg["ngram:jp."]["learned_gain"] is False


def test_ngram_jp_case_is_accessible_but_no_learned_gain() -> None:
    """review_04의 ``ngram:jp.`` 사례가 "접근 가능하나 학습 이득 없음"으로 분류된다.

    저장값은 학습 0.909(CI 하한 0.865), 셔플 상한 0.587, 미학습 상한 0.908이다. CI 하한이
    셔플 상한을 크게 넘으므로 ① accessible은 참이고, 미학습 상한은 넘지 못하므로 ②
    learned_gain은 거짓이다. 옛 표는 이 목표의 ①을 ③의 답으로 대신 채워 "접근성 통과
    시드 0개"로 보고했다 — 두 판정은 다른 질문의 답이다.
    """
    from qrphish.runner import _probe_aggregate, _probe_summary

    seed_row = {
        "kind": "binary",
        "family": "ngram",
        "score": 0.909,
        "ci": [0.865, 0.947],
        "shuffle": 0.497,
        "shuffle_ci": [0.418, 0.587],
        "random_init": 0.849,
        "random_init_ci": [0.780, 0.908],
        "accessible": True,
        "learned_gain": False,
        "accessible_and_gained": False,
        "above_random_init": False,
        "significant": False,
        "used_in_decision": None,
    }
    per_seed = [{"seed": k, "targets": {"ngram:jp.": dict(seed_row)}} for k in range(5)]
    agg = _probe_aggregate(per_seed, n_seeds=5)
    row = agg["ngram:jp."]
    assert row["accessible"] is True
    assert row["learned_gain"] is False
    assert row["accessible_and_gained"] is False
    assert row["n_seeds_accessible"] == 5
    assert row["n_seeds_learned_gain"] == 0
    # 시드별 원값이 결과에 남아야 나중에 정확 재집계가 가능하다.
    assert [r["seed"] for r in row["per_seed"]] == [0, 1, 2, 3, 4]
    assert row["per_seed"][0]["shuffle_ci"] == [0.418, 0.587]
    assert row["per_seed"][0]["accessible"] is True

    summ = _probe_summary(agg)
    assert summ["accessible"]["n"] == 1
    assert summ["learned_gain"]["n"] == 0
    assert summ["accessible_and_gained"]["n"] == 0
    assert summ["accessible_only"]["n"] == 1


def _stored_results(with_per_seed: bool) -> dict:
    """층 results.json 한 벌. ``with_per_seed``면 새 스키마(목표별 시드 CI 포함)."""
    jp = {
        "kind": "binary",
        "metric": "auroc",
        "family": "ngram",
        "score": 0.909,
        "ci": [0.865, 0.947],
        "shuffle": 0.497,
        "shuffle_ci": [0.418, 0.587],
        "random_init": 0.849,
        "random_init_ci": [0.780, 0.908],
        "significant": False,
        "above_random_init": False,
    }
    label = {
        "kind": "binary",
        "metric": "auroc",
        "family": "label",
        "score": 0.99,
        "ci": [0.98, 1.0],
        "shuffle": 0.5,
        "shuffle_ci": [0.4, 0.6],
        "random_init": 0.6,
        "random_init_ci": [0.5, 0.7],
        "significant": True,
        "above_random_init": True,
    }
    if with_per_seed:
        for row in (jp, label):
            row["per_seed"] = [
                {
                    "seed": k,
                    "score": row["score"],
                    "ci": list(row["ci"]),
                    "shuffle": row["shuffle"],
                    "shuffle_ci": list(row["shuffle_ci"]),
                    "random_init": row["random_init"],
                    "random_init_ci": list(row["random_init_ci"]),
                    "accessible": row["ci"][0] > row["shuffle_ci"][1],
                    "learned_gain": row["above_random_init"],
                }
                for k in range(2)
            ]
    return {"stratum": "v2", "seeds": [0, 1], "targets": {"ngram:jp.": jp, "label:phishing": label}}


def _write_probe_results(tmp_path, payload: dict):
    import json

    from qrphish.runner import MAIN_CONDITION

    d = tmp_path / "reports" / "probes" / MAIN_CONDITION / "v2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return d


def test_recompute_probe_summaries_marks_accessible_unaggregated(tmp_path) -> None:
    """시드별 CI가 없으면 ``accessible``을 **근사하지 않고** 미집계로 남긴다.

    옛 구현은 시드 평균 CI로 판정해 근사값을 냈다. 평균 CI는 "전 시드에서 성립"을
    담지 못하는데도 그 수가 정확한 집계처럼 보고돼 결론을 틀리게 만들었다(review_04).
    """
    import json

    from qrphish.config import from_dict
    from qrphish.runner import recompute_probe_summaries

    d = _write_probe_results(tmp_path, _stored_results(with_per_seed=False))
    cfg = from_dict({"reports_dir": str(tmp_path / "reports")})
    res = recompute_probe_summaries(cfg)

    assert res["strata"]["v2"]["accessible"] is None
    assert res["strata"]["v2"]["learned_gain"]["n"] == 0
    assert res["needs_rerun"] == ["v2"]
    assert res["missing_fields"]
    written = json.loads((d / "results.json").read_text(encoding="utf-8"))
    assert written["targets"]["ngram:jp."]["accessible"] is None
    assert written["targets"]["ngram:jp."]["accessible_basis"] == "unrecomputable:no_per_seed_ci"
    assert written["summary"]["accessible"] is None
    assert written["summary"]["accessible_only"] is None
    assert written["summary"]["needs_rerun"]
    # 미집계일 때는 옛 별칭 키도 만들지 않는다(표가 그 값을 대신 채우지 못하게).
    assert written["summary"]["used_in_decision"] is None
    assert written["summary"]["n_targets"] == 1
    assert "approx" not in json.dumps(written["summary"], ensure_ascii=False)


def test_recompute_probe_summaries_exact_from_per_seed(tmp_path) -> None:
    """``targets[*].per_seed``가 있으면 전 시드 판정으로 정확히 재집계한다."""
    import json

    from qrphish.config import from_dict
    from qrphish.runner import recompute_probe_summaries

    d = _write_probe_results(tmp_path, _stored_results(with_per_seed=True))
    cfg = from_dict({"reports_dir": str(tmp_path / "reports")})
    res = recompute_probe_summaries(cfg)

    assert res["needs_rerun"] == []
    assert res["strata"]["v2"]["accessible"]["n"] == 1
    assert res["strata"]["v2"]["learned_gain"]["n"] == 0
    assert res["strata"]["v2"]["accessible_basis"] == "exact:results_per_seed"
    written = json.loads((d / "results.json").read_text(encoding="utf-8"))
    assert written["targets"]["ngram:jp."]["accessible"] is True
    assert written["summary"]["accessible_only"]["n"] == 1


def test_recompute_probe_summaries_exact_from_seed_cache(tmp_path) -> None:
    """``seed{k}.json`` 캐시가 남아 있으면 그걸로 정확 재집계한다."""
    import json

    from qrphish.config import from_dict
    from qrphish.runner import recompute_probe_summaries

    d = _write_probe_results(tmp_path, _stored_results(with_per_seed=False))
    seed_row = {
        "kind": "binary",
        "family": "ngram",
        "score": 0.909,
        "ci": [0.865, 0.947],
        "shuffle": 0.497,
        "shuffle_ci": [0.418, 0.587],
        "random_init": 0.849,
        "random_init_ci": [0.780, 0.908],
        "accessible": True,
        "learned_gain": False,
        "accessible_and_gained": False,
        "above_random_init": False,
        "significant": False,
    }
    for k in range(2):
        (d / f"seed{k}.json").write_text(
            json.dumps({"seed": k, "targets": {"ngram:jp.": seed_row}}, ensure_ascii=False),
            encoding="utf-8",
        )
    cfg = from_dict({"reports_dir": str(tmp_path / "reports")})
    res = recompute_probe_summaries(cfg)

    assert res["needs_rerun"] == []
    assert res["strata"]["v2"]["accessible_basis"] == "exact:seed_cache"
    assert res["strata"]["v2"]["accessible"]["n"] == 1
    assert res["strata"]["v2"]["accessible_and_gained"]["n"] == 0


def test_table_probes_never_substitutes_n_significant(tmp_path, monkeypatch) -> None:
    """표 생성기가 미집계 칸에 옛 ``n_significant``를 대신 넣지 않는다.

    review_04가 지적한 "접근성 2/91"은 이 폴백의 산물이었다.
    """
    import json

    from qrphish.runner import MAIN_CONDITION

    mod = _load_tables_module()
    d = tmp_path / "probes" / MAIN_CONDITION / "v2"
    d.mkdir(parents=True)
    (d / "results.json").write_text(
        json.dumps(
            {
                "stratum": "v2",
                "summary": {
                    "n_targets": 91,
                    "accessible": None,
                    "learned_gain": {"n": 2, "frac": 2 / 91, "targets": [], "top10": []},
                    "accessible_and_gained": {
                        "n": 2,
                        "frac": 2 / 91,
                        "targets": [],
                        "top10": [],
                    },
                    "accessible_only": None,
                    "n_significant": 2,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "REPORTS", tmp_path)
    out = mod.table_probes()
    line = next(ln for ln in out.splitlines() if "`accessible`" in ln and "|" in ln)
    assert "2 / 91" not in line
    assert mod.UNAGGREGATED in line
    assert "미집계" in out
    assert "폴백" in out  # 각주가 옛 폴백의 산물임을 밝힌다


def _load_tables_module():
    """``scripts/make_results_tables.py``를 모듈로 불러온다(패키지가 아니다)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_results_tables.py"
    spec = importlib.util.spec_from_file_location("make_results_tables", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
