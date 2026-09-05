"""공용 픽스처: 작은 URL 표본과 임시 WebPhish CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

BENIGN_URLS = [
    "http://www.google.com/",
    "http://www.facebook.com",
    "https://www.wikipedia.org/wiki/QR_code",
    "http://naver.com",
    "http://www.amazon.com/gp/help",
    "https://github.com/psf/requests",
    "http://bbc.co.uk/news",
    "https://www.python.org/downloads/",
    "http://example.co.kr/a/b/c",
    "https://stackoverflow.com/questions/12345",
]

PHISHING_URLS = [
    "http://secure-login.000webhostapp.com/verify/account.php?id=1",
    "http://paypal-update.000webhostapp.com/confirm/login.html",
    "http://appleid-verify.weebly.com/signin",
    "http://bank-alert.duckdns.org/auth/session?tok=abc123",
    "http://free-gift.blogspot.com/claim/now",
    "http://192.168.10.24/wp-admin/login.php",
    "http://update-account.weebly.com/secure/step2",
    "http://login-verify.duckdns.org/portal/index.php",
    "http://mail-service.000webhostapp.com/inbox/reset",
    "http://drive-share.blogspot.com/doc/open?u=zz",
]


@pytest.fixture
def sample_urls() -> dict[str, list[str]]:
    return {"benign": list(BENIGN_URLS), "phishing": list(PHISHING_URLS)}


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """load_webphish 원본 형태(Category, Data)의 작은 표본."""
    rows = [{"Category": "ham", "Data": u} for u in BENIGN_URLS]
    rows += [{"Category": "spam", "Data": u} for u in PHISHING_URLS]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv(tmp_path: Path, sample_frame: pd.DataFrame) -> Path:
    path = tmp_path / "sample_webphish.csv"
    sample_frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def webphish_csv() -> Path:
    """실제 데이터셋 경로. 없으면 해당 테스트를 skip한다."""
    path = REPO_ROOT / "data" / "webphish.csv"
    if not path.exists():
        pytest.skip("data/webphish.csv 없음 (scripts/convert_webphish.py 먼저 실행)")
    return path
