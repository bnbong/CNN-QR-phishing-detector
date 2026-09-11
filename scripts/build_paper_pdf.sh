#!/usr/bin/env bash
# docs/PAPER_DRAFT.md 를 build/paper_draft.pdf 로 빌드한다.
#
# 툴체인: pandoc + tectonic(경량 XeTeX). 설치는 아래 한 줄로 끝난다.
#   brew install pandoc tectonic
#
# 원본 docs/PAPER_DRAFT.md 는 수정하지 않는다. 빌드에 필요한 전처리는
# build/ 안의 임시 복사본에만 적용한다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/docs/PAPER_DRAFT.md"
BUILD="$ROOT/build"
OUT="$BUILD/paper_draft.pdf"

# 한글 본문 폰트는 xeCJK 의 CJK 폰트로 지정한다.
# Apple SD Gothic Neo 에는 라틴 확장 글자(ń, ï 등)가 없어서 인용 문헌의
# 저자명이 빠진다. 그래서 라틴 문자는 별도 폰트가 맡고 한글만 이 폰트가 맡는다.
CJKFONT="${PAPER_CJKFONT:-Apple SD Gothic Neo}"
MAINFONT="${PAPER_MAINFONT:-Times New Roman}"
MONOFONT="${PAPER_MONOFONT:-Menlo}"

for bin in pandoc tectonic; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "error: $bin 이 없다. 'brew install pandoc tectonic' 으로 설치하라." >&2
    exit 1
  fi
done

[[ -f "$SRC" ]] || { echo "error: $SRC 가 없다." >&2; exit 1; }

mkdir -p "$BUILD"

# 임시 복사본. 원본 대신 이 파일에만 빌드용 전처리를 적용한다.
TMP_MD="$BUILD/paper_draft.build.md"
cp "$SRC" "$TMP_MD"

# 폭이 넓은 표(최대 7열, 셀당 수백 자)를 페이지 안에 넣기 위한 LaTeX 설정.
HEADER_TEX="$BUILD/paper_header.tex"
cat > "$HEADER_TEX" <<'TEX'
\usepackage{etoolbox}
\usepackage{array}
\usepackage{ragged2e}

% 넓은 표는 작은 글씨 + 좁은 열 간격 + 양끝맞춤 해제로 폭을 확보한다.
\AtBeginEnvironment{longtable}{%
  \scriptsize
  \setlength{\tabcolsep}{3pt}%
  \RaggedRight
  \renewcommand{\arraystretch}{1.08}%
}
% 셀 안 긴 URL·식별자가 열 폭을 넘지 않도록 임의 지점에서 줄바꿈을 허용한다.
\setlength{\emergencystretch}{3em}
\sloppy

% 그림은 항상 본문 흐름 순서대로 배치한다(뒤로 밀리는 float 방지).
\usepackage{float}
\floatplacement{figure}{H}

% 원문자 ①②③(U+2460~24FF)는 라틴 폰트에 없다. xeCJK 가 CJK 글자로 분류하게 해서
% 한글 폰트가 대신 그리도록 한다.
\makeatletter
\@ifpackageloaded{xeCJK}{%
  \xeCJKDeclareCharClass{CJK}{"2460 -> "24FF}%
}{}
\makeatother

% 캡션을 본문보다 작게.
\usepackage[font=small,labelfont=bf]{caption}
TEX

echo "[build] pandoc $(pandoc --version | head -1 | awk '{print $2}') + tectonic $(tectonic --version | awk '{print $2}')"
echo "[build] mainfont(latin): $MAINFONT / CJK: $CJKFONT"
echo "[build] $SRC -> $OUT"

# --shift-heading-level-by=-1 은 맨 앞 H1 을 문서 제목 메타데이터로 올린다.
# 본문 제목(## 1. 서론 등)이 이미 번호를 달고 있어 --number-sections 를 쓰면
# "1.4 3. 방법" 처럼 번호가 두 번 붙는다. 그래서 자동 번호는 끄고 원문 번호를 쓴다.
#
# --resource-path 에 docs/ 를 넣어야 본문의 ../reports/figures/F*.png 가 해석된다.
pandoc "$TMP_MD" \
  --from=markdown+pipe_tables+raw_tex \
  --pdf-engine=tectonic \
  --resource-path="$ROOT/docs:$ROOT:$ROOT/reports/figures" \
  --toc --toc-depth=3 \
  --shift-heading-level-by=-1 \
  --include-in-header="$HEADER_TEX" \
  -V documentclass=article \
  -V papersize=a4 \
  -V geometry:margin=2.5cm \
  -V fontsize=10pt \
  -V mainfont="$MAINFONT" \
  -V monofont="$MONOFONT" \
  -V monofontoptions="Scale=0.85" \
  -V CJKmainfont="$CJKFONT" \
  -V CJKsansfont="$CJKFONT" \
  -V CJKmonofont="$CJKFONT" \
  -V CJKoptions="BoldFont=AppleSDGothicNeo-Bold" \
  -V linestretch=1.15 \
  -V colorlinks=true \
  -V linkcolor=black \
  -V urlcolor=blue \
  -V toccolor=black \
  -V lang=ko \
  -o "$OUT"

echo "[build] done: $OUT ($(du -h "$OUT" | cut -f1))"
