#!/usr/bin/env bash
# DIRE's own repository distributes weights through Baidu and OneDrive links
# and was archived in July 2025. The SIDBench mirror on Hugging Face is
# public, ungated and scriptable, which is the property Section 5 of the
# paper argues benchmarks should have.
set -e
cd "$(dirname "$0")"
mkdir -p weights
BASE="https://huggingface.co/dkarageo/sidbench/resolve/main"

fetch () {
  url="$1"; out="$2"; want="$3"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then
    echo "  have $out"; return
  fi
  echo "  fetching $out ..."
  curl -fL --retry 3 -o "$out" "$url"
  got=$(stat -c%s "$out")
  if [ "$got" != "$want" ]; then
    echo "  SIZE MISMATCH for $out: got $got, expected $want" >&2
    exit 1
  fi
}

fetch "$BASE/dire/lsun_adm.pth"            weights/lsun_adm.pth      282534013
fetch "$BASE/preprocessing/lsun_bedroom.pt" weights/lsun_bedroom.pt  2211383297
echo "weights ready in $(pwd)/weights"
