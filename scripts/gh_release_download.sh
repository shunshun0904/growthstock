#!/usr/bin/env bash
# Release からの取り出しを、GitHub の一時的なエラーで「一部だけ」にならないようにする。
#
# なぜ要るか
# --------
# 取り込みのワークフローは `gh release download ... || echo "保存済みデータなし（初回）"` と
# 書いていた。2026-10-01 に GitHub が1アセットだけ HTTP 500 を返し（run 36882172803、
# mjrshld_2020.parquet）、gh は失敗を返したが、手順は「初回」とみなして続いた。
# 手元に無いファイルが今年の分（例: bars_2026.parquet）だったら、取り込みはその年を
# 今夜の行だけで作り直し、Release の1年分を上書きする。黙って過去データが消える。
# 日証金（jsf_hist.parquet）や EDINET（edinet_fin.parquet）では、積み上げた履歴が消える。
#
# 何をするか
# --------
# 1. Release のアセット一覧（名前と大きさ）を引く。Release が無い・合うアセットが無いなら
#    初回として exit 0
# 2. 取り出して、全ファイルが**同じ大きさで**手元にあるかを確かめる。足りなければ
#    足りないものだけ取り直す（5, 10, 20, 40 秒…待って）
# 3. GH_DOWNLOAD_TRIES（既定 5）回でそろわなければ exit 1（一部だけで先へ進ませない）
#
#   bash scripts/gh_release_download.sh TAG DIR [PATTERN...]
#
# PATTERN は gh の --pattern と同じグロブ（例: 'jsf_*'）。省略すると全アセット。
set -uo pipefail

tag="${1:-}"
dir="${2:-}"
if [ -z "$tag" ] || [ -z "$dir" ]; then
  echo "[download] 使い方: gh_release_download.sh TAG DIR [PATTERN...]" >&2
  exit 2
fi
shift 2
patterns=("$@")
tries="${GH_DOWNLOAD_TRIES:-5}"
wait_s="${GH_DOWNLOAD_WAIT:-5}"
mkdir -p "$dir"

matches() {
  local name="$1" p
  [ "${#patterns[@]}" -eq 0 ] && return 0
  for p in "${patterns[@]}"; do
    # shellcheck disable=SC2053
    [[ "$name" == $p ]] && return 0
  done
  return 1
}

# アセット一覧（名前<TAB>大きさ）。Release が無ければ "NOTFOUND" を返す
listing=""
for ((i = 1; i <= tries; i++)); do
  if out=$(gh release view "$tag" --json assets \
             --jq '.assets[] | select(.state == "uploaded") | "\(.name)\t\(.size)"' 2>&1); then
    listing="$out"
    break
  fi
  if grep -qi "not found" <<<"$out"; then
    echo "[download] Release（$tag）が無い。初回として進む"
    exit 0
  fi
  echo "[download] アセット一覧を引けなかった（${i}/${tries}回目）: $(head -c 200 <<<"$out")"
  if [ "$i" -eq "$tries" ]; then
    echo "[download] 一覧が引けないまま。一部だけで進まないよう止める" >&2
    exit 1
  fi
  sleep $((wait_s * (1 << (i - 1))))
done

declare -A want=()
while IFS=$'\t' read -r name size; do
  [ -z "$name" ] && continue
  matches "$name" && want["$name"]="$size"
done <<<"$listing"

if [ "${#want[@]}" -eq 0 ]; then
  echo "[download] Release（$tag）に合うアセットが無い。初回として進む"
  exit 0
fi

lacking() {
  local n f s
  for n in "${!want[@]}"; do
    f="$dir/$n"
    if [ ! -f "$f" ]; then printf '%s\n' "$n"; continue; fi
    s=$(stat -c %s "$f" 2>/dev/null || wc -c <"$f")
    [ "$s" = "${want[$n]}" ] || printf '%s\n' "$n"
  done
}

todo=("${!want[@]}")
for ((i = 1; i <= tries; i++)); do
  args=()
  for n in "${todo[@]}"; do args+=(--pattern "$n"); done
  gh release download "$tag" --dir "$dir" --clobber "${args[@]}" \
    || echo "[download] gh release download が失敗（${i}/${tries}回目）"
  mapfile -t todo < <(lacking | sort)
  if [ "${#todo[@]}" -eq 0 ]; then
    echo "[download] ${#want[@]}ファイルを取り出した（大きさも一致${i:+、${i}回目}）"
    exit 0
  fi
  echo "[download] 足りない・大きさが違う: ${#todo[@]}件（$(printf '%s ' "${todo[@]}" | cut -c1-200)）"
  if [ "$i" -lt "$tries" ]; then
    s=$((wait_s * (1 << (i - 1))))
    echo "[download] ${s}秒待って、足りないものだけ取り直す"
    sleep "$s"
  fi
done
echo "[download] ${tries}回やってもそろわなかった: ${todo[*]}" >&2
exit 1
