#!/usr/bin/env bash
# Release が無ければ作る（あれば何もしない）。GitHub の一時的なエラーで止まらないよう、待って試し直す。
#
#   bash scripts/gh_release_ensure.sh TAG [TITLE] [NOTES]
#
# 置き場のリポジトリは gh の流儀どおり GH_REPO（owner/name）で決まる。GH_TOKEN は呼び出し側の env で渡す。
# ティック（research/ticks.py）が年ごとの Release（data-tick-YYYY）を、初めてその年のファイルを
# 上げる前に作るのに使う（1つの Release に置けるアセットは 1,000 個まで。ティックは日ごとに年 約245ファイル）。
set -uo pipefail

tag="${1:-}"
if [ -z "$tag" ]; then
  echo "[ensure] 使い方: gh_release_ensure.sh TAG [TITLE] [NOTES]" >&2
  exit 2
fi
title="${2:-$tag}"
notes="${3:-$tag}"
tries="${GH_ENSURE_TRIES:-5}"
wait_s="${GH_ENSURE_WAIT:-5}"

for ((i = 1; i <= tries; i++)); do
  if gh release view "$tag" >/dev/null 2>&1; then
    exit 0
  fi
  if gh release create "$tag" --title "$title" --notes "$notes" --latest=false >/dev/null 2>&1; then
    echo "[ensure] Release $tag を作った"
    exit 0
  fi
  # 同時に作られた・一時的なエラー。待って、あるかどうかから見直す
  if [ "$i" -lt "$tries" ]; then
    s=$((wait_s * (1 << (i - 1))))
    echo "[ensure] $tag を確かめられなかった（${i}/${tries}回目）。${s}秒待つ"
    sleep "$s"
  fi
done
echo "[ensure] $tag を作れなかった" >&2
exit 1
