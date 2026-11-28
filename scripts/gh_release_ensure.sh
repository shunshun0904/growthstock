#!/usr/bin/env bash
# Release が無ければ作る（あれば何もしない）。GitHub の一時的なエラーで止まらないよう、待って試し直す。
#
#   bash scripts/gh_release_ensure.sh TAG [TITLE] [NOTES]
#
# GH_ENSURE_DRAFT=1 のときは**下書き（draft）**で作り、既にあるものが下書きでなければ（公開されていれば）
# exit 3 で止める。分足・ティック（J-Quants のアドオンのデータ）は、このリポジトリの下書きの Release に置く。
# 下書きは書き込み権限のある人にしか見えない（2026-11-19 に Probe Draft Release で確かめた。docs/MINUTE_BARS.md）。
# 公開された Release は誰でもダウンロードでき、J-Quants の個人の利用は第三者への配信が禁止なので、
# 公開されていたら上げない。2026-11-28 から data-raw・data-jsf・data-volume-profile も下書き（docs/OPERATIONS.md
# 「data-raw を非公開にする」）。
#
# 作るのは、gh が「見つからない（not found）」と答えたときだけ。一時的なエラー（5xx・通信）で見えなかっただけの
# ときに作ると、下書きには git のタグが無いので、同じ名前の下書きが2つになりうる。そのときは待って、あるかどうかから
# 見直す。
#
# 置き場のリポジトリは gh の流儀どおり GH_REPO（owner/name）か、手元のリポジトリで決まる。GH_TOKEN は呼び出し側の env で渡す。
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
draft="${GH_ENSURE_DRAFT:-0}"
errf=$(mktemp)
trap 'rm -f "$errf"' EXIT

for ((i = 1; i <= tries; i++)); do
  if is_draft=$(gh release view "$tag" --json isDraft --jq .isDraft 2>"$errf"); then
    if [ "$draft" = "1" ] && [ "$is_draft" != "true" ]; then
      echo "::error title=Release が公開になっている::$tag は下書きでない（誰でもダウンロードできる）。J-Quants のデータを上げないよう止める。下書きに戻すか、消して作り直す（docs/MINUTE_BARS.md）" >&2
      exit 3
    fi
    exit 0
  fi
  if grep -qi "not found" "$errf"; then
    args=(--title "$title" --notes "$notes" --latest=false)
    [ "$draft" = "1" ] && args+=(--draft)
    if gh release create "$tag" "${args[@]}" >/dev/null 2>&1; then
      echo "[ensure] Release $tag を作った$([ "$draft" = "1" ] && echo "（下書き）")"
      exit 0
    fi
    # 同時に作られた・一時的なエラー。待って、あるかどうかから見直す
    echo "[ensure] $tag を作れなかった（${i}/${tries}回目）"
  else
    # 「無い」とは言われていない。作らずに、待って見直す
    echo "[ensure] $tag があるか確かめられなかった（${i}/${tries}回目）: $(head -c 200 "$errf" | tr '\n' ' ')"
  fi
  if [ "$i" -lt "$tries" ]; then
    s=$((wait_s * (1 << (i - 1))))
    echo "[ensure] ${s}秒待つ"
    sleep "$s"
  fi
done
echo "[ensure] $tag があるか確かめられず、作れもしなかった" >&2
exit 1
