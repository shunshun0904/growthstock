#!/usr/bin/env bash
# Release へのアップロードを、GitHub の一時的なエラーで穴が開かないようにする。
#
# なぜ要るか
# --------
# `gh release upload --clobber` は、同じ名前の古いアセットを**先に消してから**新しい版を上げる。
# 2026-10-01 に GitHub が「消す」要求へ HTTP 502 を返し（実際には消えていた）、gh はそこで
# 止まった。data-raw から topix_2018.parquet と mjrshld_2023.parquet が消えたまま残った
# （run 36875423002 / 36875867672）。取得記録（manifest.json）はその日を取得済みのままなので、
# あとの差分取得では戻らない。黙ってデータが欠ける。
#
# 何をするか
# --------
# 1. gh release upload --clobber を渡されたファイル全部で行う
# 2. 失敗したら待って（5, 10, 20, 40 秒…）、**全部を**上げ直す。どれが新しい版に
#    なったか分からないので、一部だけ上げ直すと古い版が残りうる。消えた後の上げ直しは
#    ただのアップロードになる
# 3. 成功したら Release のアセット一覧を引き、渡したファイルが全部あることを確かめる。
#    無ければ全部を上げ直す
#
#   bash scripts/gh_release_upload.sh TAG FILE...
#
# GH_UPLOAD_TRIES（既定 5）回でそろわなければ exit 1。GH_UPLOAD_WAIT は最初の待ち（秒）。
# GH_TOKEN は呼び出し側の env で渡す（gh がそのまま使う）。
set -uo pipefail

tag="${1:-}"
if [ -z "$tag" ]; then
  echo "[upload] 使い方: gh_release_upload.sh TAG FILE..." >&2
  exit 2
fi
shift
if [ "$#" -eq 0 ]; then
  echo "[upload] 上げるファイルが無い" >&2
  exit 2
fi
for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "[upload] ファイルが無い: $f" >&2
    exit 2
  fi
done

tries="${GH_UPLOAD_TRIES:-5}"
wait_s="${GH_UPLOAD_WAIT:-5}"

# 渡したファイルのうち、Release に無いもの（state が uploaded でないものも含む）を1行ずつ出す。
# 一覧が引けなければ 1 を返す
missing_of() {
  local names f
  names=$(gh release view "$tag" --json assets \
            --jq '.assets[] | select(.state == "uploaded") | .name') || return 1
  for f in "$@"; do
    grep -qxF -- "$(basename "$f")" <<<"$names" || printf '%s\n' "$(basename "$f")"
  done
  return 0
}

for ((i = 1; i <= tries; i++)); do
  if gh release upload "$tag" "$@" --clobber; then
    if miss=$(missing_of "$@"); then
      if [ -z "$miss" ]; then
        if [ "$i" -gt 1 ]; then
          echo "[upload] ${i}回目でそろった（$# ファイル）"
        fi
        exit 0
      fi
      echo "[upload] gh は成功を返したが Release に無い: $(echo $miss)（${i}/${tries}回目）"
    else
      # 上げるのは成功している。一覧が引けないのは確認の側の失敗なので、成功として返す
      echo "[upload] 上げた。アセット一覧は引けなかった（確認は省く）"
      exit 0
    fi
  else
    echo "[upload] gh release upload が失敗（${i}/${tries}回目）"
  fi
  if [ "$i" -lt "$tries" ]; then
    s=$((wait_s * (1 << (i - 1))))
    echo "[upload] ${s}秒待って全部を上げ直す"
    sleep "$s"
  fi
done

if miss=$(missing_of "$@"); then
  echo "[upload] ${tries}回やってもそろわなかった。Release に無いファイル: ${miss:-なし}" >&2
else
  echo "[upload] ${tries}回やってもそろわなかった（一覧も引けない）" >&2
fi
exit 1
