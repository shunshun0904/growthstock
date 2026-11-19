#!/usr/bin/env python3
"""Publish the latest reconciled profiles to the private dashboard repository."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from build_volume_profile import PREDICTIONS, prediction_codes

DESTINATION = "dashboard/volume-profile.json"
MAX_BYTES = 75 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class PrivateStore:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository, re.ASCII):
            raise ValueError("非公開リポジトリ名を owner/repository で指定してください。")
        if not token:
            raise ValueError("Actions Secret MINUTE_DATA_TOKEN が必要です。")
        self.repository = repository
        self.url = f"https://api.github.com/repos/{repository}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                        "Content-Type": "application/json", "User-Agent": "growthstock-volume-profile"}
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, path, payload=None, missing_ok=False):
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {**self.headers, "Accept": "application/vnd.github.object+json"} if missing_ok else self.headers
        request = urllib.request.Request(self.url + path, body, headers, method="PUT" if body else "GET")
        try:
            with self.opener.open(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            raise ValueError(f"非公開リポジトリへの接続または保存に失敗しました（HTTP {exc.code}）。") from None
        except (OSError, json.JSONDecodeError):
            raise ValueError("非公開リポジトリへの通信を完了できませんでした。") from None

    def publish(self, data, day):
        repository = self.request("")
        if repository.get("private") is not True or repository.get("full_name", "").lower() != self.repository.lower():
            raise ValueError("指定した非公開リポジトリを確認できませんでした。")
        path = f"/contents/{DESTINATION}"
        current = self.request(path, missing_ok=True)
        payload = {"message": f"Update dashboard volume profiles for {day}", "branch": repository["default_branch"],
                   "content": base64.b64encode(data).decode("ascii")}
        if current is not None:
            payload["sha"] = current["sha"]
        self.request(path, payload)


def checked_snapshot(snapshot, predictions):
    if snapshot.stat().st_size > MAX_BYTES:
        raise ValueError("価格帯別出来高の集計が保存上限を超えています。")
    day, names = prediction_codes(predictions)
    data = snapshot.read_bytes()
    profile = json.loads(data)
    stocks = profile.get("stocks", [])
    if (profile.get("kind") != "growthstock.volume-profile" or profile.get("schemaVersion") != 1
            or profile.get("demo") is not False or profile.get("asOf") != day or len(stocks) != len(names)
            or {stock["code"] for stock in stocks} != names.keys() or any(stock["date"] != day for stock in stocks)):
        raise ValueError("集計データの基準日・銘柄が最新の予測と一致しません。")
    return data, day, len(stocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    args = parser.parse_args()
    try:
        data, day, count = checked_snapshot(args.snapshot, args.predictions)
        PrivateStore(os.environ.get("PROFILE_REPO", ""), os.environ.get("MINUTE_DATA_TOKEN", "")).publish(data, day)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"価格帯別出来高の保存に失敗しました。{exc}", file=sys.stderr)
        return 1
    print(f"{day}の予測対象{count}銘柄を非公開リポジトリに保存しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
