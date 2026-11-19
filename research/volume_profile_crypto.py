#!/usr/bin/env python3
"""価格帯別出来高の集計を、J-Quants の API キーから作った鍵で暗号化する（docs/VOLUME_PROFILE.md「暗号化」）。

集計は J-Quants のアドオンのデータから作るので、そのまま公開しない。Pages に載せるのはこの暗号文だけで、
画面（src/lib/volumeProfile.js）で利用者が J-Quants の API キーを入れると、ブラウザの中で同じ鍵を作って開く。
キーはどこにも送らない。

  鍵      PBKDF2-HMAC-SHA256（ITERATIONS 回、SALT）で J-Quants の API キー（前後の空白を除く）から 256 ビット
  暗号    AES-256-GCM。IV は毎回ランダムな 12 バイト。見出し（data 以外の項目）を追加認証データに入れる
  中身    集計の JSON を gzip で縮めたもの

  python3 research/volume_profile_crypto.py check FILE   見出しの形だけを確かめる（鍵は要らない。デプロイの前に使う）
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import os
import re
import sys

KIND = "growthstock.volume-profile.encrypted"
VERSION = 1
#: OWASP の PBKDF2-HMAC-SHA256 の目安（60万回）。ブラウザでは1回 0.2〜1秒
ITERATIONS = 600_000
#: 固定の塩（このアプリの用途を分けるため）。変えると画面に覚えた鍵が使えなくなり、キーを入れ直す
SALT = bytes.fromhex("e0858ccad795070ea6c47c17c667a429")
FIELDS = ("kind", "version", "asOf", "generatedAt", "kdf", "cipher", "compression", "data")
MAX_BYTES = 20 * 1024 * 1024


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str, size: int = 0) -> bytes:
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError, TypeError):
        raise ValueError("base64 として読めない") from None
    if size and len(raw) != size:
        raise ValueError("長さが想定と違う")
    return raw


def aad(doc: dict) -> bytes:
    """追加認証データ。画面（src/lib/volumeProfile.js の envelopeAad）と同じ並び。"""
    kdf, cipher = doc["kdf"], doc["cipher"]
    parts = [doc["kind"], doc["version"], doc["asOf"], doc["generatedAt"], kdf["name"], kdf["hash"],
             kdf["iterations"], kdf["salt"], cipher["name"], cipher["iv"], cipher["tagLength"], doc["compression"]]
    return "\n".join(str(p) for p in parts).encode("utf-8")


def derive_key(secret: str, salt: bytes = SALT, iterations: int = ITERATIONS) -> bytes:
    secret = (secret or "").strip()
    if not secret:
        raise ValueError("J-Quants の API キーが空")
    return hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), salt, iterations, 32)


def check_envelope(doc) -> dict:
    """暗号文の見出しの形を確かめる（平文を誤って載せないため。鍵は要らない）。"""
    if not isinstance(doc, dict) or tuple(sorted(doc)) != tuple(sorted(FIELDS)):
        raise ValueError("暗号化した集計の項目が想定と違う")
    if doc["kind"] != KIND or doc["version"] != VERSION or doc["compression"] != "gzip":
        raise ValueError("暗号化した集計の種類・版が想定と違う")
    if not (isinstance(doc["asOf"], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc["asOf"])):
        raise ValueError("asOf が日付でない")
    if not isinstance(doc["generatedAt"], str) or not doc["generatedAt"]:
        raise ValueError("generatedAt が無い")
    kdf, cipher = doc["kdf"], doc["cipher"]
    if (not isinstance(kdf, dict) or sorted(kdf) != ["hash", "iterations", "name", "salt"]
            or kdf["name"] != "PBKDF2" or kdf["hash"] != "SHA-256" or type(kdf["iterations"]) is not int
            or not 100_000 <= kdf["iterations"] <= 10_000_000):
        raise ValueError("鍵の作り方が想定と違う")
    _unb64(kdf["salt"], 16)
    if (not isinstance(cipher, dict) or sorted(cipher) != ["iv", "name", "tagLength"]
            or cipher["name"] != "AES-GCM" or cipher["tagLength"] != 128):
        raise ValueError("暗号の方式が想定と違う")
    _unb64(cipher["iv"], 12)
    data = _unb64(doc["data"])
    if len(data) < 16 + 20:
        raise ValueError("暗号文が短すぎる")
    return doc


def encrypt(plain: bytes, secret: str, as_of: str, generated_at: str,
            iterations: int = ITERATIONS, salt: bytes = SALT, iv: bytes = None) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    iv = os.urandom(12) if iv is None else iv
    doc = {"kind": KIND, "version": VERSION, "asOf": as_of, "generatedAt": generated_at,
           "kdf": {"name": "PBKDF2", "hash": "SHA-256", "iterations": iterations, "salt": _b64(salt)},
           "cipher": {"name": "AES-GCM", "iv": _b64(iv), "tagLength": 128}, "compression": "gzip", "data": ""}
    body = gzip.compress(plain, compresslevel=9, mtime=0)
    doc["data"] = _b64(AESGCM(derive_key(secret, salt, iterations)).encrypt(iv, body, aad(doc)))
    return check_envelope(doc)


def decrypt(doc: dict, secret: str) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    check_envelope(doc)
    key = derive_key(secret, _unb64(doc["kdf"]["salt"], 16), doc["kdf"]["iterations"])
    try:
        body = AESGCM(key).decrypt(_unb64(doc["cipher"]["iv"], 12), _unb64(doc["data"]), aad(doc))
    except InvalidTag:
        raise ValueError("鍵が違うか、暗号文が壊れている") from None
    return gzip.decompress(body)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "check":
        print("使い方: volume_profile_crypto.py check FILE", file=sys.stderr)
        return 2
    try:
        if os.path.getsize(args[1]) > MAX_BYTES:
            raise ValueError("大きすぎる")
        with open(args[1], encoding="utf-8") as fh:
            doc = check_envelope(json.load(fh))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"暗号化した集計の確かめに失敗: {exc}", file=sys.stderr)
        return 1
    print(f"暗号化した集計: asOf {doc['asOf']} / 暗号文 {len(_unb64(doc['data'])):,} バイト")
    return 0


if __name__ == "__main__":
    sys.exit(main())
