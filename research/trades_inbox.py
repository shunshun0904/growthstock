#!/usr/bin/env python3
"""
取引の報告（買った・売った）を、公開のリポジトリに**暗号化して**置く受け皿。

なぜ要るか
--------
運用者はセッションのチャットで「買った（ことにした）」「売った（ことにした）」を報告し、それを
スプレッドシートに反映する（運用者の依頼。2026-10-08）。シートに書けるのは GitHub Actions の
サービスアカウントだけで、チャットのセッションからは書けない。リポジトリとログは公開なので、
銘柄・日付・値段をそのまま置くと取引が公開される。

そこで、サービスアカウントの公開鍵で暗号化した行を research/trades_inbox.jsonl に足し、
ワークフロー Update Holdings Sheet が秘密鍵（Secrets の GOOGLE_SERVICE_ACCOUNT_JSON）で復号して、
シートの「取引の記録」タブに写す（research/holdings_sheet.py）。公開されるのは、行を足した日時と
行数だけ。

暗号: 行ごとに使い捨ての AES-256-GCM の鍵で中身を暗号化し、その鍵を RSA-OAEP(SHA-256) で
公開鍵（research/trades_pubkey.pem）に包む。

サービスアカウントの鍵を作り直したら、公開鍵を取り直す（holdings.yml を mode=pubkey で起動し、
ログに出る公開鍵を research/trades_pubkey.pem に置く）。古い行は復号できなくなるが、取り込み済みの
取引はシートの「取引の記録」タブに残っているので失われない。

  python3 research/trades_inbox.py pubkey      # Actions: 公開鍵を出す（公開してよい）
  python3 research/trades_inbox.py add --action buy --pick-date 2026-09-24 --code 6912 \\
      [--date 2026-09-25] [--price 2691] [--shares 100] [--note ...]
  python3 research/trades_inbox.py add --action sell --code 6912 --date 2026-10-20 [--price 2960]
  python3 research/trades_inbox.py add --action cancel --ref 1a2b3c4d   # 取り消す行の id の先頭
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import sys
import uuid
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
INBOX = os.path.join(HERE, "trades_inbox.jsonl")
PUBKEY = os.path.join(HERE, "trades_pubkey.pem")
JST = dt.timezone(dt.timedelta(hours=9))
ACTIONS = ("buy", "sell", "cancel")
MIN_REF = 8                       # 取消の対象は id の先頭 8 文字以上（取引の記録タブの id 列）


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def encrypt(event: Dict, public_pem: bytes) -> Dict:
    """取引1件を暗号化した1行（id だけは平文。重複の見分けに使う）。"""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    pub = serialization.load_pem_public_key(public_pem)
    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    body = json.dumps(event, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ct = AESGCM(key).encrypt(nonce, body, event["id"].encode("ascii"))
    wrapped = pub.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                            algorithm=hashes.SHA256(), label=None))
    return {"v": 1, "id": event["id"], "k": _b64(wrapped), "n": _b64(nonce), "c": _b64(ct)}


def decrypt(line: Dict, private_key) -> Dict:
    """encrypt の逆。鍵が合わない・改ざんがあれば例外。"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = private_key.decrypt(_unb64(line["k"]),
                              padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                           algorithm=hashes.SHA256(), label=None))
    body = AESGCM(key).decrypt(_unb64(line["n"]), _unb64(line["c"]), line["id"].encode("ascii"))
    event = json.loads(body.decode("utf-8"))
    if event.get("id") != line["id"]:
        raise ValueError("id が合わない")
    return event


def private_key_from_env():
    """GOOGLE_SERVICE_ACCOUNT_JSON の秘密鍵。無ければ None（値は出さない）。"""
    from cryptography.hazmat.primitives import serialization

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        return None
    pem = json.loads(raw).get("private_key", "")
    if not pem:
        return None
    return serialization.load_pem_private_key(pem.encode("utf-8"), password=None)


def public_pem(private_key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def read_inbox(path: str, private_key) -> Tuple[List[Dict], int]:
    """受け皿の全行を復号する。返り値は (取引の一覧, 復号できなかった行の数)。"""
    if not os.path.exists(path):
        return [], 0
    events: List[Dict] = []
    failed = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(decrypt(json.loads(raw), private_key))
            except Exception:                 # 鍵の作り直しの前の行・壊れた行
                failed += 1
    return events, failed


def _date(s: Optional[str]) -> Optional[str]:
    if s in (None, ""):
        return None
    return dt.date.fromisoformat(str(s).replace("/", "-")).isoformat()


def _num(s) -> Optional[float]:
    if s in (None, ""):
        return None
    v = float(str(s).replace(",", ""))
    if v <= 0:
        raise ValueError(f"正の数ではない: {s}")
    return v


def make_event(action: str, code: Optional[str] = None, pick_date: Optional[str] = None,
               date: Optional[str] = None, price=None, shares=None, note: str = "",
               now: Optional[dt.datetime] = None, ref: Optional[str] = None) -> Dict:
    """報告1件。値の形をここで確かめる（買いは予測日か日付、売りは日付、取消は対象の id が要る）。"""
    if action not in ACTIONS:
        raise ValueError(f"action は {ACTIONS} のどれか: {action}")
    code = str(code or "").strip().upper()
    ref = str(ref or "").strip().lower()
    if action == "cancel":
        if not re.fullmatch(r"[0-9a-f]{%d,32}" % MIN_REF, ref):
            raise ValueError(f"取消には 取り消す行の id（先頭 {MIN_REF} 文字以上）が要る")
    elif not re.fullmatch(r"[0-9][0-9A-Z]{3}", code):
        raise ValueError(f"code は4桁の銘柄コード: {code!r}")
    pick = _date(pick_date)
    day = _date(date)
    if action == "buy" and pick is None and day is None:
        raise ValueError("買いには 予測日 か 買った日 のどちらかが要る")
    if action == "buy" and pick is not None and day is not None and day <= pick:
        raise ValueError("買った日が予測日以前（予測は引けのあとに出るので、買うのは次の営業日から）")
    if action == "sell" and day is None:
        raise ValueError("売りには 売った日 が要る")
    now = now or dt.datetime.now(JST)
    return {"id": uuid.uuid4().hex, "at": now.isoformat(timespec="seconds"),
            "action": action, "pick_date": pick, "code": code or None, "date": day,
            "price": _num(price), "shares": _num(shares), "note": str(note or ""),
            "ref": ref or None}


def append(path: str, line: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="取引の報告を暗号化して受け皿に足す")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pubkey", help="サービスアカウントの公開鍵を出す（Actions で使う）")
    a = sub.add_parser("add", help="取引を1件足す（手元で使う）")
    a.add_argument("--action", required=True, choices=ACTIONS)
    a.add_argument("--code", default=None, help="銘柄コード（買い・売り）")
    a.add_argument("--pick-date", default=None, help="予測日（予測ログの行を見つけるのに使う）")
    a.add_argument("--date", default=None, help="買った日・売った日（買いは省くと予測日の次の営業日）")
    a.add_argument("--price", default=None, help="値段（省くと買いはその日の始値、売りはその日の終値）")
    a.add_argument("--shares", default=None, help="株数（省くと100）")
    a.add_argument("--ref", default=None, help="取消: 取り消す行の id の先頭 8 文字以上")
    a.add_argument("--note", default="")
    a.add_argument("--inbox", default=INBOX)
    a.add_argument("--pubkey", default=PUBKEY)
    args = ap.parse_args(argv)

    if args.cmd == "pubkey":
        key = private_key_from_env()
        if key is None:
            raise SystemExit("GOOGLE_SERVICE_ACCOUNT_JSON が無い")
        sys.stdout.write(public_pem(key).decode("ascii"))
        return 0
    with open(args.pubkey, "rb") as f:
        pem = f.read()
    event = make_event(args.action, args.code, args.pick_date, args.date,
                       args.price, args.shares, args.note, ref=args.ref)
    append(args.inbox, encrypt(event, pem))
    print(f"[trades] 1件を暗号化して足した（id {event['id'][:8]}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
