#!/usr/bin/env python3
"""
取引の報告の受け皿（research/trades_inbox.py）の単体テスト。通信しない。

固定すること
  - 暗号化した行は id のほかに中身（コード・日付・値段）が読めない。正しい鍵でだけ元に戻る
  - 中身や id を書き換えた行、別の鍵の行は戻らず、数えるだけで止まらない
  - 報告の形の確かめ（買いは予測日か日付、売りは日付、取消は id の先頭 8 文字以上）
  - add は1行足して、id の先頭だけを出す（銘柄・値段を出さない）
  - pubkey は環境変数の秘密鍵から公開鍵を出す

  python3 tests/test_trades_inbox.py
"""
import base64
import contextlib
import datetime as dt
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import trades_inbox as TI  # noqa: E402

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    HAVE_CRYPTO = True
except Exception:                         # noqa: BLE001  手元の環境で読み込めないとき
    HAVE_CRYPTO = False

NOW = dt.datetime(2026, 10, 8, 20, 0, tzinfo=TI.JST)


class MakeEvent(unittest.TestCase):
    def test_buy(self):
        e = TI.make_event("buy", " 6912 ", pick_date="2026/09/24", price="2,691", shares="100",
                          now=NOW)
        self.assertEqual(len(e["id"]), 32)
        self.assertEqual({k: e[k] for k in ("at", "action", "pick_date", "code", "date", "price",
                                            "shares", "ref")},
                         {"at": "2026-10-08T20:00:00+09:00", "action": "buy",
                          "pick_date": "2026-09-24", "code": "6912", "date": None,
                          "price": 2691.0, "shares": 100.0, "ref": None})
        self.assertEqual(TI.make_event("buy", "130a", date="2026-10-09")["code"], "130A")
        self.assertNotEqual(TI.make_event("buy", "6912", "2026-09-24")["id"],
                            TI.make_event("buy", "6912", "2026-09-24")["id"])

    def test_sell_and_cancel(self):
        s = TI.make_event("sell", "6912", date="2026-10-20")
        self.assertEqual((s["action"], s["date"], s["price"]), ("sell", "2026-10-20", None))
        c = TI.make_event("cancel", ref="ABCDEF12")
        self.assertEqual((c["action"], c["ref"], c["code"]), ("cancel", "abcdef12", None))

    def test_bad_reports_are_refused(self):
        bad = [dict(action="hold", code="6912", pick_date="2026-09-24"),
               dict(action="buy", code="", pick_date="2026-09-24"),
               dict(action="buy", code="69120", pick_date="2026-09-24"),        # 5桁
               dict(action="buy", code="6912"),                                  # 日付が無い
               dict(action="buy", code="6912", pick_date="2026-09-24", date="2026-09-24"),
               dict(action="buy", code="6912", pick_date="2026-09-31"),          # 無い日
               dict(action="buy", code="6912", pick_date="2026-09-24", price="0"),
               dict(action="buy", code="6912", pick_date="2026-09-24", shares="-100"),
               dict(action="sell", code="6912"),                                 # 売った日が無い
               dict(action="cancel", ref="abc1234"),                             # 7文字
               dict(action="cancel", ref="zzzzzzzz")]                            # 16進でない
        for kw in bad:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                TI.make_event(**kw)


@unittest.skipUnless(HAVE_CRYPTO, "cryptography を読み込めない")
class Crypto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.pub = TI.public_pem(cls.key)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.event = TI.make_event("buy", "6912", pick_date="2026-09-24", price="2691", now=NOW)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_round_trip_and_nothing_readable(self):
        line = TI.encrypt(self.event, self.pub)
        self.assertEqual(set(line), {"v", "id", "k", "n", "c"})
        self.assertEqual(line["id"], self.event["id"])
        self.assertNotIn(b"6912", base64.b64decode(line["c"]))
        self.assertNotIn(b"pick_date", base64.b64decode(line["c"]))
        self.assertEqual(TI.decrypt(line, self.key), self.event)

    def test_tampered_or_foreign_lines_fail(self):
        line = TI.encrypt(self.event, self.pub)
        c = bytearray(base64.b64decode(line["c"]))
        c[0] ^= 1
        with self.assertRaises(Exception):
            TI.decrypt(dict(line, c=base64.b64encode(bytes(c)).decode()), self.key)
        with self.assertRaises(Exception):                 # id は暗号の付帯データ
            TI.decrypt(dict(line, id="0" * 32), self.key)
        with self.assertRaises(Exception):
            TI.decrypt(line, self.other)

    def test_read_inbox_counts_lines_it_cannot_open(self):
        path = os.path.join(self.dir, "inbox.jsonl")
        TI.append(path, TI.encrypt(self.event, self.pub))
        TI.append(path, TI.encrypt(TI.make_event("sell", "6912", date="2026-10-20"),
                                   TI.public_pem(self.other)))
        with open(path, "a", encoding="utf-8") as f:
            f.write("\nnot json\n")
        events, failed = TI.read_inbox(path, self.key)
        self.assertEqual(([e["id"] for e in events], failed), ([self.event["id"]], 2))
        self.assertEqual(TI.read_inbox(os.path.join(self.dir, "none.jsonl"), self.key), ([], 0))

    def test_cli_add_prints_only_the_id(self):
        pub = os.path.join(self.dir, "pub.pem")
        with open(pub, "wb") as f:
            f.write(self.pub)
        inbox = os.path.join(self.dir, "inbox.jsonl")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            TI.main(["add", "--action", "buy", "--code", "6912", "--pick-date", "2026-09-24",
                     "--price", "2691", "--inbox", inbox, "--pubkey", pub])
        text = out.getvalue()
        self.assertRegex(text, r"^\[trades\] 1件を暗号化して足した（id [0-9a-f]{8}）\n$")
        for word in ("6912", "2691", "2026"):
            self.assertNotIn(word, text)
        events, failed = TI.read_inbox(inbox, self.key)
        self.assertEqual((len(events), failed), (1, 0))
        self.assertEqual((events[0]["code"], events[0]["price"]), ("6912", 2691.0))
        self.assertTrue(text.strip().endswith(f"id {events[0]['id'][:8]}）"))

    def test_cli_pubkey_reads_the_service_account(self):
        pem = self.key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                     serialization.NoEncryption()).decode("ascii")
        old = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        try:
            os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps({"private_key": pem})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                TI.main(["pubkey"])
            self.assertEqual(out.getvalue().encode("ascii"), self.pub)
            self.assertNotIn("PRIVATE", out.getvalue())
            os.environ.pop("GOOGLE_SERVICE_ACCOUNT_JSON")
            with self.assertRaises(SystemExit):
                TI.main(["pubkey"])
        finally:
            if old is None:
                os.environ.pop("GOOGLE_SERVICE_ACCOUNT_JSON", None)
            else:
                os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = old


class RepoFiles(unittest.TestCase):
    def test_inbox_lines_are_encrypted(self):
        """リポジトリの受け皿に平文の行が混ざっていない（id と暗号文だけ）。"""
        if not os.path.exists(TI.INBOX):
            self.skipTest("受け皿がまだ無い")
        with open(TI.INBOX, encoding="utf-8") as f:
            for raw in f:
                if raw.strip():
                    line = json.loads(raw)
                    self.assertEqual(set(line), {"v", "id", "k", "n", "c"})
                    self.assertRegex(line["id"], r"^[0-9a-f]{32}$")

    def test_public_key_is_a_public_key(self):
        if not os.path.exists(TI.PUBKEY):
            self.skipTest("公開鍵がまだ無い")
        with open(TI.PUBKEY, encoding="ascii") as f:
            text = f.read()
        self.assertTrue(text.startswith("-----BEGIN PUBLIC KEY-----"))
        self.assertNotIn("PRIVATE", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
