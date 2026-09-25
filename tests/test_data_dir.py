#!/usr/bin/env python3
"""
学習データを作るときに、引数のデータ置き場（data_dir）がどの読み込みにも渡ることのテスト。

2026-09-25、build_dataset.build(data_dir, ...) を別の置き場のデータで回すと、
バリュエーション（/equities/valuation の PER・PBR・ROE）だけが既定の research/_data から
読まれていた（api_valuation の呼び出しに data_dir を渡していなかった）。既定の置き場に
データが無いと API の値が丸ごと抜け、ある場合は別の時点のデータが混ざる。本番は既定の
置き場で作るので影響しないが、生データの写しで作り直す比べ方（実験45・§10 の確かめ）を
誤らせる。
"""
import ast
import inspect
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import build_dataset as B  # noqa: E402

#: data_dir を受け取って読み込む関数と、data_dir を渡す位置（位置引数で渡すときの番号）
READERS = {"api_valuation": 2, "quarterize_panel": 1, "load_parts": 1,
           "load_market_segments": 0}


def calls(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if fname == name:
                yield node


def passes_data_dir(call: ast.Call, pos: int) -> bool:
    if any(k.arg == "data_dir" for k in call.keywords):
        return True
    return len(call.args) > pos and isinstance(call.args[pos], ast.Name) \
        and call.args[pos].id == "data_dir"


class TestDataDirIsPassed(unittest.TestCase):
    def test_every_reader_call_gets_data_dir(self):
        tree = ast.parse(inspect.getsource(B))
        for name, pos in READERS.items():
            found = list(calls(tree, name))
            self.assertTrue(found, f"{name} の呼び出しが見つからない")
            for c in found:
                self.assertTrue(passes_data_dir(c, pos),
                                f"{name} の呼び出し（{c.lineno}行目）に data_dir を渡していない")

    def test_readers_accept_data_dir(self):
        for name in READERS:
            params = inspect.signature(getattr(B, name)).parameters
            self.assertIn("data_dir", params, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
