#!/usr/bin/env python3
"""
公開の Release を下書き（非公開）に戻す手順の決めごと（.github/workflows/make-releases-private.yml ほか）。

運用者の選択（2026-11-27）「案のとおり進める」「3つとも一緒に非公開」「土曜 21:00」。docs/OPERATIONS.md「data-raw を非公開にする」。

- 下書きに戻すワークフローは手でだけ起動し、confirm が yes でなければ何もしない。ほかの run が動いていれば止める
- 対象の既定は data-raw・data-jsf・pre-dedupe-20260924・data-volume-profile
- 下書きに戻し（--draft=true）、git のタグを消し、鍵なしで見えないこと・鍵ありで取れることを確かめる。公開には戻さない
- data-raw・data-jsf を作るワークフローは、無いときに下書きで作る

  python3 -m unittest tests.test_make_releases_private
"""
import os
import re
import unittest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, ".github", "workflows")


def load(name):
    with open(os.path.join(WF, name), encoding="utf-8") as fh:
        text = fh.read()
    return text, yaml.safe_load(text)


class TestMakeReleasesPrivate(unittest.TestCase):
    def setUp(self):
        self.text, self.wf = load("make-releases-private.yml")

    def test_manual_only_and_confirmed(self):
        on = self.wf[True]                                               # yaml は "on" を True と読む
        self.assertEqual(set(on), {"workflow_dispatch"})
        inputs = on["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["confirm"]["default"], "no")
        self.assertTrue(inputs["confirm"]["required"])
        self.assertEqual(inputs["tags"]["default"].split(),
                         ["data-raw", "data-jsf", "pre-dedupe-20260924", "data-volume-profile"])
        first = self.wf["jobs"]["private"]["steps"][1]["run"]
        self.assertIn('if [ "$CONFIRM" != "yes" ]', first)
        self.assertIn("status=in_progress", first)
        self.assertIn("status=queued", first)

    def test_permissions(self):
        self.assertEqual(self.wf["permissions"], {"contents": "write", "actions": "read"})

    def test_converts_deletes_tag_and_checks(self):
        run = self.wf["jobs"]["private"]["steps"][2]["run"]
        self.assertIn('gh release edit "$t" --draft=true', run)
        self.assertIn('-X DELETE "repos/$GH_REPO/git/refs/tags/$t"', run)
        for want in ("鍵なしで Release（API）を取れない", "鍵なしでアセット（API）を取れない",
                     "鍵なしで前のダウンロードの URL から取れない", "鍵なしで前の Release のページを開けない",
                     "鍵ありでタグ名から見つかる", "取り出しの台本で同じ中身が取れる", "アセットの数が変わらない",
                     "鍵なしの Release の一覧に出ない"):
            self.assertIn(want, run)
        self.assertIn("scripts/gh_release_download.sh", run)
        self.assertIn('[ "${#ng[@]}" -eq 0 ]', run)                       # 1つでも FAIL なら失敗にする
        self.assertNotIn("--draft=false", self.text)                      # 公開には戻さない
        self.assertNotIn("gh release delete", self.text)                  # Release そのものは消さない


class TestCreateAsDraft(unittest.TestCase):
    """data-raw・data-jsf（と生データの写し）を作る手順は、無いときに下書きで作る。"""

    def test_create_lines_are_draft(self):
        for name in ("update-data.yml", "fetch-jsf.yml", "snapshot-raw.yml", "migrate-to-release.yml"):
            text, _ = load(name)
            lines = [ln for ln in text.splitlines() if re.search(r"\bgh release create\b", ln)
                     and not ln.lstrip().startswith("#")]
            self.assertTrue(lines, name)
            for ln in lines:
                self.assertIn("--draft", ln, f"{name}: {ln.strip()}")

    def test_download_required_for_private_tags(self):
        with open(os.path.join(ROOT, "scripts", "gh_release_download.sh"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn('if [ "$tag" = "data-raw" ] || [ "$tag" = "data-jsf" ]; then', text)

    def test_readers_use_write_token(self):
        """data-raw を読むワークフローは、下書きが見える書き込みの鍵を使う。"""
        for name in sorted(os.listdir(WF)):
            if not name.endswith(".yml"):
                continue
            text, wf = load(name)
            reads = re.search(r"gh_release_download\.sh (data-raw|data-jsf|\"\$RELEASE_TAG\")|gh release download (data-raw|\"\$SNAP\")",
                              text)
            if not reads:
                continue
            perms = [wf.get("permissions") or {}] + [(j or {}).get("permissions") or {} for j in wf["jobs"].values()]
            self.assertIn("write", [p.get("contents") for p in perms if p], name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
