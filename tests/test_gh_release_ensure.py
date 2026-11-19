#!/usr/bin/env python3
"""
scripts/gh_release_ensure.sh（Release が無ければ作る）のテスト。偽の gh を PATH の先頭に置く。

- あれば何もしない（作らない）
- 無ければ作る（題と説明を渡す）
- 作るのが一時的に失敗しても、待って試し直す
- 試し直しても作れなければ exit 1
- タグが無ければ exit 2
- GH_ENSURE_DRAFT=1: 下書きで作る。既にあるものが下書きでなければ（公開されていれば）exit 3 で止める

  python3 -m unittest tests/test_gh_release_ensure.py
"""
import os
import stat
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "gh_release_ensure.sh")

# 偽の gh。Release を $STUB_STATE/<tag>/ のディレクトリで表す。create は $STUB_CREATE_FAILS 回だけ失敗する
FAKE_GH = r"""#!/usr/bin/env bash
set -u
state="$STUB_STATE"
cmd="$1 $2"; tag="$3"; shift 3
if [ "$cmd" = "release view" ]; then
  if [ -d "$state/$tag" ]; then
    if [ -f "$state/$tag/draft" ]; then echo true; else echo false; fi
    exit 0
  fi
  echo "release not found" >&2; exit 1
fi
if [ "$cmd" = "release create" ]; then
  n=$(( $(cat "$state/creates" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$state/creates"
  if [ "$n" -le "${STUB_CREATE_FAILS:-0}" ]; then echo "HTTP 502" >&2; exit 1; fi
  mkdir -p "$state/$tag"; printf '%s\n' "$@" > "$state/$tag/args"
  for a in "$@"; do [ "$a" = "--draft" ] && touch "$state/$tag/draft"; done
  exit 0
fi
echo "fake gh: unknown $cmd" >&2; exit 3
"""


class TestEnsure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.bin = os.path.join(d, "bin")
        self.state = os.path.join(d, "state")
        os.makedirs(self.bin)
        os.makedirs(self.state)
        gh = os.path.join(self.bin, "gh")
        with open(gh, "w") as fh:
            fh.write(FAKE_GH)
        os.chmod(gh, os.stat(gh).st_mode | stat.S_IEXEC)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args, **env):
        e = dict(os.environ, PATH=self.bin + os.pathsep + os.environ["PATH"], STUB_STATE=self.state,
                 GH_ENSURE_WAIT="0", **env)
        return subprocess.run(["bash", SCRIPT, *args], env=e, capture_output=True, text=True, timeout=60)

    def creates(self):
        p = os.path.join(self.state, "creates")
        if not os.path.exists(p):
            return 0
        with open(p) as fh:
            return int(fh.read())

    def test_existing_release_is_left_alone(self):
        os.makedirs(os.path.join(self.state, "data-tick-2025"))
        r = self.run_script("data-tick-2025", "題", "説明")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.creates(), 0)

    def test_missing_release_is_created_with_title_and_notes(self):
        r = self.run_script("data-tick-2026", "J-Quants ティック data-tick-2026", "日ごとのファイル")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.state, "data-tick-2026", "args")) as fh:
            args = fh.read().splitlines()
        self.assertIn("J-Quants ティック data-tick-2026", args)
        self.assertIn("--latest=false", args)

    def test_transient_failure_is_retried(self):
        r = self.run_script("data-tick-2026", STUB_CREATE_FAILS="2")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.creates(), 3)
        self.assertTrue(os.path.isdir(os.path.join(self.state, "data-tick-2026")))

    def test_gives_up(self):
        r = self.run_script("data-tick-2026", STUB_CREATE_FAILS="99", GH_ENSURE_TRIES="3")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(self.creates(), 3)

    def test_draft_is_created_as_draft(self):
        r = self.run_script("data-minute", "題", "説明", GH_ENSURE_DRAFT="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.state, "data-minute", "draft")))
        # 2回目は作らない（下書きのまま見つかる）
        r = self.run_script("data-minute", GH_ENSURE_DRAFT="1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.creates(), 1)

    def test_published_release_is_refused_when_draft_is_required(self):
        os.makedirs(os.path.join(self.state, "data-minute"))           # 公開（下書きの印が無い）
        r = self.run_script("data-minute", GH_ENSURE_DRAFT="1")
        self.assertEqual(r.returncode, 3)
        self.assertIn("公開になっている", r.stderr)
        self.assertEqual(self.creates(), 0)
        # 下書きを求めないなら、公開でもそのまま
        self.assertEqual(self.run_script("data-minute").returncode, 0)

    def test_usage(self):
        self.assertEqual(self.run_script().returncode, 2)


if __name__ == "__main__":
    unittest.main()
