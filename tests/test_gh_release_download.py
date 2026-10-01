#!/usr/bin/env python3
"""
scripts/gh_release_download.sh（Release からの取り出しを「一部だけ」にしない）のテスト。

偽の gh を PATH の先頭に置き、Release の中身をディレクトリで持つ。2026-10-01 に起きたこと
（1アセットだけ HTTP 500 で取り出せず、手順は「初回」とみなして続いた）を再現して確かめる。

- 失敗が無ければ1回で全ファイルが同じ大きさでそろう
- 途中で失敗したら、足りないものだけを取り直してそろえる
- gh が成功を返しても大きさが違えば取り直す
- 失敗が続けば exit 1（一部だけで先へ進ませない）
- Release が無い・合うアセットが無いのは初回として exit 0
- パターンで絞ったら、それ以外は取らない
- 一覧が一時的に引けなくても、やり直して進む

  python3 tests/test_gh_release_download.py
"""
import os
import stat
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "gh_release_download.sh")

FAKE_GH = r"""#!/usr/bin/env bash
set -u
state="$STUB_STATE"
cmd="$1 $2"; tag="$3"; shift 3
count() { local f="$state/$1"; local n=$(( $(cat "$f" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$f"; echo "$n"; }
if [ "$cmd" = "release view" ]; then
  n=$(count view_calls)
  if [ "${STUB_NOTFOUND:-0}" = "1" ]; then echo "release not found" >&2; exit 1; fi
  if [[ ",${STUB_VIEW_FAIL_CALLS:-}," == *",$n,"* ]]; then echo "HTTP 502: Bad Gateway" >&2; exit 1; fi
  for f in "$state/rel/"*; do [ -f "$f" ] && printf '%s\t%s\n' "$(basename "$f")" "$(stat -c %s "$f")"; done
  exit 0
fi
if [ "$cmd" = "release download" ]; then
  n=$(count dl_calls)
  dir=""; pats=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --dir) dir="$2"; shift 2;;
      --pattern) pats+=("$2"); shift 2;;
      *) shift;;
    esac
  done
  printf '%s\n' "${pats[*]:-ALL}" >> "$state/dl_patterns"
  matched=()
  for f in "$state/rel/"*; do
    b=$(basename "$f"); ok=0
    if [ ${#pats[@]} -eq 0 ]; then ok=1; fi
    for p in "${pats[@]}"; do [[ "$b" == $p ]] && ok=1; done
    [ $ok = 1 ] && matched+=("$f")
  done
  if [[ ",${STUB_DL_FAIL_CALLS:-}," == *",$n,"* ]]; then
    cp "${matched[0]}" "$dir/"
    echo "HTTP 500 (https://api.github.com/repos/x/y/releases/assets/1)" >&2
    exit 1
  fi
  for f in "${matched[@]}"; do cp "$f" "$dir/"; done
  if [[ ",${STUB_TRUNC_CALLS:-}," == *",$n,"* ]]; then
    head -c 3 "${matched[0]}" > "$dir/$(basename "${matched[0]}")"
  fi
  exit 0
fi
echo "fake gh: unknown $cmd" >&2; exit 3
"""


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.bin = os.path.join(d, "bin")
        self.state = os.path.join(d, "state")
        self.rel = os.path.join(self.state, "rel")
        self.out = os.path.join(d, "out")
        for p in (self.bin, self.rel):
            os.makedirs(p)
        gh = os.path.join(self.bin, "gh")
        with open(gh, "w") as fh:
            fh.write(FAKE_GH)
        os.chmod(gh, os.stat(gh).st_mode | stat.S_IEXEC)
        self.files = {"bars_2025.parquet": "x" * 40, "bars_2026.parquet": "y" * 25,
                      "jsf_hist.parquet": "h" * 30, "manifest.json": "{}"}
        for n, body in self.files.items():
            with open(os.path.join(self.rel, n), "w") as fh:
                fh.write(body)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args, **env):
        e = dict(os.environ, PATH=self.bin + os.pathsep + os.environ["PATH"],
                 STUB_STATE=self.state, GH_DOWNLOAD_WAIT="0")
        e.update({k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", SCRIPT, *args], env=e, capture_output=True, text=True)

    def local(self):
        if not os.path.isdir(self.out):
            return {}
        out = {}
        for n in os.listdir(self.out):
            with open(os.path.join(self.out, n)) as fh:
                out[n] = fh.read()
        return out

    def calls(self, name):
        p = os.path.join(self.state, name)
        return int(open(p).read()) if os.path.exists(p) else 0

    def patterns(self):
        return open(os.path.join(self.state, "dl_patterns")).read().splitlines()

    def test_all_at_once(self):
        r = self.run_script("data-raw", self.out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.local(), self.files)
        self.assertEqual(self.calls("dl_calls"), 1)

    def test_partial_failure_refetches_only_missing(self):
        """10/1 の再現: 1回目は1ファイルだけ取れて 500。2回目は残りだけを取る。"""
        r = self.run_script("data-raw", self.out, STUB_DL_FAIL_CALLS="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.local(), self.files)
        self.assertEqual(self.calls("dl_calls"), 2)
        second = self.patterns()[1].split()
        self.assertEqual(len(second), 3)                      # 1回目に取れた1つは除く
        self.assertNotIn("bars_2025.parquet", second)

    def test_size_mismatch_is_refetched(self):
        r = self.run_script("data-raw", self.out, STUB_TRUNC_CALLS="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.local(), self.files)
        self.assertEqual(self.calls("dl_calls"), 2)

    def test_gives_up_loudly(self):
        r = self.run_script("data-raw", self.out, STUB_DL_FAIL_CALLS="1,2,3",
                            GH_DOWNLOAD_TRIES="3")
        self.assertEqual(r.returncode, 1)
        self.assertIn("そろわなかった", r.stderr)

    def test_release_not_found_is_first_run(self):
        r = self.run_script("data-raw", self.out, STUB_NOTFOUND="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("初回", r.stdout)
        self.assertEqual(self.calls("dl_calls"), 0)

    def test_no_matching_assets_is_first_run(self):
        r = self.run_script("data-jsf", self.out, "edinet_*")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("初回", r.stdout)
        self.assertEqual(self.calls("dl_calls"), 0)

    def test_pattern_limits_what_is_fetched(self):
        r = self.run_script("data-jsf", self.out, "jsf_*")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.local(), {"jsf_hist.parquet": "h" * 30})

    def test_listing_retry(self):
        r = self.run_script("data-raw", self.out, STUB_VIEW_FAIL_CALLS="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.local(), self.files)

    def test_listing_never_available_stops(self):
        r = self.run_script("data-raw", self.out, STUB_VIEW_FAIL_CALLS="1,2",
                            GH_DOWNLOAD_TRIES="2")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(self.calls("dl_calls"), 0)


class TestWorkflowsUseHelper(unittest.TestCase):
    """本番のワークフローは gh release download を直接呼ばず、確かめる手順を通す。"""

    WORKFLOWS = ["update-data.yml", "predict.yml", "fetch-edinetdb.yml",
                 "fetch-jsf.yml", "retrain-weekly.yml"]

    def test_no_bare_download(self):
        for w in self.WORKFLOWS:
            with open(os.path.join(ROOT, ".github", "workflows", w), encoding="utf-8") as fh:
                text = fh.read()
            lines = [ln for ln in text.splitlines()
                     if "gh release download" in ln and not ln.lstrip().startswith("#")]
            self.assertEqual(lines, [], f"{w}: gh release download を直接呼んでいる")
            self.assertIn("scripts/gh_release_download.sh", text, w)


if __name__ == "__main__":
    unittest.main(verbosity=2)
