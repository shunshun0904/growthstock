#!/usr/bin/env python3
"""
.github/workflows/fetch-jsf.yml の「Back up history before the refetch」（日証金の過去ファイルを取り直す前の控え。
docs/DATA_JSF.md「取り直し（2026-11）」）を、偽の gh で実際に走らせて確かめる。

- 控えが無ければ、今の過去ファイルを控えとして置き、取り出し直して同じ中身なら REFETCH_OK=1
- 控えがもうあれば置かずに REFETCH_OK=1
- Release の一覧が引けない run では取り直さない（REFETCH_OK を書かない。exit 0）
- 取り直しが始まった後（REFETCH_BEFORE 以降に取った銘柄がある）に控えが無ければ、作らずに止める（exit 1）
- 過去ファイルが無ければ控えも作らず、取り直さない

  python3 -m unittest tests.test_fetch_jsf_backup
"""
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, ".github", "workflows", "fetch-jsf.yml")

FAKE_GH = r"""#!/usr/bin/env bash
set -u
state="$STUB_STATE"
cmd="$1 $2"; tag="$3"; shift 3
if [ "$cmd" = "release view" ]; then
  if [ "${STUB_VIEW_FAIL:-0}" = "1" ]; then echo "HTTP 502" >&2; exit 1; fi
  jq=""
  while [ $# -gt 0 ]; do case "$1" in --jq) jq="$2"; shift 2;; *) shift;; esac; done
  for f in "$state/rel/"*; do
    [ -f "$f" ] || continue
    if [[ "$jq" == *'\t'* ]]; then printf '%s\t%s\n' "$(basename "$f")" "$(stat -c %s "$f")"
    else basename "$f"; fi
  done
  exit 0
fi
if [ "$cmd" = "release upload" ]; then
  echo x >> "$state/uploads"
  for a in "$@"; do [ -f "$a" ] && cp "$a" "$state/rel/"; done
  exit 0
fi
if [ "$cmd" = "release download" ]; then
  dir=""; pats=()
  while [ $# -gt 0 ]; do
    case "$1" in --dir) dir="$2"; shift 2;; --pattern) pats+=("$2"); shift 2;; *) shift;; esac
  done
  for f in "$state/rel/"*; do
    b=$(basename "$f")
    for p in "${pats[@]}"; do [[ "$b" == $p ]] && cp "$f" "$dir/"; done
  done
  exit 0
fi
echo "fake gh: unknown $cmd" >&2; exit 3
"""


def step_script():
    with open(WF, encoding="utf-8") as fh:
        wf = yaml.safe_load(fh)
    steps = wf["jobs"]["fetch"]["steps"]
    step = next(s for s in steps if s.get("name") == "Back up history before the refetch")
    return wf["env"], step["run"]


class TestBackupStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work = os.path.join(self.tmp, "work")
        self.state = os.path.join(self.tmp, "state")
        self.bin = os.path.join(self.tmp, "bin")
        self.runner = os.path.join(self.tmp, "runner")
        for d in (os.path.join(self.work, "research", "_data"), os.path.join(self.state, "rel"), self.bin,
                  self.runner):
            os.makedirs(d)
        for name in ("jsf_fetch.py", "probe_jsf.py"):
            os.symlink(os.path.join(ROOT, "research", name), os.path.join(self.work, "research", name))
        os.symlink(os.path.join(ROOT, "scripts"), os.path.join(self.work, "scripts"))
        gh = os.path.join(self.bin, "gh")
        with open(gh, "w") as fh:
            fh.write(FAKE_GH)
        os.chmod(gh, os.stat(gh).st_mode | stat.S_IEXEC)
        self.env_file = os.path.join(self.tmp, "github_env")
        open(self.env_file, "w").close()
        self.wf_env, self.script = step_script()
        self.hist = os.path.join(self.work, "research", "_data", "jsf_hist.parquet")
        self.backup_name = self.wf_env["HIST_BACKUP"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_hist(self, content=b"PAR1-fake-history-bytes"):
        with open(self.hist, "wb") as fh:
            fh.write(content)

    def write_manifest(self, at):
        m = {"daily": {}, "hist": {"72030": {"status": "ok", "at": at}}}
        with open(os.path.join(self.work, "research", "_data", "jsf_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(m, fh)

    def run_step(self, **extra):
        env = dict(os.environ)
        env.update({k: str(v) for k, v in self.wf_env.items()})
        env.update({"PATH": self.bin + os.pathsep + env["PATH"], "STUB_STATE": self.state,
                    "GITHUB_ENV": self.env_file, "RUNNER_TEMP": self.runner, "GH_TOKEN": "dummy",
                    "GH_UPLOAD_WAIT": "0", "GH_DOWNLOAD_WAIT": "0"})
        env.update(extra)
        return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", self.script],
                              cwd=self.work, env=env, capture_output=True, text=True, timeout=120)

    def github_env(self):
        with open(self.env_file, encoding="utf-8") as fh:
            return fh.read()

    def uploads(self):
        p = os.path.join(self.state, "uploads")
        if not os.path.exists(p):
            return 0
        with open(p, encoding="utf-8") as fh:
            return len(fh.read().split())

    def test_creates_and_verifies_backup_once(self):
        import pandas as pd
        pd.DataFrame({"申込日": pd.to_datetime(["2026-09-24"]), "code": ["72030"]}).to_parquet(self.hist, index=False)
        self.write_manifest("2026-10-01T13:30:00+09:00")
        r = self.run_step()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("REFETCH_OK=1", self.github_env())
        saved = os.path.join(self.state, "rel", self.backup_name)
        with open(saved, "rb") as a, open(self.hist, "rb") as b:
            self.assertEqual(a.read(), b.read())
        self.assertIn("控えを置いた", r.stdout)
        self.assertEqual(self.uploads(), 1)
        # 2回目は置かない
        open(self.env_file, "w").close()
        r2 = self.run_step()
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        self.assertIn("REFETCH_OK=1", self.github_env())
        self.assertIn("控えは Release にある", r2.stdout)
        self.assertEqual(self.uploads(), 1)

    def test_listing_failure_skips_refetch(self):
        self.write_hist()
        r = self.run_step(STUB_VIEW_FAIL="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("REFETCH_OK", self.github_env())
        self.assertEqual(self.uploads(), 0)

    def test_does_not_back_up_after_refetch_started(self):
        import pandas as pd
        pd.DataFrame({"申込日": pd.to_datetime(["2026-09-24"]), "code": ["72030"]}).to_parquet(self.hist, index=False)
        self.write_manifest("2026-11-13T13:30:00+09:00")          # 取り直しの日以降に取った銘柄がある
        r = self.run_step()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertNotIn("REFETCH_OK", self.github_env())
        self.assertEqual(self.uploads(), 0)
        self.assertFalse(os.path.exists(os.path.join(self.state, "rel", self.backup_name)))

    def test_missing_history_skips(self):
        r = self.run_step()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("REFETCH_OK", self.github_env())
        self.assertEqual(self.uploads(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
