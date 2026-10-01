#!/usr/bin/env python3
"""
scripts/gh_release_upload.sh（Release へのアップロードを一時的なエラーに強くする）のテスト。

本物の gh の代わりに、Release の中身をディレクトリで持つ偽の gh を PATH の先頭に置く。
2026-10-01 に起きたこと（--clobber が古い版を消した後、GitHub が 502 を返して gh が止まる）を
再現して、次を確かめる。

- 失敗が無ければ1回で終わり、全ファイルが新しい版になる
- 消された後に失敗しても、やり直しで全ファイルがそろい、どれも新しい版になる
  （古い版のまま残るファイルが無い）
- 失敗が続けば決めた回数で諦めて exit 1、何が無いかを言う
- gh が成功を返したのに一覧に無いファイルがあれば、上げ直す
- 引数の誤り（ファイルが無い）は exit 2

  python3 tests/test_gh_release_upload.py
"""
import os
import stat
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "gh_release_upload.sh")

# 偽の gh。Release を $STUB_STATE/<tag>/ のファイルで表す。
#   upload: 呼ばれた回数を数え、$STUB_FAIL_CALLS に入っている回では、1つ目のファイルを上げた後に
#           2つ目の古い版を消して exit 1（--clobber が消した後に 502 が返った状況）
#           $STUB_DROP_CALLS に入っている回では、全部上げて成功を返すが、最後のファイルを消しておく
#   view:   アセット名を1行ずつ（--jq の結果と同じ形）
FAKE_GH = r"""#!/usr/bin/env bash
set -u
state="$STUB_STATE"
cmd="$1 $2"; tag="$3"; shift 3
mkdir -p "$state/$tag"
if [ "$cmd" = "release upload" ]; then
  n=$(( $(cat "$state/calls" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$state/calls"
  files=()
  for a in "$@"; do [ "$a" = "--clobber" ] || files+=("$a"); done
  if [[ ",${STUB_FAIL_CALLS:-}," == *",$n,"* ]]; then
    cp "${files[0]}" "$state/$tag/$(basename "${files[0]}")"
    if [ "${#files[@]}" -gt 1 ]; then rm -f "$state/$tag/$(basename "${files[1]}")"; fi
    echo "HTTP 502: Server Error (https://api.github.com/repos/x/y/releases/assets/1)" >&2
    exit 1
  fi
  for f in "${files[@]}"; do cp "$f" "$state/$tag/$(basename "$f")"; done
  if [[ ",${STUB_DROP_CALLS:-}," == *",$n,"* ]]; then
    last="${files[${#files[@]}-1]}"; rm -f "$state/$tag/$(basename "$last")"
  fi
  exit 0
fi
if [ "$cmd" = "release view" ]; then
  ls -1 "$state/$tag"
  exit 0
fi
echo "fake gh: unknown $cmd" >&2; exit 3
"""


class TestUpload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.bin = os.path.join(d, "bin")
        self.state = os.path.join(d, "state")
        self.work = os.path.join(d, "work")
        for p in (self.bin, self.state, self.work):
            os.makedirs(p)
        gh = os.path.join(self.bin, "gh")
        with open(gh, "w") as fh:
            fh.write(FAKE_GH)
        os.chmod(gh, os.stat(gh).st_mode | stat.S_IEXEC)
        # Release には古い版が3つある
        os.makedirs(os.path.join(self.state, "data-raw"))
        self.names = ["a_2018.parquet", "b_2023.parquet", "c_2026.parquet"]
        for n in self.names:
            with open(os.path.join(self.state, "data-raw", n), "w") as fh:
                fh.write("old")
            with open(os.path.join(self.work, n), "w") as fh:
                fh.write("new")

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args, **env):
        e = dict(os.environ, PATH=self.bin + os.pathsep + os.environ["PATH"],
                 STUB_STATE=self.state, GH_UPLOAD_WAIT="0")
        e.update({k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", SCRIPT, *args], cwd=self.work, env=e,
                              capture_output=True, text=True)

    def release(self):
        out = {}
        for n in sorted(os.listdir(os.path.join(self.state, "data-raw"))):
            with open(os.path.join(self.state, "data-raw", n)) as fh:
                out[n] = fh.read()
        return out

    def calls(self):
        with open(os.path.join(self.state, "calls")) as fh:
            return int(fh.read())

    def test_no_failure_uploads_once(self):
        r = self.run_script("data-raw", *self.names)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(self.calls(), 1)
        self.assertEqual(self.release(), {n: "new" for n in self.names})

    def test_retry_after_delete_then_502_restores_everything(self):
        """10/1 の再現: 1回目は2つ目を消したところで 502。やり直しで全部が新しい版になる。"""
        r = self.run_script("data-raw", *self.names, STUB_FAIL_CALLS="1")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(self.calls(), 2)
        self.assertEqual(self.release(), {n: "new" for n in self.names})
        self.assertIn("全部を上げ直す", r.stdout)

    def test_gives_up_after_tries_and_names_missing(self):
        r = self.run_script("data-raw", *self.names, STUB_FAIL_CALLS="1,2,3",
                            GH_UPLOAD_TRIES="3")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(self.calls(), 3)
        self.assertIn("b_2023.parquet", r.stderr)

    def test_success_but_missing_asset_is_reuploaded(self):
        r = self.run_script("data-raw", *self.names, STUB_DROP_CALLS="1")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(self.calls(), 2)
        self.assertEqual(self.release(), {n: "new" for n in self.names})

    def test_single_file(self):
        r = self.run_script("data-raw", "c_2026.parquet", STUB_FAIL_CALLS="1")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual(self.release()["c_2026.parquet"], "new")

    def test_bad_arguments(self):
        self.assertEqual(self.run_script().returncode, 2)
        self.assertEqual(self.run_script("data-raw").returncode, 2)
        self.assertEqual(self.run_script("data-raw", "nope.parquet").returncode, 2)


class TestWorkflowsUseHelper(unittest.TestCase):
    """本番のワークフローは --clobber を直接呼ばず、この再試行つきの手順を通す。"""

    WORKFLOWS = ["update-data.yml", "predict.yml", "fetch-edinetdb.yml",
                 "fetch-jsf.yml", "retrain-weekly.yml"]

    def test_no_bare_clobber(self):
        for w in self.WORKFLOWS:
            with open(os.path.join(ROOT, ".github", "workflows", w), encoding="utf-8") as fh:
                text = fh.read()
            lines = [ln for ln in text.splitlines()
                     if "gh release upload" in ln and not ln.lstrip().startswith("#")]
            self.assertEqual(lines, [], f"{w}: gh release upload を直接呼んでいる")
            self.assertIn("scripts/gh_release_upload.sh", text, w)


if __name__ == "__main__":
    unittest.main(verbosity=2)
