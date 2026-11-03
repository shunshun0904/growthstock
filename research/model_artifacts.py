"""Install and verify fixed-return artifacts without reusing legacy sigma binaries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import label_specs as LS

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "research/model"
PREFIX = LS.VERSION + "_"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def directory(algo, root=MODEL_ROOT):
    return Path(root) if algo == "lgbm" else Path(root) / "models" / algo

def filename(algo, kind):
    if algo == "lgbm":
        local = "model.txt" if kind == "model" else "oof.parquet"
    else:
        local = algo + ("_model.joblib" if kind == "model" else "_oof.parquet")
    return PREFIX + local

def local_name(algo, kind):
    return ("model.txt" if algo == "lgbm" else "model.joblib") if kind == "model" else "oof.parquet"

def hashes(algo, root=MODEL_ROOT):
    d = directory(algo, root)
    return {local_name(algo, k): sha256(d / local_name(algo, k)) for k in ("model", "oof")}

def require(meta, algo, root=MODEL_ROOT, kinds=("model", "oof")):
    LS.require(meta, algo)
    for kind in kinds:
        name = local_name(algo, kind)
        path = directory(algo, root) / name
        expected = meta.get("artifactHashes", {}).get(name)
        if not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{algo}: {name} does not match its training metadata")

def verify_all(root=MODEL_ROOT):
    for algo in LS.BASE_MODELS + LS.STRICT_MODELS:
        meta = json.loads((directory(algo, root) / "meta.json").read_text())
        require(meta, algo, root)

def overlay(data_dir, root=MODEL_ROOT):
    """Supply current OOFs to research consumers of the flat raw-data folder."""
    data_dir = Path(data_dir)
    for algo in LS.BASE_MODELS + LS.STRICT_MODELS:
        meta = json.loads((directory(algo, root) / "meta.json").read_text())
        LS.require(meta, algo)
        for kind in ("model", "oof"):
            src = data_dir / filename(algo, kind)
            name = local_name(algo, kind)
            expected = meta.get("artifactHashes", {}).get(name)
            if not src.is_file() or not expected or sha256(src) != expected:
                raise ValueError(f"{algo}: downloaded fixed-return {kind} is missing or inconsistent")
            dst = (name if algo == "lgbm" else algo + "_" + name)
            shutil.copyfile(src, data_dir / dst)

def install(root=MODEL_ROOT):
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="growthstock-models-") as temp:
        subprocess.run(["bash", str(ROOT / "scripts/gh_release_download.sh"),
                        "data-raw", temp, PREFIX + "*"], check=True, cwd=ROOT)
        for algo in LS.BASE_MODELS + LS.STRICT_MODELS:
            dest = directory(algo, root);dest.mkdir(parents=True, exist_ok=True)
            meta = json.loads((dest / "meta.json").read_text())
            LS.require(meta, algo)
            # Verify the complete pair before installing either file.
            for kind in ("model", "oof"):
                src = Path(temp) / filename(algo, kind)
                expected = meta.get("artifactHashes", {}).get(local_name(algo, kind))
                if not src.is_file() or not expected or sha256(src) != expected:
                    raise ValueError(f"{algo}: incomplete fixed-return release")
            for kind in ("model", "oof"):
                shutil.copyfile(Path(temp) / filename(algo, kind), dest / local_name(algo, kind))
        verify_all(root)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["install", "verify", "overlay"])
    ap.add_argument("--model-dir", default=str(MODEL_ROOT))
    ap.add_argument("--data-dir")
    args = ap.parse_args()
    if args.action == "install": install(args.model_dir)
    elif args.action == "verify": verify_all(args.model_dir)
    else: overlay(args.data_dir, args.model_dir)
