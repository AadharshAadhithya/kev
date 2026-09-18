"""Run kev studies on Modal: one GPU container per trial, results pulled back into runs/.

    KEV_GPU=T4 uv run modal run modal_app.py::smoke                        # ~2 min end to end on a T4 (free tier)
    uv run modal run modal_app.py::study --suite evals/decision-v1 \\
        --plan experiments/mbp-comparison.json --name mbp-comparison-v1     # N trials in parallel on H100s
    uv run modal run modal_app.py::evaluate --run jaredpalmer/kev-0.5b \\
        --suite evals/transfer-v1 --name transfer-kev-v01-h100             # score a Hub checkpoint

The same `kev.experiment.execute_trial` runs here and on the MBP; only the device differs. Every trial records
the local git commit (KEV_GIT_COMMIT), the suite hash, and the hashes of the kev/*.py files that were shipped, and
`kev.experiment --aggregate` ranks the study locally afterwards so the ledger is produced by one code path.

Volumes: kev-hf-cache (base weights, downloaded once), kev-runs (trial outputs). Secrets: none required; set
KEV_HF_SECRET=<modal secret name> to attach a Secret carrying HF_TOKEN for gated bases.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "kev-research"
ROOT = Path(__file__).resolve().parent
RUNS_MOUNT, HF_MOUNT = "/runs", "/hf"
GPU = os.environ.get("KEV_GPU", "H100")   # H100 needs a payment method on the workspace; KEV_GPU=T4 for the free tier

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(ROOT), groups=[])           # exact locked deps; Linux torch wheels are the CUDA build
    .env({"HF_HOME": HF_MOUNT, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("kev")
    .add_local_file(ROOT / "uv.lock", "/root/uv.lock")
    .add_local_file(ROOT / "pyproject.toml", "/root/pyproject.toml")
    .add_local_dir(ROOT / "evals", "/root/evals")
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)
secrets = [modal.Secret.from_name(os.environ["KEV_HF_SECRET"])] if os.environ.get("KEV_HF_SECRET") else []


def local_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def local_source_hashes():
    sys.path.insert(0, str(ROOT))
    from kev.experiment import source_hashes
    return source_hashes()


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 49152), max_containers=4, retries=0, timeout=1800,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_trial(study, index, label, config, suite, expected_sources, git_commit, existing=None, transfer=None):
    """One trial in one container. `existing` is a checkpoint path on the runs volume or a Hub id (legacy scoring)."""
    import torch
    from kev.experiment import execute_trial, source_hashes

    os.environ["KEV_GIT_COMMIT"] = git_commit
    if source_hashes() != expected_sources:
        raise RuntimeError("container received different kev/*.py than the launcher hashed")
    out = Path(RUNS_MOUNT) / study / f"{index:02d}-{label}"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite remote trial: {out}")
    print(f"[{label}] {torch.cuda.get_device_name(0)} torch {torch.__version__} config={json.dumps(config)}", flush=True)
    try:
        report, _ = execute_trial(config or {}, Path("/root") / suite, out, expected_sources, "cuda", existing, Path("/root") / transfer if transfer else None)
    finally:
        runs_volume.commit()
        hf_cache.commit()
    return {"label": label, "objective": report["objective"], "clean_acc": report["clean"]["acc"],
            "wall_seconds": report["wall_seconds"], "gates": report["gates"]["checks"]}


def pull_study(study):
    """Download a study directory from the runs volume into runs/<study> and rank it."""
    target = ROOT / "runs" / study
    if target.exists():
        raise FileExistsError(f"refusing to overwrite local study: {target}")
    target.parent.mkdir(exist_ok=True)
    # `modal volume get <vol> /<study> runs/` recreates runs/<study>/... locally, checkpoints included (gitignored)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", f"/{study}", str(target.parent)], check=True)
    subprocess.run([sys.executable, "-m", "kev.experiment", "--aggregate", "--out", str(target)], check=True, cwd=ROOT)
    return target


def launch(suite, plan_path, name, gpu, existing=(), transfer=None, budget=20.0, timeout=1800):
    from kev.experiment import load_plan

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("study name must be a simple unique identifier")
    if (ROOT / "runs" / name).exists():
        raise FileExistsError("choose a new study name; existing results are immutable")
    if not 60 <= timeout <= 1800 or not 0 < budget <= 25:
        raise ValueError("timeout must be 60..1800 seconds and study budget <= $25")
    trials = load_plan(ROOT / suite, ROOT / plan_path) if plan_path else []
    rates = {"H100": 3.95, "T4": .59}
    if gpu not in rates:
        raise ValueError("no verified cost bound for this GPU")
    upper = (rates[gpu] + 2 * .04730 + 48 * .008) * timeout / 3600 * (len(trials) + len(existing))
    if upper > budget:
        raise ValueError(f"timeout-based compute bound ${upper:.2f} exceeds budget ${budget:.2f}")
    print(f"Compute admission bound ${upper:.2f}; excludes image build, startup, and storage; no automatic retries.", flush=True)
    commit, sources = local_git_commit(), local_source_hashes()
    if subprocess.run(["git", "status", "--porcelain", "kev", "evals"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("warning: kev/ or evals/ has uncommitted changes; provenance records the last commit, not the working tree", flush=True)
    entries = [(None, p) for p in existing] + [(t, None) for t in trials]
    jobs = [(name, i, Path(ex).name if ex else f"trial-{i}", cfg or {}, suite, sources, commit, ex, transfer) for i, (cfg, ex) in enumerate(entries)]
    fn = run_trial.with_options(gpu=gpu, timeout=timeout, retries=0, max_containers=4)
    print(f"launching {len(jobs)} trial(s) on {gpu} for study {name}", flush=True)
    results = list(fn.starmap(jobs, return_exceptions=True))
    for job, result in zip(jobs, results):
        print(job[2], result if isinstance(result, Exception) else json.dumps(result), flush=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if len(failures) == len(results):
        raise SystemExit(f"all {len(results)} trial(s) failed; nothing to pull")
    target = pull_study(name)
    print(f"study pulled to {target}; {len(failures)} failure(s)", flush=True)
    if failures:
        raise SystemExit(1)


@app.local_entrypoint()
def study(suite: str, plan: str, name: str, gpu: str = GPU, existing: str = "", transfer: str = "", budget: float = 20.0, timeout: int = 1800):
    launch(suite, plan, name, gpu, [e for e in existing.split(",") if e], transfer or None, budget, timeout)


@app.local_entrypoint()
def smoke(gpu: str = GPU):
    launch("evals/smoke-v1", "experiments/smoke.json", "smoke", gpu)


@app.local_entrypoint()
def evaluate(run: str, suite: str, name: str, gpu: str = GPU, transfer: str = ""):
    """Score an existing checkpoint (Hub id, or a path under the runs volume) on a suite's development partition."""
    launch(suite, None, name, gpu, [run], transfer or None)
