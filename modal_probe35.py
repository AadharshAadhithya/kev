"""Zero-shot base-model probe for Qwen3.5 bases on the frozen transfer suite, in its own image (transformers >= 5 is
required for the qwen3_5 architecture; the main Modal app pins the repo's uv.lock, which is on transformers 4.x).

    uv run modal run modal_probe35.py --bases Qwen/Qwen3.5-4B-Base,Qwen/Qwen3.5-9B-Base

Writes benchmark-compatible rows/report to the kev-runs volume under /probes/<name> and pulls them to runs/probes/.
Same readout as scripts/base_mmlu_probe.py: next-token letter logits over the rendered options, no training.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("kev-probe35")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers>=5.17,<6", "peft>=0.18", "accelerate", "datasets", "numpy", "scikit-learn",
                    "huggingface_hub", "pydantic", "flash-linear-attention", "triton")
    .add_local_dir(ROOT / "kev", "/root/kev")
    .add_local_dir(ROOT / "evals", "/root/evals")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret")]


@app.function(image=image, gpu=os.environ.get("KEV_PROBE_GPU", "H100"), cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache}, secrets=secrets)
def probe(base, suite, name, tasks="all", prompt="plain", split="development", revision=None, adapter=None):
    import os
    out = Path("/runs/probes") / name
    if out.exists():
        raise FileExistsError(f"probe {name} exists")
    try:
        subprocess.run([sys.executable, "/root/scripts/base_mmlu_probe.py", "--base", base, "--suite", f"/root/{suite}", "--tasks", tasks, "--device", "cuda", "--out", str(out), "--prompt", prompt, "--split", split] + (["--revision", revision] if revision else []) + (["--adapter", adapter] if adapter else []),
                       check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        runs_volume.commit(); hf_cache.commit()
    return json.loads((out / "report.json").read_text())["clean"]


@app.local_entrypoint()
def main(bases: str, suite: str = "evals/v4/transfer-v4", tasks: str = "all", prompt: str = "plain", split: str = "development", revision: str = "", adapter: str = "", tag: str = ""):
    jobs = []
    for base in bases.split(","):
        name = base.split("/")[-1].lower().replace(".", "") + ("-semif" if prompt == "semif" else "-base") + (f"-{tag}" if tag else "") + "-" + suite.split("/")[-1] + ("" if split == "development" else f"-{split}")
        if (ROOT / "runs/probes" / name).exists():
            print(f"skip {name}: exists locally"); continue
        jobs.append((base, suite, name, tasks, prompt, split, revision or None, adapter or None))
    for (base, _, name, *_), result in zip(jobs, probe.starmap(jobs, return_exceptions=True)):
        if isinstance(result, Exception):
            print(f"{name}: FAILED {type(result).__name__}: {str(result)[:200]}"); continue
        target = ROOT / "runs/probes" / name; target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", f"/probes/{name}", str(target.parent)], check=True)
        print(f"{name}: acc {result['acc']:.3f} brier {result['brier']:.3f} conf-err {result['confident_error_rate']:.3f}")


@app.function(image=image, gpu=os.environ.get("KEV_PROBE_GPU", "H100"), cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache}, secrets=secrets)
def bench(run, suite, name):
    """kev.benchmark for a Hub checkpoint on any local suite directory (mounted at run time), written to /runs/bench/<name>."""
    import os
    out = Path("/runs/bench") / name
    if out.exists(): raise FileExistsError(f"bench {name} exists")
    try:
        subprocess.run([sys.executable, "-m", "kev.benchmark", "--run", run, "--suite", f"/root/{suite}", "--out", str(out), "--device", "cuda"], check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        runs_volume.commit(); hf_cache.commit()
    return json.loads((out / "report.json").read_text())["clean"]


@app.local_entrypoint()
def benchmarks(jobs: str):
    """jobs: comma-separated run@suite@name triples."""
    triples = [j.split("@") for j in jobs.split(",")]
    for (run, suite, name), result in zip(triples, bench.starmap(triples, return_exceptions=True)):
        if isinstance(result, Exception): print(f"{name}: FAILED {type(result).__name__}: {str(result)[:300]}"); continue
        target = ROOT / "runs" / name; subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", f"/bench/{name}", str(target.parent)], check=True)
        print(f"{name}: acc {result['acc']:.3f} brier {result['brier']:.3f}")
