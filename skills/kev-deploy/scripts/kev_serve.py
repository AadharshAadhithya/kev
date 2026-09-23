"""Your own Kev endpoint on Modal: one file, one command. Speaks TypeSafe's System One protocol (POST /v1/systemone,
GET /v1/models), so existing Jev / TypeSafe clients only change their base URL.

    pip install modal && modal setup                                   # once: sign in to Modal (opens a browser)
    modal deploy kev_serve.py                                          # Kev-4B on an L4 -> https://<workspace>--kev-api.modal.run
    KEV_MODEL=jaredpalmer/kev-9b modal deploy kev_serve.py             # another model; the GPU follows the model
    KEV_API_KEY=$(openssl rand -hex 24) modal deploy kev_serve.py      # require Authorization: Bearer <key> (recommended)
    curl -L --max-time 900 https://<workspace>--kev-api.modal.run/v1/models -H "authorization: Bearer $KEV_API_KEY"   # wait for the cold start
    modal app stop kev                                                 # take it down

Settings are read when you deploy: KEV_MODEL (any Kev checkpoint on the Hugging Face Hub, `repo` or `repo@revision`;
default jaredpalmer/kev-4b), KEV_GPU (override the GPU), KEV_API_KEY (bearer auth; without it the URL is the only secret),
HF_TOKEN (for a private checkpoint; if it is set in your shell it is uploaded as a Modal secret, so unset it for public
checkpoints), KEV_MIN_CONTAINERS (1 keeps one container warm; default 0 scales to zero after 5 idle
minutes), KEV_APP_NAME (default "kev"; one app per endpoint). The image installs the kev package at KEV_REF, the code the
released checkpoints were measured with; weights and compiled kernels are cached on the `kev-hf-cache` volume, so only the
first cold start downloads them. A request that waits longer than 150 s for a cold start gets an HTTP 303 to a result URL
(Modal's web limit): follow redirects (`curl -L`) or warm the endpoint first.
"""
import os
import time

import modal

KEV_REF = "557598fced1dada75dfbf36ed144dce309ac6ceb"   # github.com/jaredpalmer/kev commit whose kev package this endpoint runs
# GPU preference lists (Modal takes the first with capacity). 9B: the fp32 LoRA merge needs 36 GB; 27B: 54 GB of bf16 weights.
GPU_FOR = {"kev-0.8b": ["L4", "A10G", "L40S"], "kev-4b": ["L4", "A10G", "L40S"], "kev-9b": ["A100-80GB", "H100"], "kev-27b": ["H100", "H200"]}

# Deploy-time settings travel in the image env, so the container evaluates this file with the same values.
SETTINGS = {"KEV_MODEL": "jaredpalmer/kev-4b", "KEV_APP_NAME": "kev", "KEV_MIN_CONTAINERS": "0", "KEV_GPU": ""}
SETTINGS = {k: os.environ.get(k, v) for k, v in SETTINGS.items()}
MODEL = SETTINGS["KEV_MODEL"]
NAME = MODEL.split("@")[0].split("/")[-1]                             # "kev-4b-support" (a fine-tune) gets kev-4b's GPUs
GPU = SETTINGS["KEV_GPU"] or next((gpus for size, gpus in sorted(GPU_FOR.items(), key=lambda kv: -len(kv[0])) if NAME.startswith(size)), ["H100", "H200"])
# Secret values never go into the image: they ride in a Modal secret (present in the container's env, so this stays equal there).
SECRET = {k: os.environ[k] for k in ("KEV_API_KEY", "HF_TOKEN") if os.environ.get(k)}

app = modal.App(SETTINGS["KEV_APP_NAME"])
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(f"kev[serve] @ git+https://github.com/jaredpalmer/kev.git@{KEV_REF}")
    .uv_pip_install("flash-linear-attention", "triton>=3.7.1")   # Gated DeltaNet kernels for the Qwen3.5 backbones (fla needs triton >= 3.7.1 on Hopper)
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "TRITON_CACHE_DIR": "/hf/triton-cache", **SETTINGS})
)
cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
WARMUP = {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "model": "kev-latest",
          "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                       "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges, payments"}},
                        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"}}}


@app.cls(image=image, gpu=GPU, cpu=2, memory=(16384, 131072), volumes={"/hf": cache}, secrets=[modal.Secret.from_dict(SECRET)] if SECRET else [],
         min_containers=int(SETTINGS["KEV_MIN_CONTAINERS"]), scaledown_window=300, timeout=600, startup_timeout=1200)
@modal.concurrent(max_inputs=8)
class Kev:
    @modal.enter()
    def load(self):
        import torch
        from kev.api import SystemOneRequest
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.serve import Server, app as api
        started = time.time()
        ck = Checkpoint(MODEL)                                             # downloads from the Hub on the first cold start
        tok, model = ck.load("cuda", LoadOptions(dtype=torch.bfloat16))    # the checkpoint's fitted temperature is applied automatically
        api.state.server = Server(ck, tok, model, "cuda")
        api.state.server.answer(SystemOneRequest.model_validate(WARMUP))   # compile the DeltaNet kernels now, not on the first request
        cache.commit()
        print(f"serving {MODEL} on {torch.cuda.get_device_name(0)} (temperature {model.head.temperature:.2f}, "
              f"auth {'on' if os.environ.get('KEV_API_KEY') else 'off'}), ready in {time.time() - started:.0f}s", flush=True)
        self.api = api

    @modal.asgi_app(label=f"{SETTINGS['KEV_APP_NAME']}-api")
    def web(self):
        return self.api
