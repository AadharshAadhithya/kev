# Deploy Kev on Modal

Your own Kev endpoint, speaking TypeSafe's System One protocol, in three commands. It scales to zero when idle, so an
unused endpoint costs nothing.

```bash
pip install modal && modal setup                  # once: sign in to Modal in the browser
curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py
KEV_API_KEY=$(openssl rand -hex 24) modal deploy kev_serve.py
```

The deploy prints `https://<your-workspace>--kev-api.modal.run`. Keep the key: requests need
`Authorization: Bearer <key>`. Point any TypeSafe client at the URL:

```python
client = TypeSafeClient(api_key=KEV_API_KEY, base_url="https://<your-workspace>--kev-api.modal.run", model="kev-latest")
```

| Model | Set | GPU | First request after idle |
| --- | --- | --- | --- |
| Kev-0.8B | `KEV_MODEL=jaredpalmer/kev-0.8b` | L4 | ~35 s (77 s the very first time) |
| Kev-4B (default) | nothing | L4 | ~50 s |
| Kev-9B | `KEV_MODEL=jaredpalmer/kev-9b` | A100-80GB or H100 | ~45 s (about 2 min the very first time) |

`KEV_MIN_CONTAINERS=1` keeps it warm; `modal app stop kev` takes it down. With an agent, `npx skills add
jaredpalmer/kev@kev-deploy` and ask it to deploy Kev; it follows [SKILL.md](SKILL.md). `modal skills install` adds
Modal's own agent skill and docs alongside it, for anything beyond this file (GPUs, secrets, logs, billing).
