# Development instructions (qumus-simlab)

- Chat with Jialei in Chinese; code, comments and commits in English.
- This repository is the simulated laboratory only. It must never import or depend on
  qumus-agent. Everything the agent needs goes through the HTTP interface in
  [docs/interface.md](docs/interface.md); change that interface only together with the
  agent side.
- Only the Hall-bar bench (`hallbar-linked`). Do not bring back older benches or scenarios.
- Never commit API keys, `.env`, lab data, GDS files or target bundles.
- Before pushing: `.venv/bin/ruff check qumus_simlab tests` and
  `.venv/bin/python -m pytest -n 4 --dist loadfile`. Known failures inherited from
  qumus-v2 0cc7826 (they fail there the same way in a Linux container) are listed in
  [docs/known-failures.md](docs/known-failures.md).
