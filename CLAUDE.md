# Repository instructions

This is `qumus-simlab`, the simulated Hall-bar laboratory. Use [AGENTS.md](AGENTS.md) for the shared
development instructions and [README.md](README.md) for how to run the sim lab.

- Same seed = same flakes: process parameters never change yield or layers (no hidden best recipe).
- A full search runs only on the Mac in practice (a cloud container needs ~4 h per chip); never run the
  full test suite in the same container as a running sim lab (memory).
- Default branch is `main`; open PRs against it.

## Qumus project map (shared by all Qumus repositories, 2026-11-12)

Owner: Jialei Chen (GitHub `papaya2003`), Sanfeng Wu lab, Princeton. Talk to Jialei in Chinese;
code, comments, commits and docs in English. Jialei has no software background: explain simply.

| Repository | Role | main (2026-11-12) |
| --- | --- | --- |
| `qumus-agent` | The brain: AI agents (OpenAI Agents SDK), web UI on 8765, History, budget, run store, contract client. Never contains laboratory code. | 1792a5e |
| `qumus-simlab` | Simulated Hall-bar lab as its own program: exfoliation 8781, stacking 8782, vdW reactor 8783. Never imports qumus-agent. | 80dfe3f |
| `qumus-lab` | Windows lab-PC software: drivers, SafetyGate, module services 8781/8782, debug page 8780 (never tunnel it). | 5f76aa1 |
| `lab-contract` | Contract 2 spec, schemas, `conformance`, `labkit` (stations) and `labclient` (the brain's only client). Private. | 917e41c |
| `qumus-v2` | Frozen phase-1 demo; main 0cc7826 must not change. | 0cc7826 |

- The brain talks to every station only over HTTP (contract 2; older services still speak v1 and are
  auto-detected). Same computer or another computer differs only by URL; no shared files or databases.
  `stations.yaml` in qumus-agent picks sim or real per module.
- Standing permission: code, commit, push branches, open PRs, merge to main, install deps, run tests in these
  repositories (not qumus-v2 main). Ask Jialei first before: driving real hardware; pushing anywhere else or
  changing repository visibility; deleting files outside the repositories and project notes; changing an
  already decided design decision; any paid model call (state the cap).
- Never commit keys, `.env`, lab data or photos, GDS files, target bundles, run databases or model weights.
- GitHub Actions has no minutes: run `ruff` and `pytest` locally and say so in the PR.
- Contract changes need Jialei's approval (2.x is additive only). Simulated operations must match Qumus v1
  exactly; when the sim cannot support a v1 operation, change the sim, not the flow.
- Full context, history, decisions and next steps live in the project notes
  (`notes/handoff-20261112/HANDOFF.md` and `CLAUDE.md` there), not in this repository.
