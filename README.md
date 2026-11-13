# qumus-simlab

**Qumus v2.2 · stage 2 · simulated laboratory.** Driven by [qumus-agent](https://github.com/papaya2003/qumus-agent) (the brain) through [lab-contract](https://github.com/papaya2003/lab-contract); the real-lab counterpart is [qumus-lab](https://github.com/papaya2003/qumus-lab).

The Qumus **simulated laboratory**: the v1-linked Hall-bar testbench (chips, flakes,
stamps, microscope pictures, the whole exfoliation → search → stacking flow and the vdW
reactor) as an independent program. It knows nothing about the AI agent. The agent
system ([qumus-agent](https://github.com/papaya2003/qumus-agent)) reaches it only over
HTTP, through the same instrument service contract v1 as the real lab software
([qumus-lab](https://github.com/papaya2003/qumus-lab)) and the reactor.

> 中文摘要：这是独立的“模拟实验室”程序。它开三个窗口：剥离 8781、堆叠 8782、vdW reactor 8783，
> 规则和真实实验室（qumus-lab）一样都是 contract v1。agent（qumus-agent）只通过这三个窗口跟它说话，
> 自己有自己的数据库，不读这里的文件。默认只在本机监听；另一台电脑的 agent 可以走 SSH 隧道
> （`--remote-ports`）或直接连局域网地址（`--host`）。只有 Hall-bar 这一套实验台。

## Start

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"      # or: uv pip install -e ".[dev]"
.venv/bin/python -m qumus_simlab serve --db runs/simlab-$(date +%Y%m%d-%H%M)/qumus.db --seed 3 \
    [--target-bundle <folder with target.json + electrode-layout.gds>]
```

```
exfoliation  http://127.0.0.1:8781/api/describe
stacking     http://127.0.0.1:8782/api/describe
reactor      http://127.0.0.1:8783/api/describe
```

- `--db` must be a new path: the sim lab always starts from a fresh bench. Pictures go to
  `images/` next to it.
- `--seed` world seed (default 3, the known stable Hall-bar seed).
- `--target-bundle` the operator's imported electrode target (lab data: never commit it).
  Without it the built-in eight-electrode target is used.
- Free: no model, no network.
- Optional environment variables (sim lab only): `QUMUS_SIMLAB_LEASE_S=<s>` lengthens the
  contract lease (minimum and default 3 s) when a busy simulation answers heartbeats late;
  `QUMUS_SIMLAB_MEMWATCH=<GB>` writes `memwatch-*.txt` allocation reports next to `--db` once
  the process has held more than that (diagnosis only; it slows the lab).

### Same computer or another computer

| Agent runs on | Start the sim lab with | Agent's station URLs |
| --- | --- | --- |
| the same computer | (defaults) | `http://127.0.0.1:8781/` … `8783` |
| another computer, SSH tunnel (recommended) | `--remote-ports` | the tunnel end, forwarded to `127.0.0.1:28781` … `28783` |
| another computer, direct LAN | `--host 0.0.0.0` (or the LAN address) | `http://<this computer>:8781/` … |

The remote listeners (`--remote-ports`, 28781–28783) treat every request as coming from
another computer, so an SSH tunnel, which looks like loopback, still cannot call the
local-only actions (`jog`, raw `set_switch`, `unlock`, fault injection). Those stay on the
lab computer itself, as in qumus-lab.

### Save points (faster debugging)

`POST /api/sim/checkpoint {"dir": "<folder>", "label": "..."}` (local only; refused with 409
while a job is open) saves the run store and the whole simulated world. Start again from it
with

```sh
.venv/bin/python -m qumus_simlab serve --resume <folder> --db runs/<new folder>/qumus.db
```

The seed and target bundle come from the save point. The resumed lab keeps the same
`lab_id` and record sequence, and the earlier workspace's pictures keep their image ids
(keep that workspace). The agent saves its own store at the same moment and resumes it
with its record cursor. A save point refuses to load when a file that decides the search
changed (`--accept-changed <file>` only for a change proven not to alter it).

## The three windows

| Window | Port | Devices | Whole-job actions |
| --- | --- | --- | --- |
| `exfoliation` | 8781 | arm_left, exfoliator (tape), exfoliation heater/fan/vacuum, overview camera | `exfoliate_chips` |
| `stacking` | 8782 | arm_right, transfer stage (+vacuum, fan), microscope, stamp heater, stamp manipulator (+vacuum), overview camera | `place_chip_on_transfer_stage`, `search_chip`, `revisit_chip`, `stack_device`, `cool_experiment_stages`, `inspect_bench`, `prepare_reactor_pair`, `prepare_material_metal`, `job_resume_all`, `complete_experiment` |
| `reactor` | 8783 | arm_reactor, reactor slot + insert stages, reactor heater, gas flow, reactor microscope, overview camera | `assess_reactor_inputs`, `load_reactor_chip`, `position_reactor_slot`, `reactor_growth`, `reactor_reaction`, `reactor_material_metal`, `verify_reactor_state`, `hold_reactor_observation`, `observe_reactor`, `inspect_reactor_product`, `shutdown_reactor`, `unload_reactor_chip` |

All three share one simulated table, one run store and one job queue. Every window also has
the level-1/2 actions of the contract (`read_all`, `home_query`, `capture`, `go_to_point`,
`heat`, `heat_off`, local-only `jog`/`set_switch`) and the job controls (`job_cancel`,
`job_retry`, `job_reverify`, `job_resume`, `job_review_waiting`, `job_status`). Whole-job
actions are marked `"job": true` in `/api/describe`.

## Interface

Contract v1 (ADR 0001 wire format, Qumus rules): `GET /api/describe`, `GET /api/state`,
`POST /api/command`, `POST /api/stop`, `POST /api/heartbeat|renew|release`,
`GET /api/events?since=`, `POST /api/unlock` (local only). Header `X-Qumus-Client`.
Lease 3 s; an unknown motion outcome locks the window until a person unlocks it.

Extensions agreed with the agent (see [docs/interface.md](docs/interface.md)):

- `[Q]` `GET /api/images/<image_id>` (+ `/meta`): picture bytes.
- `[Q]` `GET /api/records?since=<seq>`: the lab's sample / reservation / run / step / event
  changes in order, with `lab_id`. Picture paths appear as `"qumus-image:<image_id>"`.
- `[Q]` command `put_records`: the agent's own records the flows read (project state,
  samples it edited, reservations, UI instructions).
- `[Q-sim]` `GET /api/sim/layout`, `/api/sim/bench_state`, `/api/sim/preview/<camera>`,
  `GET|POST /api/sim/faults`: what the agent's page draws for the simulation only.

## Tests

```sh
.venv/bin/python -m pytest -n 4 --dist loadfile      # pytest-xdist; slow end-to-end: -m slow
```

## Where it came from

Moved out of qumus-agent main `349404f` (itself qumus-v2 `0cc7826`) on 2026-11-10:
`testbench/`, `devices/sim`, `flows/`, `safety/`, `perception/`, the lab half of
`runtime/` and `records/`, and `simlab/` (the windows). Only the Hall-bar bench
(`hallbar-linked` profile) is served; older benches, scenarios and agent code were not
carried over. Package `qumus_simlab`.

Lab-private: Sanfeng Wu lab, Princeton. Never commit API keys, `.env`, lab data, GDS files
or target bundles.

## Windows

Runs on Windows 10/11 with Python 3.11+ the same way, with `.venv\Scripts\python.exe`
in place of `.venv/bin/python`. Every text file is read and written as UTF-8
(`tests/test_windows_portability.py`, ruff PLW1514), because the Windows default is
the locale code page (GBK). `QUMUS_SIMLAB_MEMWATCH` is POSIX-only and ignored on
Windows. qumus-agent's `scripts/check_platform.py` checks the brain and this lab
together on a new computer.
