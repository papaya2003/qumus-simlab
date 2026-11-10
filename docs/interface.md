# Sim lab interface (contract v1 + extensions)

Agreed with the agent thread on 2026-11-10 (shared note
`notes/agent-brain-20261110/brain-simlab-interface.md`). Base rules: contract v1 = reactor
ADR 0001 wire format plus the Qumus rules marked [Q] (`notes/night-20261108/contract-v1-qumus.md`).

Every window (8781 exfoliation, 8782 stacking, 8783 reactor) serves all of this; the three
windows share one simulated table, so `/api/records` and `/api/sim/*` answer the same on
each. The agent reads them from stacking (8782).

## Contract 2 (next to v1)

Contract 2 (papaya2003/lab-contract `spec/contract-2.md`, decision 0008) is served next to the
v1 routes by `qumus_simlab/simlab/contract2.py`. The sim lab is **one station**: every port
(8781-8783 and the remote listeners 28781-28783) answers the same entry.

| Route | Meaning |
|---|---|
| `GET /api/station` | entry: services `exfoliation/`, `stacking/`, `reactor/`, `boot_id` |
| `GET /<window>/api/describe` | command specs (units, `deadline_s`, `lease`, `hazard`, ...), `lease.duration_s` |
| `GET /<window>/api/state` | flat state; devices and jobs under `readings` |
| `GET /<window>/api/events?since=&limit=` | events with `kind` |
| `POST`/`DELETE /<window>/api/lease` | take or renew (`renew_only: true`) / release |
| `POST /<window>/api/command` | `{command_id, action, params, deadline_s?}`; 202, duplicate, `id_reused` |
| `GET /<window>/api/commands/<id>` | command record (last 500 of this boot) |
| `GET /<window>/api/jobs/<run_id>` | jobs profile |
| `POST /<window>/api/stop` | any client, no lease |

Header `X-Lab-Client`; errors are `{"error": {code, message, retry}}`. The v1 routes below stay
until the brain has moved. The contract 2 `records` profile (brain-issued sample ids) is not
offered yet; records stay on the v1 feed. Check a running sim lab with lab-contract's
`python -m conformance --demo http://127.0.0.1:8782/`; the test
`tests/test_simlab_contract2.py` runs the same check when `LAB_CONTRACT=<lab-contract checkout>`.

## Ownership

- Sim lab owns: the simulated world, runs/steps of jobs, samples it creates or moves,
  reservations it makes, lab events, picture files.
- Agent owns: projects, plans, instructions, experience, agent events, budget.
- Both write: `sample`, `reservation`, and the project `state` the flows read. The agent
  sends its writes with `put_records`; the lab's come back through `/api/records`.
- No shared files. Each side has its own SQLite run store.

## `GET /api/records?since=<seq>&limit=<n>` [Q]

```json
{"changes": [{"seq": 17, "table": "sample", "key": "chip_e01", "row": {...}}],
 "next": 17, "head": 2031, "lab_id": "3f0c..."}
```

| table | key | row |
| --- | --- | --- |
| `sample` | id | `{id, kind, location, status, props, updated_at}` (RunStore.get_sample) |
| `reservation` | `<kind>:<key>` | `{kind, key, run_id, flow, note, created_at}` |
| `run` | run id | RunStore.get_run (incl. `meta`, `bench`) |
| `step` | `<run_id>:<seq>` | RunStore.get_steps item plus `run_id` |
| `event` | lab event id | `{id, ts, kind, run_id, agent, payload[, clock: {basis, domain}]}` |

- `row: null` = deleted / released. The row is the CURRENT row when the page is read
  (several changes of one key may all show the newest row).
- Ordered by `seq` (SQLite change-log triggers, commit order). Page size up to 2000.
- `lab_id` changes when the lab starts a new store: start again from `since=0`.
- Pictures: every string that is a path to one of the lab's picture files is replaced by
  `"qumus-image:<image_id>"`, anywhere in the row. Shapes and types never change.
- A job's run row is in the feed from the moment the job command is accepted.

## `GET /api/images/<image_id>` [Q] (+ `/meta`)

- `image_id` = the file's path relative to the lab's picture folder; it may contain `/`.
- Bytes with `Content-Type`; 400 malformed id, 404 missing.
- `/meta`: `{image_id, bytes, content_type, version, width, height}`.

## Command `put_records` [Q]

any_client, not motion, accepted while busy.

```json
{"action": "put_records",
 "project": [{"id": "prj_1", "name": "demo", "state": {...}}],
 "samples": [{"id": "flake_3", "kind": "flake", "location": "chip_002", "status": "found", "props": {...}}],
 "removed_samples": ["..."],
 "reservations": [{"kind": "slot", "key": "transfer_stage", "run_id": "run_1", "flow": "...", "note": ""},
                  {"kind": "slot", "key": "x", "released": true}],
 "instructions": [{"id": "ins_1", "ts": 0, "project_id": "prj_1", "source": "ui", "kind": "message", "text": "...", "status": "done", "note": null}]}
```

Rows are upserted exactly as given (last write wins); `"qumus-image:<id>"` strings become
the lab's own paths again. No lab events are emitted for them; the changed rows echo back
through the feed. `project` may be one object or a list.

**Placement guard (one exception to "as given").** The lab owns where a sample physically
is: its `location` and the props `pose_on_stage`, `corner_stage_xy`, `rotation_deg`. For a
sample the lab already has, each of these in the pushed row is checked against every value
that sample has held:

- equal to the lab's current value: nothing to do;
- a value the sample held EARLIER but not now (a stale copy, e.g. the brain's row from
  before a lab move or from a save point), or missing / `null`: the lab keeps its current value;
- a value the sample never had (the brain's own measurement, e.g. a brain-side rotation
  check): taken.

`"placement_from_brain": true` on a sample row takes its placement as given (e.g. a chip
carried back by hand to where it was before). The other fields of the row are always taken.
Picture paths get the same protection: a string in a pushed sample's props that is an
absolute path outside the lab's picture folders (the brain's local copy, sent without
turning it back into `"qumus-image:<id>"`) keeps the lab's own path at the same place in
the row.

The command result carries `"placement_kept": {sample_id: [field, ...]}` for what the lab
kept (picture paths as `props.<where>`).

## Jobs

- A whole-job command is marked `"job": true` in `describe().commands`; its params are the
  flow's own JSON schema plus `job: {project_id, requested_by}`.
- `202 {command_id, run_id, result: {run_id, flow}}`; follow the job through the feed (run /
  step / event rows) or `job_status {run_id}` (any_client) →
  `{run_id, flow, status, done, result, error, workflow, steps, module}`.
- Controls: `job_cancel`, `job_retry`, `job_reverify`, `job_resume`, `job_review_waiting`
  (its `failure_event_id` is a LAB event id) on the run's own window; `job_resume_all`,
  `complete_experiment` on stacking.

## [Q-sim] bench views (simulation only; a real station never has them)

- `GET /api/sim/layout` → `{bench, layout: {...bench, regions, positions}}`
- `GET /api/sim/bench_state` → `{devices, objects, twin, stamp, faults}`
- `GET /api/sim/preview/<camera_id>` → `{camera, image_id, v, timestamp, meta}` (`v` changes
  with every new frame; 404 unknown camera)
- `GET /api/sim/faults` → the fault menu; `POST /api/sim/faults {fault, params}` → inject
  (local only).
- `POST /api/sim/checkpoint {dir, label}` → `{ok, dir, lab_id, records_head}` (local only;
  409 while a job is open). `serve --resume <dir> --db <new folder>/qumus.db` starts from it
  with the same `lab_id`; the agent continues its feed from `records_head`.

## Same computer / another computer

- Default bind 127.0.0.1. `--host 0.0.0.0` (or a LAN address) for a direct connection.
- `--remote-ports`: loopback listeners 28781/28782/28783 where every request counts as
  remote: the end of an SSH tunnel. Local-only actions (`jog`, `set_switch`, `unlock`,
  `POST /api/sim/faults`) are refused there and from any non-loopback address.
