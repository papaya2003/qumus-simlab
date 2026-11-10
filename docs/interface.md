# Sim lab interface (contract v1 + extensions)

Agreed with the agent thread on 2026-11-10 (shared note
`notes/agent-brain-20261110/brain-simlab-interface.md`). Base rules: contract v1 = reactor
ADR 0001 wire format plus the Qumus rules marked [Q] (`notes/night-20261108/contract-v1-qumus.md`).

Every window (8781 exfoliation, 8782 stacking, 8783 reactor) serves all of this; the three
windows share one simulated table, so `/api/records` and `/api/sim/*` answer the same on
each. The agent reads them from stacking (8782).

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
