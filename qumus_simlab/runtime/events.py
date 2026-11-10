"""Event schema (CONTRACT between runtime/agents and the UI).

Every event is one row in the run store's `event` table. The UI never talks to
devices or agents directly: it tails this table by id (GET /api/events?after=N
or the /ws/events websocket). This also works across processes (job service,
agent process, UI server all write/read the same SQLite file).

Payload fields per kind are listed below. Producers may add fields; consumers
must ignore fields they don't know.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# kind -> payload fields (documentation; enforced loosely by tests)
EVENT_KINDS: dict[str, str] = {
    "microscope.revisit_checkpoint": "version, step, checkpoint_sha256 (original same-job commitment to resumable candidate evidence and entry optical anchor)",
    "microscope.rgb_recognition": "version, flake_id, source_chip_id, material, capture_timestamp, "
                                  "chip_rotation_deg, recognition_chain (original captured RGB, "
                                  "calibration, paired masks and material policy; not material accuracy)",
    # conversation
    "user.message": "text, instruction_id",
    "agent.message": "text  (agent = who spoke: orchestrator/advisor/eyes/hands/picker/designer)",
    "agent.delegate": "to, task  (orchestrator handed work to a subagent)",
    "agent.done": "summary  (a subagent returned to the orchestrator)",
    "agent.delegation_checked": "to, step, ok, status, facts[]  (code checked a returned "
                                "delegation against the run records; status = its plan step)",
    "agent.job_linked": "project_id, step_id, job_id, flow  (durable physical work association across delegations)",
    "agent.web": "tool, tool_use_id, input, phase (call/result), chars, excerpt  (WebSearch / "
                 "WebFetch, logged only)",
    "agent.interrupted": "reason",
    "agent.turn_end": "session_id, project_id, is_error, num_turns, cost_usd, reason  (SDK turn finished)",
    "agent.usage": "provider, gateway, session_id, call_id, request_id, token_measurement, input_tokens, output_tokens, cost_usd, cost_basis "
                   "(SDK-normalized response cost event; configured rates; request_usage preserves raw field presence)",
    "agent.request": "provider, gateway, session_id, call_id, request_id, model, api, attempt, text_chars, input_composition, "
                     "instructions_chars, conversation_chars, tools_chars, image_count, unit, "
                     "measurement, token_measurement, context_window_tokens, context_limit_source, "
                     "max_output_tokens, prefix_comparison (counts/categories only; one event per HTTP "
                     "send attempt; token usage follows separately)",
    "agent.request_usage": "provider, gateway, session_id, call_id, request_id, model, api, attempt, "
                           "input_tokens, output_tokens, total_tokens, provider_total_tokens, "
                           "cached_input_tokens, cache_write_tokens, reasoning_output_tokens, usage_status, measurement, unit, "
                           "context_window_tokens, context_limit_source, max_output_tokens "
                           "(provider report for exactly one send; unknowns are null; never a cost event)",
    "agent.request_failure": "provider, session_id, call_id, request_id, model, api, attempt, "
                             "failure_kind, exception_type, http_status, provider_request_id, provider_error_code "
                             "(sanitized failure of one attempt, before retries; no messages or cost)",
    "agent.error": "error, type, session_id, project_id  (a turn raised; retained after a later scoped recovery)",
    "user.dropped": "instruction_id, text  (Stop pressed before the message was delivered)",
    # Stop / Continue (agents/driver.py)
    "system.stopped": "instruction_id, dropped[]  (interrupted; every job pauses at a safe point)",
    "system.continued": "instruction_id, resumed[{run_id, flow, from_step, step, reason}], "
                        "waiting_human[{run_id, flow, step, reason}]",
    # tools (written by the PreToolUse / PostToolUse hooks or the scripted runtime)
    "tool.call": "tool, args, tool_use_id",
    "tool.result": "tool, tool_use_id, status, summary, data, images",
    "tool.denied": "tool, args, reason, rule",
    # plan / project
    "plan.updated": "project_id, revision, plan, note",
    "plan.step": "project_id, revision, step_id, status, plan  (progress, same revision)",
    # jobs and closed-loop steps (written by the job service / StepRunner)
    "job.submitted": "flow, params",
    "laboratory.origin": "version, kind, backend_id, submission_event_id "
                         "(immutable device-bundle provenance immediately after job submission; not a user label)",
    "job.workflow": "workflow, reason (submitted or checked replan; maps molecular ranges to original checkpoints)",
    "job.workflow_rejected": "stage, error (saved/current structural bindings disagree; original manifest retained)",
    "job.status": "status, message",
    "job.human_reviewed": "version, failure_event_id, failure_event_sha256, failed_step "
                          "(seq, name, event_id, event_sha256), reviewer, reason, source, "
                          "from_status, to_status, safe_energy_targets, scope "
                          "(operator report; only parks a held job, never executes or verifies a process)",
    "job.params_changed": "path, value, note  (resolved params changed at run time: one entry "
                          "at `path`, or path [] = all of them, e.g. a stamp changed)",
    # sample ledger (runtime/store.py upsert_sample; records/stamps.py)
    "sample.updated": "id, kind, location, status, props (material, film, stack, ... only: "
                      "store.SAMPLE_EVENT_PROPS; flakes also chip_id, labels, area_um2, "
                      "confidence, stage_xy, rel_corner_mm, images, residue)  (the ledger as "
                      "events, for the Lab page, the flake library and replays)",
    "sample.removed": "id, kind, reason  (a sample left the ledger: a new testbench was set up "
                      "in a run store kept from an earlier serve)",
    "stamp.film_replaced": "stamp_id, status (fresh), previous_status, what, by, via "
                           "(ui:lab | chat), instruction_id, ts  (a person put a new PC film "
                           "on a used or torn stamp)",
    "reactor.observation": "chip_id, job_id, timestamp, source, coverage, uniformity, "
                           "defect_fraction, quality_score, temperature_c, flow_sccm, images",
    "reactor.goal": "project_id, goal (chip_id, revision, field thresholds, source)",
    "reactor.result_accepted": "chip_id, evidence_job_id, goal, observation, accepted, scope",
    "step.started": "seq, name, expected",
    "step.verified": "seq, name, verdict, measurements, expected, attempt, images",
    "step.corrected": "seq, name, attempt, action",
    "step.finished": "seq, name, status, corrections",
    "microscope.scan_progress": "version, step, chip_id, attempt, phase, tiles_scanned, tiles_total, "
                                "blurred_tiles, verification=false (acquisition progress; "
                                "not an independent verification or new laboratory state)",
    "stack.release_footprint": "policy, stamp_id, load_ids, selection_sha256, witness_sha256, frame, receipts, focus_event, loaded, profile, noise_scale, footprint_captures, axis (one original conditional SIM observation)",
    "stack.release_approach": "phase, footprint, reference, entry_heat_commands, at (owned bounded entry heat before one monotone observed approach)",
    "stack.release_support": "footprint, reference, current, reference_receipts, current_receipts, trajectory, transports, support_context, background_bound_mm, surface_branch, result, actual_command_z_mm (conditional modeled support; opaque integrity unknown, never transfer proof)",
    "stack.control": "version, phase, state, started_at, observed_at, observations, reacquisitions, "
                     "reference_image/meta, relative_reference, motion, uncertainty_mm, drift_bound_mm_s, "
                     "placement_error/uncertainty/goal_mm, source; terminal lift intent/readback, "
                     "reference_transition, verification, outcome (local checkpoints; no completion claim)",
    "stack.pickup_macro_admission": "phase, profile, rule, from_readback_z_mm, requested_delta_mm, "
                                    "requested_target_mm, admitted_delta_mm, admitted_target_mm, minimum_z_mm, "
                                    "readback_tolerance_mm, global_z_interval_mm, process_reference_sha256, "
                                    "reference_sha256, admission_observation_count, admitted_at, capture_event "
                                    "(one consumed final process-budget prefix; never transfer proof)",
    "stack.source_view": "state, source_id, stamp_id, stamp_flake_ids, source_objective, previous_source_image, "
                         "original_source_image, current_source_image, source_polygon_px, source_calibration, "
                         "observed_at, photometry, fixed_goal_mm, counters_and_drift_preserved, previous_binding, "
                         "previous_optical_reference, original_source_reference "
                         "(same named full-source admission using an existing higher objective; original references retained)",
    "stack.stamp_phase_reference": "key, context, reference, image_sha256, inherited_stamp_uncertainty_mm, "
                                   "admission, clearance_z_mm, safe_z_mm, fronts, fixed_goal_history_and_counters_preserved "
                                   "(one immutable original-identity-attested precontact stamp view per substrate context)",
    "stack.alignment": "step, state, stamp_id, target_id, relative_xy_mm, error_mm, uncertainty_mm, "
                       "corrections, observed_at, image/source_image (local approach alignment; "
                       "bounded corrections at safe clearance, not experiment completion)",
    "stack.aperture_exposure": "image, sha256, captured_at, calibration, holder_xy_mm, holder_pose, "
                               "camera_world_mm, event_id, capture_event_ts, event_sha256 "
                               "(saved pixels and original capture/encoder binding)",
    "stack.aperture_alignment": "profile, step_name, aperture_shape, stamp_id, run_id, config, "
                                "baseline_holder_xy_mm, centered_holder_xy_mm, frames, initial_fit, "
                                "confirmation_fit, final_frame_event_id, wafer_focus_event_id, "
                                "wafer_focus_event_sha256, wafer_focus_event_ts, started_at, ended_at, status",
    "stack.first_contact_sampling": "phase, policy, reference_event (frozen finite approach policy)",
    "stack.design_contact_dwell": "phase, hold (Clock start, duration, completion and recipe source)",
    "stack.design_source_park": "phase, park (actual holder pose and completion before source inspection)",
    "stack.electrode_refind": "chip_id, matched, score, angle_deg, match_center_px, shift_um, corner_stage_xy_before/after, reference_image, image (v1 20x site refind)",
    "stack.design_pickup_coverage": "design_digest, stamp_id, source_id, covered, frame, units, "
                                    "hbn_outline_mm, gr_outline_mm, source_frame, stamp_frame, scope",
    "stack.device_design_result": "status, design_digest, run_id, observations, scope, reason, "
                                  "overlap_area_um2 (complete optical geometry; no electrical certification)",
    # devices and safety
    "device.observation_rejected": "device, command, reason, capture_started_at, capture_finished_at, "
                                   "frame_timestamp, clock_basis, clock_domain (current microscope "
                                   "exposure refused; nonfinite timestamps are null; no motion violation)",
    "device.command": "device, command, params",
    "lab.frame": "device, command (that changed it); changed parts of LabTwin.state(): arms, "
                 "stages, manipulator, exfoliator, microscope, objects, gone, geometry, "
                 "microscope_preview (immutable presentation only; runtime/twin.py)",
    "lab.evidence": "schema_version, assertion_id, subject, property, bench, value, units, frame, "
                    "source, observed_at, timebase, calibration_id/version, uncertainty, status, "
                    "unknown_reason, supersedes, conflicts_with (immutable laboratory assertions; "
                    "source kind is not sensing authority or motion permission)",
    "safety.violation": "device, command, params, reason",
    "notification": "level (info/warn/needs_human/danger), text",
    # experience book (spec 10; records/experience.py)
    "experience.proposed": "experience_id, topic, claim, sandbox",
    "experience.voted": "experience_id, run_id, verdict, note  (agent = advisor or code)",
    "experience.status": "experience_id, from, to, reason",
    "experience.taught": "experience_id, claim, instruction_id",
    "outcome.recorded": "sample_id, verdict, note, instruction_id, ts",
    # testbench
    "world.snapshot": "objects[], devices{}  (sim ground truth for the lab view)",
    "fault.injected": "fault, params",
}


class Event(BaseModel):
    id: int
    ts: float
    kind: str
    run_id: str | None = None
    agent: str | None = None
    payload: dict[str, Any] = {}
