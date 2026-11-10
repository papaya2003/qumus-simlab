# Tests that fail before and after the move

These fail identically on qumus-agent main 349404f (= qumus-v2 0cc7826) in the Linux cloud
container (checked 2026-11-10). They are not caused by the move.

- `tests/test_geometry_safety.py::test_holder_sweep_rejects_objective_with_safe_endpoints`
- `tests/test_geometry_safety.py::test_stage_sweep_rejects_objective_when_sample_is_tall`
- `tests/test_geometry_safety.py::test_focus_stroke_checks_both_mechanical_clearances[sample]`
- `tests/test_geometry_safety.py::test_focus_stroke_checks_both_mechanical_clearances[stamp]`
- `tests/test_stamp_discovery.py::test_quieter_stage_stays_inside_the_declared_stage_bound`
- `tests/test_stamp_discovery.py::test_steadier_stamp_stays_inside_the_declared_drift_bound[lateral_drift_mm_s-steady0-faster0]`
- `tests/test_stamp_discovery.py::test_steadier_stamp_stays_inside_the_declared_drift_bound[approach_drift_mm_per_mm-steady1-faster1]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[curvature]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[tau]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[drift]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[tilt]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[apex_prior]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[noise]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[stage_noise]`
- `tests/test_stamp_discovery.py::test_declared_policy_cannot_mask_changed_actual_generators[focus_speed]`
- `tests/test_motion_preview_evidence.py::test_swept_clearance_uses_actual_translated_integrated_workcell`
- `tests/test_stage_kinematics.py::test_polling_at_fixed_clock_preserves_physics_and_both_rng_streams`
- `tests/test_rgb_revisit.py::test_exact20_source_has_original_capture_rotation_and_immutable_masks`
- `tests/test_rgb_revisit.py::test_original_capture_receipt_rejects_coherent_scale_and_outline_rewrite`
- `tests/test_rgb_revisit.py::test_unsupported20_topology_still_persists_distinct_original_masks`
- `tests/test_rgb_revisit.py::test_real_revisit_step_keeps20_geometry_when100_classification_arrives[True]`
- `tests/test_rgb_revisit.py::test_real_revisit_step_keeps20_geometry_when100_classification_arrives[False]`
- `tests/test_rgb_revisit.py::test_latest_original_rotation_receipt_is_used_and_must_precede_capture`
- `tests/test_native_revisit_evidence.py::test_native_revisit_preserves_original_events_and_never_grants_admission[20]`
- `tests/test_native_revisit_evidence.py::test_native_revisit_preserves_original_events_and_never_grants_admission[100]`
- `tests/test_report.py::test_report_lists_unverifiable_steps`
