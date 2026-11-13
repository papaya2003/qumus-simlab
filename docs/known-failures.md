# Tests that still fail

Checked 2026-11-13 in the Linux cloud container (Python 3.13, SQLite 3.53) on branch
`claude/project-thread-q2jmv7`. Of the 26 failures inherited from the move (qumus-agent main
349404f = qumus-v2 0cc7826), these 7 remain. Their test doubles predate the current revisit
evidence rules (original-capture policy, `outline_mm`, `measurement_100x`, `saved_path`); the
production code was not changed to make them pass.

- `tests/test_rgb_revisit.py::test_exact20_source_has_original_capture_rotation_and_immutable_masks`
- `tests/test_rgb_revisit.py::test_original_capture_receipt_rejects_coherent_scale_and_outline_rewrite`
- `tests/test_rgb_revisit.py::test_unsupported20_topology_still_persists_distinct_original_masks`
- `tests/test_rgb_revisit.py::test_real_revisit_step_keeps20_geometry_when100_classification_arrives[True]`
- `tests/test_rgb_revisit.py::test_real_revisit_step_keeps20_geometry_when100_classification_arrives[False]`
- `tests/test_native_revisit_evidence.py::test_native_revisit_preserves_original_events_and_never_grants_admission[20]`
- `tests/test_native_revisit_evidence.py::test_native_revisit_preserves_original_events_and_never_grants_admission[100]`

Fixed on 2026-11-13: the geometry safety and motion preview cases (obstacles follow the raised
objective), the stamp discovery cases (`with_stamp_discovery` keeps the legacy focus tilt), the
stage kinematics snapshot count (identical snapshots share one file) and the report case (the
prepared chips now supply a compatible stack).

The simulation needs SQLite 3.38 or newer (`->` JSON operator); Ubuntu 22.04's system Python has
3.37 and refuses to start with a clear message.
