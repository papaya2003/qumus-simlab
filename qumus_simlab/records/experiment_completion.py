"""Existing project obligations, shared by completion admission and final commit."""

from qumus_simlab.records.plan_completion import outstanding_physical_jobs
from qumus_simlab.records.reactor_requirements import acceptance_errors, requirements_record


def requires_stage_cooling(store, project_id):
    """A task label cannot erase established experiment or heated-process history."""
    project = store.get_project(project_id) or {}
    heated_processes = {"exfoliate_chips", "stack_device", "prepare_material_metal", "prepare_reactor_pair",
                        "reactor_material_metal", "reactor_reaction"}
    return (project.get("state", {}).get("task_kind") == "experiment"
            or bool(requirements_record(store, project_id))
            or any(r["project_id"] == project_id and r["flow"] in heated_processes for r in store.list_runs()))


def completion_error(store, project_id):
    project = store.get_project(project_id) or {}
    if not project:
        return {"status": "error", "summary": "A known project is required before completing an experiment."}
    unfinished = [s["id"] for s in project.get("plan", []) if s.get("owner") in ("hands", "reactor")
                  and s.get("status") not in ("done", "skipped")]
    if unfinished:
        return {"status": "error", "summary": f"Physical plan steps {unfinished} are still incomplete; "
                "verify their execution and declared outcomes before completing the project."}
    if outstanding := outstanding_physical_jobs(store, project_id):
        jobs = ", ".join(f"{run['id']} ({run['status']})" for run in outstanding)
        return {"status": "error", "summary": "Physical jobs remain unresolved in this project: " + jobs
                + ". Recover or safely cancel them; another successful job cannot erase them."}
    active = [r["id"] for r in store.list_runs() if r["project_id"] == project_id
              and r["status"] in ("queued", "running") and r["flow"] != "cool_experiment_stages"]
    if active:
        return {"status": "error", "summary": f"Jobs {active} are still queued or running; experiment completion must wait."}
    if errors := acceptance_errors(store, project_id):
        return {"status": "error", "summary": "; ".join(errors), "data": {"requirement_errors": errors}}
    return None
