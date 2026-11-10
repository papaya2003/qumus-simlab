"""Arm moves an object between two named places, verified by the overview camera.

This is the general form of the first worked example (spec 4.1, 13), "arm
puts a chip on the transfer stage -> camera verifies -> correct". v1 had one
atom per pair of places (place chip on exfoliation stage, store chip, place
carrier, store carrier, place chip on transfer stage, store search chip) and
none of them looked at the result; here every move is measured and re-done if
it is off.

The arm works in its own frame (bench.yaml); what the step reports
(expected, measured pose, the ledger's pose) is in the world frame, so the UI
and the other flows see one coordinate system.
"""

from __future__ import annotations

import math
from typing import Any

from qumus_simlab.core.errors import PreconditionFailed
from qumus_simlab.core.types import Pose, angle_diff
from qumus_simlab.flows.bench import arm_positions, is_slot, slots, to_world, within_reach
from qumus_simlab.flows.context import FlowContext
from qumus_simlab.flows.library.primitives import (
    lift,
    look,
    pick_and_place,
    place_held,
    require_sample,
)
from qumus_simlab.flows.step import Correction, Observation, ProbeResult, Step, Verdict

#: how close (mm) a detection must be to a place to count as "at" it
NEAR_MM = 3.0
#: camera frames per check; an object counts when seen in most of them
VIEWS = 3
#: frames of the second look when an expected object was not seen
RECHECK_VIEWS = 5
#: two placement errors this close (mm, or half the error's size if larger) are
#: one systematic error
SAME_OFFSET_MM = 0.5


class MoveObject(Step):
    """Move sample `sample_id` (a chip or carrier) from where the ledger says it
    is (or `source`) to `dest`, with `arm`, checked by `camera`.

    `release_switch` is a stage vacuum that holds the object at the source; it
    is switched off before the pick. Tolerances come from
    config/verification.yaml[cfg_key].

    `fallback` = (board, holds): when `dest` is a slot that turns out to be
    taken at run time (another job filled it since this one was planned), the
    step moves the object to the next free slot of that board instead, books
    it, and writes the new slot into the run's params at `param_path` (a list
    of keys / indexes), so a resume builds the same steps.
    """

    def __init__(self, sample_id: str, kind: str, *, arm: str, camera: str, dest: str,
                 cfg_key: str, name: str | None = None, source: str | None = None,
                 release_switch: str | None = None, status: str | None = None,
                 fallback: tuple[str, str] | None = None,
                 param_path: list[Any] | None = None) -> None:
        self.sample_id, self.kind, self.arm, self.camera = sample_id, kind, arm, camera
        self.dest, self.cfg_key, self.source = dest, cfg_key, source
        self.release_switch, self.new_status = release_switch, status
        self.fallback, self.param_path = fallback, param_path
        self.name = name or f"move_{kind}_to_{dest}"
        self.devices = tuple(d for d in (arm, camera, release_switch) if d)
        self._src: Pose | None = None
        self._holding = False  # resumed after a crash with the object on the nozzle
        self._offsets: list[tuple[float, float]] = []  # placement errors seen this run
        self._corrections = 0

    # -- helpers ----------------------------------------------------------

    def _cfg(self, ctx: FlowContext) -> dict[str, Any]:
        return ctx.verification[self.cfg_key]

    def _places(self, ctx: FlowContext) -> dict[str, Pose]:
        return arm_positions(ctx.bench, self.arm)

    def _target(self, ctx: FlowContext) -> Pose:
        return self._places(ctx)[self.dest]

    def _source_name(self, ctx: FlowContext) -> str:
        if self.source:
            return self.source
        s = ctx.store.get_sample(self.sample_id) or {}
        return s.get("location") or ""

    def _source(self, ctx: FlowContext) -> Pose:
        if self._src is None:
            self._src = self._places(ctx).get(self._source_name(ctx), self._target(ctx))
        return self._src

    def _look(self, ctx: FlowContext, region: str, images: list[str] | None = None,
              label: str | None = "", expect: bool = False) -> list:
        """`expect`: the object should be there; an empty look is checked again."""
        return look(ctx, self.camera, region, self.arm,
                    label=self.kind if label == "" else label, images=images, views=VIEWS,
                    recheck_empty=RECHECK_VIEWS if expect else 0)

    def _world(self, ctx: FlowContext, pose: Pose) -> dict[str, float]:
        return to_world(ctx.bench, self.arm, pose).model_dump()

    def _strays(self, ctx: FlowContext, images: list[str] | None = None) -> list:
        """Objects of this kind lying away from every named place (dropped on
        the way) that the arm can reach."""
        places = list(self._places(ctx).values())
        lim = ctx.gate.limits.arms[self.arm]
        return [d for d in self._look(ctx, "workspace", images)
                if all(d.pose.xy_distance(p) > NEAR_MM for p in places)
                and lim.x[0] <= d.pose.x <= lim.x[1] and lim.y[0] <= d.pose.y <= lim.y[1]
                and within_reach(ctx.bench, self.arm, d.pose)]

    def _failed_placement_pose(self, ctx: FlowContext) -> Pose | None:
        """Only this job's last failed, observed placement can admit a re-place."""
        events, after = [], 0
        while batch := ctx.store.events_after(after, limit=500, run_id=ctx.run_id):
            events.extend(batch)
            after = batch[-1].id
        finishes = [e for e in events if e.kind == "step.finished"]
        if not finishes:
            return None
        failed = finishes[-1]
        if failed.payload.get("name") != self.name or failed.payload.get("status") != "failed":
            return None
        starts = [e for e in events if e.kind == "step.started" and e.id < failed.id
                  and e.payload.get("name") == self.name and e.payload.get("seq") == failed.payload.get("seq")]
        if not starts:
            return None
        views = [e for e in events if starts[-1].id < e.id < failed.id and e.kind == "step.verified"
                 and e.payload.get("name") == self.name]
        if not views or views[-1].payload.get("verdict") != "correctable":
            return None
        observation = views[-1].payload
        measured = observation.get("measurements", {})
        sample = ctx.store.get_sample(self.sample_id) or {}
        props = sample.get("props", {})
        if (observation.get("expected") != self.expected(ctx) or measured.get("found_at") != self.dest
                or props.get("location_uncertain") or props.get("pose") != measured.get("pose")
                or props.get("last_seen") != {"at": self.dest, "pose": measured.get("pose"),
                                               "offset_mm": measured.get("offset_mm")}):
            return None
        try:
            pose = Pose(**measured["arm_pose"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in pose.model_dump().values()) or self._world(ctx, pose) != measured.get("pose"):
            return None
        cfg, target = self._cfg(ctx), self._target(ctx)
        if (pose.xy_distance(target) >= cfg["correctable_within_mm"]
                or pose.xy_distance(target) <= cfg["tolerance_mm"]
                and abs(angle_diff(pose.rotation, target.rotation)) <= cfg["tolerance_deg"]):
            return None
        return pose

    def _interrupted_place(self, ctx: FlowContext, src_name: str, at_dest: list,
                           failed_pose: Pose | None = None) -> Pose | None:
        """A run killed between release and check leaves the object at `dest`,
        off target, and nothing at the source. Then the one object of this
        kind at `dest` is this sample (no other sample is booked there) and
        the step re-places it from where it lies."""
        if len(at_dest) != 1 or at_dest[0].label != self.kind:
            return None
        if any(x["location"] == self.dest and x["id"] != self.sample_id
               for x in ctx.store.list_samples()):
            return None
        if failed_pose is None and self._look(ctx, src_name, expect=True):
            return None
        p = at_dest[0].pose
        if failed_pose is not None and (not all(math.isfinite(v) for v in p.model_dump().values())
                                        or p.xy_distance(failed_pose) > NEAR_MM):
            return None
        if p.xy_distance(self._target(ctx)) >= self._cfg(ctx)["correctable_within_mm"]:
            return None
        return p

    def _on_nozzle(self, ctx: FlowContext, src_name: str) -> bool:
        """Vacuum on and the object is neither at the source nor at the
        destination: it is on the nozzle (a run killed mid-move)."""
        return (ctx.gate.arm(self.arm).vacuum_on() and not self._look(ctx, self.dest)
                and not self._look(ctx, src_name))

    # -- Step contract ----------------------------------------------------

    def preconditions(self, ctx: FlowContext) -> None:
        cfg = self._cfg(ctx)
        self.max_corrections = cfg.get("max_corrections", self.max_corrections)
        require_sample(ctx, self.sample_id, self.kind)
        src_name = self._source_name(ctx)
        places = self._places(ctx)
        if src_name not in places:
            raise PreconditionFailed(f"{self.sample_id} is at {src_name!r}, which {self.arm} "
                                     f"cannot reach (it serves: "
                                     f"{', '.join(ctx.bench[self.arm].get('reach', []))})")
        if self.dest not in places:
            raise PreconditionFailed(f"{self.arm} cannot reach {self.dest!r} (it serves: "
                                     f"{', '.join(ctx.bench[self.arm].get('reach', []))})")
        failed_pose = self._failed_placement_pose(ctx) if src_name == self.dest else None
        if src_name == self.dest and failed_pose is None:
            raise PreconditionFailed(f"{self.sample_id} is already at {self.dest}")
        self._holding = False
        self._offsets = []
        self._corrections = 0
        if ctx.gate.arm(self.arm).vacuum_on():
            if not self._on_nozzle(ctx, src_name):
                raise PreconditionFailed(f"{self.arm} vacuum is on (holding something?)")
            self._holding = True
        if "transfer_stage" in (src_name, self.dest):
            load = Pose(**ctx.bench["transfer_stage"]["load"])
            if ctx.gate.stage("transfer_stage").get_pose().xy_distance(load) > 0.05:
                raise PreconditionFailed("transfer stage is not at its load position")
        at_dest = self._look(ctx, self.dest, label=None)
        if at_dest and not self._holding:
            placed = self._interrupted_place(ctx, src_name, at_dest, failed_pose)
            if placed is not None:
                self._src = placed
                ctx.store.emit("notification", {
                    "level": "warn", "text": f"{self.sample_id} is already at {self.dest} but "
                    f"not where it should be (a place cut short?): placing it again"},
                    run_id=ctx.run_id)
                return
        if src_name == self.dest:
            raise PreconditionFailed(f"{self.sample_id} same-job re-placement lacks one current unambiguous public pose")
        if at_dest:
            self._retarget(ctx, at_dest)
        if self._holding:
            return
        seen = [d for d in self._look(ctx, src_name, expect=True)
                if d.pose.xy_distance(places[src_name]) < cfg["correctable_within_mm"]]
        if not seen:
            raise PreconditionFailed(f"no {self.kind} seen at {src_name} "
                                     f"(ledger says {self.sample_id} is there)")
        # Pick where the camera sees it, not where the ledger thinks it is.
        self._src = seen[0].pose

    def _retarget(self, ctx: FlowContext, at_dest: list) -> None:
        """`dest` is taken: move to another free slot of the fallback board, or
        stop with a precondition that says what sits there."""
        there = ", ".join(sorted({d.label or "object" for d in at_dest}))
        if self.fallback is None:
            raise PreconditionFailed(f"{self.dest} is not empty ({there} there)")
        board, holds = self.fallback
        taken = ctx.store.occupied_slots(exclude_run=ctx.run_id) | {self.dest}
        places = self._places(ctx)
        for slot in slots(ctx.bench, board, holds):
            if slot in taken or slot not in places:
                continue
            if self._look(ctx, slot, label=None):
                continue  # booked by nobody, but something lies there
            old, self.dest = self.dest, slot
            ctx.store.release_reservations(ctx.run_id, "slot", [old])
            ctx.store.reserve("slot", slot, ctx.run_id, note=f"{self.sample_id} ({self.name})")
            if self.param_path:
                ctx.store.update_run_params(ctx.run_id, self.param_path, slot)
            ctx.store.emit("notification", {
                "level": "warn", "text": f"{self.name}: {old} is taken ({there} there); "
                f"{self.sample_id} goes to {slot} instead"}, run_id=ctx.run_id)
            return
        raise PreconditionFailed(f"{self.dest} is not empty ({there} there) and the {board} "
                                 f"board has no other free {holds} slot")

    def expected(self, ctx: FlowContext) -> dict[str, Any]:
        t, c = self._target(ctx), self._cfg(ctx)
        w = to_world(ctx.bench, self.arm, t)
        return {"sample": self.sample_id, "at": self.dest, "x": w.x, "y": w.y,
                "rotation": t.rotation, "frame": "world", "tolerance_mm": c["tolerance_mm"],
                "tolerance_deg": c["tolerance_deg"]}

    def act(self, ctx: FlowContext, correction: Correction | None) -> None:
        if self._holding and correction is None:
            place_held(ctx, self.arm, self._target(ctx))
            return
        src = self._source(ctx)
        if correction is not None:
            src = Pose(**correction.params["pick_from"])
            if correction.params.get("rehome"):
                lift(ctx, self.arm)
                ctx.gate.arm(self.arm).home()
        if self.release_switch:
            ctx.gate.switch(self.release_switch).set(False)
        region = self._source_name(ctx) if correction is None else "workspace"

        def still_there() -> bool:
            return any(d.pose.xy_distance(src) < NEAR_MM for d in self._look(ctx, region))

        pick_and_place(ctx, self.arm, src, self._target(ctx), still_there=still_there)

    def verify(self, ctx: FlowContext) -> Observation:
        c, t = self._cfg(ctx), self._target(ctx)
        exp = self.expected(ctx)
        imgs: list[str] = []
        at_dest = self._look(ctx, self.dest, imgs, expect=True)
        if len(at_dest) > 1:
            return Observation(Verdict.UNCLEAR, {f"{self.kind}s_at_dest": len(at_dest)}, exp,
                               images=imgs, message=f"more than one {self.kind} at {self.dest}")
        if not at_dest:
            src_name = self._source_name(ctx)
            left = [d for d in self._look(ctx, src_name, imgs)
                    if d.pose.xy_distance(self._source(ctx)) < NEAR_MM] if src_name else []
            if left:
                return Observation(Verdict.CORRECTABLE,
                                   {"found_at": src_name, "pose": self._world(ctx, left[0].pose),
                                    "arm_pose": left[0].pose.model_dump()},
                                   exp, images=imgs,
                                   message=f"{self.kind} still at {src_name} (pick failed)")
            strays = self._strays(ctx, imgs)
            if len(strays) == 1:
                return Observation(Verdict.CORRECTABLE,
                                   {"found_at": "workspace",
                                    "pose": self._world(ctx, strays[0].pose),
                                    "arm_pose": strays[0].pose.model_dump()},
                                   exp, images=imgs,
                                   message=f"{self.kind} dropped on the way, found on the bench")
            return Observation(Verdict.UNCLEAR, {"found_at": None, "strays": len(strays)}, exp,
                               images=imgs,
                               message=f"{self.kind} not found at {self.dest} or {src_name}")
        p = at_dest[0].pose
        dx, dy = p.x - t.x, p.y - t.y
        dist = p.xy_distance(t)
        drot = angle_diff(p.rotation, t.rotation)
        m = {"found_at": self.dest, "pose": self._world(ctx, p),
             "arm_pose": p.model_dump(), "dx_mm": round(dx, 3),
             "dy_mm": round(dy, 3), "offset_mm": round(dist, 3), "drot_deg": round(drot, 2)}
        if dist <= c["tolerance_mm"] and abs(drot) <= c["tolerance_deg"]:
            return Observation(Verdict.OK, m, exp, images=imgs)
        if dist <= c["correctable_within_mm"]:
            return Observation(Verdict.CORRECTABLE, m, exp, images=imgs,
                               message=f"off by {dist:.2f} mm / {drot:.1f} deg")
        return Observation(Verdict.UNCLEAR, m, exp, images=imgs, message=f"off by {dist:.2f} mm")

    def correct(self, ctx: FlowContext, obs: Observation) -> Correction | None:
        pose = obs.measurements.get("arm_pose")
        if pose is None:
            return None
        # pick_from is what the arm is commanded to (its own frame); the world
        # pose rides along for whoever reads the event (the measurements are world)
        at = {"pick_from": pose, "pick_from_frame": self.arm,
              "pick_from_world": obs.measurements.get("pose")}
        self._corrections += 1
        off = (obs.measurements.get("dx_mm"), obs.measurements.get("dy_mm"))
        if off[0] is not None and off[1] is not None:
            prev = self._offsets[-1] if self._offsets else None
            cur = (float(off[0]), float(off[1]))
            self._offsets.append(cur)
            if prev is not None and math.dist(prev, cur) <= max(
                    SAME_OFFSET_MM, 0.5 * min(math.hypot(*prev), math.hypot(*cur))):
                # the same error twice: the arm itself is off (lost steps), not the place
                return Correction("same offset twice: home the arm (lost steps?), re-pick "
                                  "and re-place", {**at, "rehome": True})
            if self._corrections >= self.max_corrections:
                # last try: home first, so lost steps are ruled out before giving up
                return Correction("last try: home the arm, re-pick and re-place",
                                  {**at, "rehome": True})
        return Correction("re-pick at measured pose and re-place", at)

    def ledger_props(self, obs: Observation | None) -> dict[str, Any]:
        return {"pose": obs.measurements.get("pose")} if obs else {}

    def on_success(self, ctx: FlowContext, obs: Observation | None) -> None:
        props = {**self.ledger_props(obs), "location_uncertain": False, "last_seen": None}
        if self.kind == "chip" and is_slot(ctx.bench, self.dest):
            props["home_slot"] = self.dest
        ctx.store.upsert_sample(self.sample_id, self.kind, location=self.dest,
                                status=self.new_status, props=props)
        # the object sits there now: the booking has done its job
        ctx.store.release_reservations(ctx.run_id, "slot", [self.dest])

    def on_give_up(self, ctx: FlowContext, obs: Observation) -> None:
        """Placed but off target: the ledger moves it to the destination (with
        the measured pose); dropped somewhere or not found: the ledger keeps the
        old place but marks it uncertain. The status is left as it was."""
        m = obs.measurements
        if "found_at" not in m and not any(k.endswith("_at_dest") for k in m):
            return  # nothing was looked at
        found = m.get("found_at")  # None: not found anywhere
        if found is not None and found == self._source_name(ctx):
            return  # the pick failed: it is still where the ledger says
        seen = {"at": found, "pose": m.get("pose"), "offset_mm": m.get("offset_mm")}
        if found == self.dest:
            ctx.store.upsert_sample(self.sample_id, self.kind, location=self.dest,
                                    props={"pose": m.get("pose"), "last_seen": seen,
                                           "location_uncertain": False})
        else:
            ctx.store.upsert_sample(self.sample_id, self.kind,
                                    props={"last_seen": seen, "location_uncertain": True})

    def safe_state(self, ctx: FlowContext) -> None:
        # Keep the vacuum as is (dropping a held object is worse); just stop high.
        lift(ctx, self.arm)

    def probe_state(self, ctx: FlowContext) -> ProbeResult:
        if ctx.gate.arm(self.arm).vacuum_on():
            # on the nozzle: resuming the step finishes the place; else unclear
            src = self._source_name(ctx)
            return ProbeResult.NOT_DONE if self._on_nozzle(ctx, src) else ProbeResult.UNKNOWN
        at_dest = self._look(ctx, self.dest)
        if len(at_dest) == 1:
            verdict = self.verify(ctx).verdict
            if verdict == Verdict.OK:
                return ProbeResult.DONE
            # placed but off: the resumed step re-places it (_interrupted_place)
            src = self._source_name(ctx)
            return (ProbeResult.NOT_DONE if verdict == Verdict.CORRECTABLE
                    and self._interrupted_place(ctx, src, at_dest) is not None
                    else ProbeResult.UNKNOWN)
        if not at_dest:
            src = self._source_name(ctx)
            return ProbeResult.NOT_DONE if src and self._look(ctx, src) else ProbeResult.UNKNOWN
        return ProbeResult.UNKNOWN
