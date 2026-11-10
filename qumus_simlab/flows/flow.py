"""A Flow = an ordered list of Steps (v1 molecule / assembly level).

CONTRACT. Hardware tools submit flows to the job service by name. Each flow:
  - declares a pydantic Params model (validated before the job is queued),
  - declares the devices it needs (locked for the whole run),
  - builds its Step list from params. Steps are checkpointed by sequence number,
    so build_steps must be deterministic for the same params (resume relies on it).
  - may fill in defaults from the sample ledger once, at submit time
    (`resolve_params`, e.g. "which bare chips"); the resolved params are what
    the run stores, so a resumed run makes the same choices.
  - may go on another way when a step gives up (`on_step_failed`, e.g. change a
    torn stamp): it returns changed params whose steps keep every step up to the
    failed one and add new ones after it; the job service stores them in one
    write, marks the failed step `skipped` and runs on, and a resume builds the
    same list from the stored params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from qumus_simlab.flows.step import Step, StepOutcome
from qumus_simlab.flows.workflow import Assembly, Molecule

if TYPE_CHECKING:
    from qumus_simlab.flows.context import FlowContext
    from qumus_simlab.runtime.store import RunStore


@dataclass
class Replan:
    """A flow's answer to a step that gave up: go on with these params (their
    steps keep everything up to the failed step and add steps after it).
    `note` says what changes, for the step record and a notification."""

    params: BaseModel
    note: str


class Flow:
    name: ClassVar[str]
    description: ClassVar[str] = ""
    Params: ClassVar[type[BaseModel]]
    devices: ClassVar[tuple[str, ...]] = ()
    restore_heaters_on_resume: ClassVar[bool] = True

    @property
    def shutdown_devices(self) -> tuple[str, ...]:
        """Ordinary failure cleanup scope; read-only flows may lock devices without commanding them."""
        return self.devices

    @property
    def bench(self) -> dict[str, Any]:
        """The bench layout (config/bench.yaml): slots and stations for resolve_params."""
        from qumus_simlab.config import load_yaml

        return getattr(self, "_bench", None) or load_yaml("bench.yaml")

    def validate_saved_params(self, raw: dict[str, Any], *, saved_run: dict | None = None, store: Any = None) -> None:
        """Apply a shared Params contract before defaults or recovery I/O."""
        if not isinstance(raw, dict):
            raise ValueError("Saved flow parameters must be an object")
        validate = getattr(self.Params, "validate_saved_execution_binding", None)
        if validate is not None:
            validate(raw, saved_run=saved_run, store=store) if saved_run is not None else validate(raw)
        self.Params.model_validate(raw)

    def resolve_params(self, params: BaseModel, store: RunStore) -> BaseModel:
        """Fill ledger-dependent defaults; raise ValueError if the job cannot run."""
        return params

    def reserve(self, params: BaseModel, store: RunStore, run_id: str) -> None:  # noqa: B027
        """Book in the store what the resolved params plan on (slots the job
        will fill, samples it will use), right after the run is created, so a
        job submitted later plans around them (`RunStore.reserve`)."""

    def build_workflow(self, params: BaseModel) -> Assembly | Molecule | None:
        """Optional executable hierarchy. Legacy flat flows return None."""
        return None

    def build_steps(self, params: BaseModel) -> list[Step]:
        workflow = self.build_workflow(params)
        if workflow is None:
            raise NotImplementedError
        return workflow.build_steps()

    def on_step_failed(self, params: BaseModel, step: Step, outcome: StepOutcome,
                       ctx: FlowContext) -> Replan | None:
        """`step` gave up (handed back: unclear, corrections used up, a
        precondition). Return a Replan when the flow can go on without a
        person, else None (the job parks as before). Called by the job service
        with the params the steps were built from."""
        return None


FLOWS: dict[str, type[Flow]] = {}


def register_flow(cls: type[Flow]) -> type[Flow]:
    if cls.name in FLOWS:
        raise ValueError(f"duplicate flow {cls.name}")
    FLOWS[cls.name] = cls
    return cls


def get_flow(name: str) -> Flow:
    import qumus_simlab.flows.library  # noqa: F401 - registers built-in flows

    try:
        return FLOWS[name]()
    except KeyError:
        raise KeyError(f"unknown flow {name!r}; known: {sorted(FLOWS)}") from None
