"""Qumus agent system (started from the qumus-v2 demo, 0cc7826).

The agent reaches lab modules only through instrument service contract v1
(qumus_simlab/contract client; qumus_simlab/simlab serves the simulation the same way).

Layering (top to bottom, each layer only calls the one below it):

    ui            web UI; reads the run store, writes user messages to the inbox
    agents        OpenAI Agents SDK team (orchestrator + subagents), tools, hooks
    runtime       job service, run store (SQLite), events, recovery
    flows         closed-loop Steps grouped into flows (molecule / assembly)
    safety        SafetyGate: every device command is checked against limits
    devices       device interfaces; sim/ (testbench); real drivers live in qumus-lab
    contract      contract v1 client for every lab window (qumus-lab, reactor)
    simlab        the simulated lab served through contract v1

See docs/architecture.md for the repositories and docs/sim-contracts.md for
the contracts between layers.
"""

__version__ = "0.1.0"
