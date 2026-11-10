"""The sim lab: the simulated bench served through the instrument service contract v1.

The agent talks to every lab module through the same small HTTP contract
(describe / state / command / stop / lease / events). qumus-lab serves it for
the real exfoliation and stacking modules; the reactor services serve it for
the reactor. This package serves it for the full simulation that came with the
qumus-v2 demo, so the agent can run a whole simulated Hall bar through the same
door it will use for real instruments.

    service.py     generic contract v1 service (lease, busy, locked, stop, events)
    modules.py     the exfoliation and stacking modules over one simulated bench
    agent_link.py  the agent side: a JobService that submits work as module actions
    server.py      `qumus_simlab simlab`: both modules on loopback ports 8781 / 8782
"""
