"""Command line: `python -m qumus_simlab serve` starts the three sim-lab windows."""

from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="qumus-simlab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("serve", help="run the simulated lab: exfoliation :8781, "
                        "stacking :8782, vdW reactor :8783 (contract v1, loopback)")
    sp.add_argument("--db", required=True,
                    help="fresh run-store path; pictures go next to it in images/")
    sp.add_argument("--seed", type=int, default=3, help="world seed (default 3)")
    sp.add_argument("--target-bundle", default=None,
                    help="imported Hall-bar target folder (target.json + electrode-layout.gds); "
                    "default: the built-in eight-electrode target. Lab data: keep it out of git")
    sp.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; 0.0.0.0 or a LAN address lets "
                    "another computer connect directly)")
    sp.add_argument("--remote-ports", action="store_true",
                    help="also listen on 18781/18782/18783 (loopback) as remote windows: "
                    "the end of an SSH tunnel from the agent's computer")
    sp.add_argument("--exfoliation-port", type=int, default=None)
    sp.add_argument("--stacking-port", type=int, default=None)
    sp.add_argument("--reactor-port", type=int, default=None)
    sp.add_argument("--motion-delay", type=float, default=0.0,
                    help="seconds per single motion segment (go_to_point, jog)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.cmd == "serve":
        from qumus_simlab.simlab.server import main as serve

        serve(args)


if __name__ == "__main__":
    main()
