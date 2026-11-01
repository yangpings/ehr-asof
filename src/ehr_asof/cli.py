"""Command-line interface: ``ehr-asof <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ehr_asof.events import build_events, delay_report
from ehr_asof.io import read_tables, write_tables


def _synth(args: argparse.Namespace) -> None:
    from ehr_asof.synth import SynthConfig, generate

    tables = generate(SynthConfig(n_patients=args.patients, seed=args.seed))
    path = write_tables(tables, args.out)
    print(f"wrote synthetic MIMIC-shaped tables to {path} (synthetic data, not real patients)")


def _delays(args: argparse.Namespace) -> None:
    report = delay_report(build_events(read_tables(args.data)))
    print(report.round(2).to_string())


def _experiment(args: argparse.Namespace) -> None:
    from ehr_asof.experiment import TaskConfig, run

    cfg = TaskConfig(
        outcome=args.outcome, horizon_hours=args.horizon, seed=args.seed, n_boot=args.n_boot
    )
    result = run(read_tables(args.data), cfg)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ehr-asof", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write synthetic MIMIC-shaped tables")
    p.add_argument("--out", required=True)
    p.add_argument("--patients", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_synth)

    p = sub.add_parser("delays", help="how late each signal becomes available")
    p.add_argument(
        "--data", required=True, help="MIMIC-IV root (with hosp/) or a flat CSV directory"
    )
    p.set_defaults(func=_delays)

    p = sub.add_parser("experiment", help="reported vs honest vs deployed AUROC")
    p.add_argument("--data", required=True)
    p.add_argument("--outcome", choices=["mortality", "long_los"], default="mortality")
    p.add_argument("--horizon", type=float, default=24.0, help="hours after admission")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out")
    p.set_defaults(func=_experiment)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
