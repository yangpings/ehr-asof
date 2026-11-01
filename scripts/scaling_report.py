"""Summarize results/scaling.jsonl as a markdown table, and optionally plot it.

uv run python scripts/scaling_report.py
uv run --with matplotlib python scripts/scaling_report.py --plot docs/scaling.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/scaling.jsonl")
    ap.add_argument("--plot")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    print("| Configuration | Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | Errors |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(
            f"| {r['label']} | {r['clients']} | {r['throughput_rps']} | {r['p50_ms']} "
            f"| {r['p95_ms']} | {r['errors']} |"
        )

    if args.plot:
        import matplotlib.pyplot as plt

        by_label: dict[str, list[dict[str, float]]] = defaultdict(list)
        for r in rows:
            by_label[r["label"]].append(r)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for label, rs in by_label.items():
            xs = [r["clients"] for r in rs]
            ax1.plot(xs, [r["throughput_rps"] for r in rs], marker="o", label=label)
            ax2.plot(xs, [r["p95_ms"] for r in rs], marker="o", label=label)
        ax1.set(xlabel="Concurrent clients", ylabel="Throughput (req/s)", title="Throughput")
        ax2.set(xlabel="Concurrent clients", ylabel="p95 latency (ms)", title="p95 latency")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
