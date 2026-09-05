#!/usr/bin/env python3
"""Small helper to create an ADAIL-vs-LO2LN report from ADAIL history.json."""

from __future__ import annotations

import argparse

from adail.baselines import get_lo2ln_experiment
from adail.benchmarking.comparison import (
    extract_adail_metrics,
    extract_lo2ln_metrics,
    load_json_metrics,
    make_comparison_table,
    write_comparison_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--adail-history", required=True)
    parser.add_argument("--lo2ln-average-accuracy", type=float, default=None)
    parser.add_argument("--output", default="comparison.md")
    args = parser.parse_args()

    history = load_json_metrics(args.adail_history)
    adail_row = extract_adail_metrics(history, args.dataset, args.scenario, args.adail_history)
    exp = get_lo2ln_experiment(args.dataset, args.scenario)
    lo2ln_accuracy = args.lo2ln_average_accuracy
    if lo2ln_accuracy is None:
        lo2ln_accuracy = exp.official_reported_avg_inc_accuracy
    lo2ln_row = extract_lo2ln_metrics(
        dataset=args.dataset,
        scenario=args.scenario,
        average_accuracy=lo2ln_accuracy,
        source="official README reported value"
        if args.lo2ln_average_accuracy is None
        else "user supplied",
    )
    rows = [adail_row, lo2ln_row]
    print(make_comparison_table(rows))
    write_comparison_report(rows, args.output)


if __name__ == "__main__":
    main()
