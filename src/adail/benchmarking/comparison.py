from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkMetrics:
    method: str
    dataset: str
    scenario: str
    average_accuracy: float | None = None
    average_forgetting: float | None = None
    mae: float | None = None
    qwk: float | None = None
    source: str | None = None

    def to_row(self) -> dict[str, str]:
        payload = asdict(self)
        return {k: "" if v is None else str(v) for k, v in payload.items()}


def load_json_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _last_seen_metric(history: dict[str, Any], key: str) -> float | None:
    results = history.get("task_results") or []
    if not results:
        return None
    metrics = results[-1].get("metrics", {})
    val = metrics.get(key)
    return float(val) if val is not None else None


def _as_percent(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def extract_adail_metrics(
    history: dict[str, Any],
    dataset: str,
    scenario: str,
    source: str | None = None,
) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        method="ADAIL",
        dataset=dataset,
        scenario=scenario,
        average_accuracy=_as_percent(float(history.get("average_accuracy", 0.0))),
        average_forgetting=_as_percent(float(history.get("average_forgetting", 0.0))),
        mae=_last_seen_metric(history, "seen_mae"),
        qwk=_last_seen_metric(history, "seen_qwk"),
        source=source,
    )


def extract_lo2ln_metrics(
    dataset: str,
    scenario: str,
    average_accuracy: float | None = None,
    average_forgetting: float | None = None,
    source: str | None = None,
) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        method="LO2LN",
        dataset=dataset,
        scenario=scenario,
        average_accuracy=average_accuracy,
        average_forgetting=average_forgetting,
        source=source,
    )


def make_comparison_table(rows: list[BenchmarkMetrics]) -> str:
    headers = [
        "method",
        "dataset",
        "scenario",
        "average_accuracy",
        "average_forgetting",
        "mae",
        "qwk",
        "source",
    ]
    rendered_rows = [r.to_row() for r in rows]
    widths = {h: len(h) for h in headers}
    for row in rendered_rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    sep = "| " + " | ".join("-" * widths[h] for h in headers) + " |"
    head = "| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |"
    body = [
        "| " + " | ".join(row[h].ljust(widths[h]) for h in headers) + " |" for row in rendered_rows
    ]
    return "\n".join([head, sep, *body])


def write_comparison_report(rows: list[BenchmarkMetrics], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "dataset",
                    "scenario",
                    "average_accuracy",
                    "average_forgetting",
                    "mae",
                    "qwk",
                    "source",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_row())
        return
    with output.open("w", encoding="utf-8") as f:
        f.write("# ADAIL vs LO2LN Comparison\n\n")
        f.write(make_comparison_table(rows))
        f.write("\n")
