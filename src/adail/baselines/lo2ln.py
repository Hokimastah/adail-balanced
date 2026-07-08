from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LO2LNExperiment:
    """Command metadata for running the official LO2LN repository.

    The ADAIL package does not vendor/copy LO2LN code. It builds reproducible
    commands for the official external repository at
    https://github.com/EvelynChee/LO2LN so ADAIL can be compared with LO2LN
    under matching class-incremental scenarios.
    """

    dataset: str
    scenario: str
    initial_increment: int
    increment: int
    batch_size: int
    option_file: str
    data_order_file: str
    default_data_subdir: str
    official_reported_avg_inc_accuracy: float | None = None


def _normalize_dataset(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "cch5000": "cch5000",
        "colorectal": "cch5000",
        "crc5000": "cch5000",
        "ham10000": "ham10000",
        "ham": "ham10000",
        "eyepacs": "eyepacs",
        "diabetic": "eyepacs",
        "eye": "eyepacs",
    }
    if key not in aliases:
        raise KeyError("dataset must be one of: cch5000, ham10000, eyepacs")
    return aliases[key]


def get_lo2ln_experiment(dataset: str, scenario: str) -> LO2LNExperiment:
    """Return LO2LN command metadata for a dataset/scenario pair.

    Parameters
    ----------
    dataset:
        cch5000, ham10000, or eyepacs.
    scenario:
        ADAIL/LO2LN class-incremental split string such as 4-1 or 3-2.
    """

    dataset_key = _normalize_dataset(dataset)
    try:
        base, inc = [int(x) for x in scenario.split("-", maxsplit=1)]
    except Exception as exc:  # noqa: BLE001
        raise ValueError("scenario must look like '4-1', '4-2', '3-1', or '3-2'") from exc

    if dataset_key == "cch5000":
        if base != 4 or inc not in {1, 2}:
            raise ValueError("CCH5000 comparison uses scenario 4-1 or 4-2")
        reported = 94.5 if inc == 1 else 94.6
        return LO2LNExperiment(
            dataset="cch5000",
            scenario=scenario,
            initial_increment=base,
            increment=inc,
            batch_size=32,
            option_file="options/der/derE3_colorectal.yaml",
            data_order_file="options/data/colorectal_3orders.yaml",
            default_data_subdir="data/colorectal",
            official_reported_avg_inc_accuracy=reported,
        )

    if dataset_key == "ham10000":
        if base != 3 or inc not in {1, 2}:
            raise ValueError("HAM10000 comparison uses scenario 3-1 or 3-2")
        reported = 78.1 if inc == 1 else 82.0
        return LO2LNExperiment(
            dataset="ham10000",
            scenario=scenario,
            initial_increment=base,
            increment=inc,
            batch_size=32,
            option_file="options/der/derE3_ham10000.yaml",
            data_order_file="options/data/ham10000_3orders.yaml",
            default_data_subdir="data/ham10000",
            official_reported_avg_inc_accuracy=reported,
        )

    if dataset_key == "eyepacs":
        if base != 3 or inc not in {1, 2}:
            raise ValueError("EyePACS comparison uses scenario 3-1 or 3-2")
        # The official README reports EyePACS for 1 new class per step.
        reported = 81.9 if inc == 1 else None
        return LO2LNExperiment(
            dataset="eyepacs",
            scenario=scenario,
            initial_increment=base,
            increment=inc,
            batch_size=128,
            option_file="options/der/derE3_diabetic.yaml",
            data_order_file="options/data/diabetic_3orders.yaml",
            default_data_subdir="data/diabetic",
            official_reported_avg_inc_accuracy=reported,
        )
    raise AssertionError(dataset_key)


def build_lo2ln_command(
    lo2ln_root: str | Path,
    dataset: str,
    scenario: str,
    data_path: str | Path | None = None,
    device: str = "0",
    workers: int = 4,
    label: str | None = None,
    save_model: bool = False,
    python_executable: str = "python3",
    fixed_memory: bool = True,
) -> list[str]:
    """Build the official LO2LN command for a fair baseline run."""

    exp = get_lo2ln_experiment(dataset, scenario)
    _ = Path(lo2ln_root)  # kept for API clarity; execute the command from this cwd.
    data_arg = str(data_path) if data_path is not None else exp.default_data_subdir
    label = label or f"lo2ln_{exp.dataset}_{exp.scenario.replace('-', '_')}"
    cmd = [
        python_executable,
        "-m",
        "inclearn",
        "--options",
        exp.option_file,
        exp.data_order_file,
        "--initial-increment",
        str(exp.initial_increment),
        "--increment",
        str(exp.increment),
    ]
    if fixed_memory:
        cmd.append("--fixed-memory")
    cmd.extend(
        [
            "--workers",
            str(workers),
            "--batch-size",
            str(exp.batch_size),
            "--device",
            str(device),
            "--label",
            label,
            "--data-path",
            str(data_arg),
        ]
    )
    if save_model:
        cmd.extend(["--save-model", "task"])
    return cmd


def command_to_string(command: list[str]) -> str:
    """Convert a command list into a shell-safe one-liner."""

    return " ".join(shlex.quote(part) for part in command)
