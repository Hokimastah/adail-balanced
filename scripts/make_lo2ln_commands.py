#!/usr/bin/env python3
"""Generate LO2LN baseline commands for all proposal scenarios."""

from adail.baselines import build_lo2ln_command, command_to_string

EXPERIMENTS = [
    ("cch5000", "4-1", "data/colorectal"),
    ("cch5000", "4-2", "data/colorectal"),
    ("ham10000", "3-1", "data/ham10000"),
    ("ham10000", "3-2", "data/ham10000"),
    ("eyepacs", "3-1", "data/diabetic"),
]

for dataset, scenario, data_path in EXPERIMENTS:
    command = build_lo2ln_command(
        lo2ln_root="external/LO2LN",
        dataset=dataset,
        scenario=scenario,
        data_path=data_path,
        device="0",
    )
    print(f"# {dataset} {scenario}")
    print("cd external/LO2LN && " + command_to_string(command))
    print()
