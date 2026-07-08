from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import __version__
from .config import ADAILConfig, DataConfig
from .data import ClassIncrementalScenario, build_imagefolder_loaders, get_preset
from .models import ADAILNet
from .trainer import ADAILTrainer
from .baselines import build_lo2ln_command, command_to_string, get_lo2ln_experiment
from .benchmarking.comparison import (
    extract_adail_metrics,
    extract_lo2ln_metrics,
    load_json_metrics,
    make_comparison_table,
    write_comparison_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="adail", description="ADAIL training utility")
    parser.add_argument("--version", action="version", version=f"ADAIL {__version__}")
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init-config", help="write a default YAML configuration")
    init.add_argument("--output", default="config.yaml")
    init.add_argument("--num-classes", type=int, default=None)
    init.add_argument("--dataset", choices=["cch5000", "ham10000", "eyepacs"], default=None)
    init.add_argument("--scenario", default=None)
    init.add_argument("--memory-size", type=int, default=2000)

    presets = sub.add_parser("list-presets", help="show proposal-aligned dataset presets")
    presets.add_argument("--json", action="store_true")

    train = sub.add_parser("train", help="train ADAIL from an ImageFolder dataset")
    train.add_argument("--config", default=None, help="optional YAML config created by init-config")
    train.add_argument("--data", default=None, help="dataset root with train/test folders")
    train.add_argument("--output", default="runs/adail")
    train.add_argument("--dataset", default="generic", help="generic, cch5000, ham10000, or eyepacs")
    train.add_argument("--num-classes", type=int, default=None)
    train.add_argument("--scenario", default=None, help="class scenario, e.g. 4-1, 4-2, 3-1, 3-2")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--image-size", type=int, default=None)
    train.add_argument("--memory-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--ordinal", action="store_true")
    train.add_argument("--no-pretrained", action="store_true")
    train.add_argument("--no-mask-unseen", action="store_true")

    lo2ln = sub.add_parser("lo2ln-command", help="print an official LO2LN baseline command")
    lo2ln.add_argument("--lo2ln-root", default="external/LO2LN", help="path to cloned EvelynChee/LO2LN repository")
    lo2ln.add_argument("--dataset", choices=["cch5000", "ham10000", "eyepacs"], required=True)
    lo2ln.add_argument("--scenario", required=True, help="4-1, 4-2, 3-1, or 3-2")
    lo2ln.add_argument("--data-path", default=None, help="dataset path as expected by LO2LN")
    lo2ln.add_argument("--device", default="0")
    lo2ln.add_argument("--workers", type=int, default=4)
    lo2ln.add_argument("--label", default=None)
    lo2ln.add_argument("--save-model", action="store_true")
    lo2ln.add_argument("--output-script", default=None, help="optional .sh file to write")

    comp = sub.add_parser("compare-results", help="make an ADAIL vs LO2LN comparison report")
    comp.add_argument("--dataset", choices=["cch5000", "ham10000", "eyepacs"], required=True)
    comp.add_argument("--scenario", required=True)
    comp.add_argument("--adail-history", required=True, help="path to ADAIL history.json")
    comp.add_argument("--lo2ln-average-accuracy", type=float, default=None, help="LO2LN average incremental accuracy; if omitted, official README value is used when available")
    comp.add_argument("--lo2ln-average-forgetting", type=float, default=None)
    comp.add_argument("--output", default="comparison.md", help=".md or .csv output")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ADAILConfig:
    if getattr(args, "config", None):
        cfg = ADAILConfig.from_yaml(args.config)
    else:
        preset = None
        if getattr(args, "dataset", None) and args.dataset != "generic":
            preset = get_preset(args.dataset)
        num_classes = args.num_classes or (preset.num_classes if preset is not None else None)
        if num_classes is None:
            raise SystemExit("--num-classes is required unless --dataset is a known preset")
        cfg = ADAILConfig(num_classes=num_classes)
        if preset is not None:
            cfg.use_ordinal = preset.ordinal
            cfg.data.dataset_name = preset.name
            cfg.data.scenario = args.scenario or preset.default_scenarios[0]
    if getattr(args, "data", None) is not None:
        cfg.data.root = args.data
    if getattr(args, "scenario", None) is not None:
        cfg.data.scenario = args.scenario
    if getattr(args, "batch_size", None) is not None:
        cfg.data.batch_size = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        cfg.data.num_workers = args.num_workers
    if getattr(args, "image_size", None) is not None:
        cfg.data.image_size = args.image_size
    if getattr(args, "memory_size", None) is not None:
        cfg.memory_size = args.memory_size
    if getattr(args, "lr", None) is not None:
        cfg.optimizer.lr = args.lr
    if getattr(args, "ordinal", False):
        cfg.use_ordinal = True
    if getattr(args, "no_mask_unseen", False):
        cfg.mask_unseen_classes = False
    return cfg


def main() -> None:
    args = _parse_args()
    if args.command == "init-config":
        preset = get_preset(args.dataset) if args.dataset else None
        num_classes = args.num_classes or (preset.num_classes if preset is not None else 5)
        cfg = ADAILConfig(num_classes=num_classes, memory_size=args.memory_size)
        if preset is not None:
            cfg.use_ordinal = preset.ordinal
            cfg.data = DataConfig(
                dataset_name=preset.name,
                scenario=args.scenario or preset.default_scenarios[0],
                batch_size=32,
                image_size=224,
                num_workers=4,
            )
        elif args.scenario is not None:
            cfg.data.scenario = args.scenario
        cfg.to_yaml(args.output)
        print(f"Wrote {args.output}")
        return

    if args.command == "list-presets":
        from .data import PRESETS

        payload = {k: v.__dict__ for k, v in PRESETS.items()}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for key, preset in PRESETS.items():
                scenarios = ", ".join(preset.default_scenarios)
                ordinal = "yes" if preset.ordinal else "no"
                print(f"{key}: {preset.num_classes} classes | scenarios: {scenarios} | ordinal: {ordinal}")
        return

    if args.command == "lo2ln-command":
        cmd = build_lo2ln_command(
            lo2ln_root=args.lo2ln_root,
            dataset=args.dataset,
            scenario=args.scenario,
            data_path=args.data_path,
            device=args.device,
            workers=args.workers,
            label=args.label,
            save_model=args.save_model,
        )
        command = command_to_string(cmd)
        shell_command = "cd " + command_to_string([args.lo2ln_root]) + " && " + command
        print(shell_command)
        if args.output_script:
            script = Path(args.output_script)
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + command_to_string([args.lo2ln_root]) + "\n" + command + "\n", encoding="utf-8")
            script.chmod(0o755)
            print(f"Wrote {script}")
        return

    if args.command == "compare-results":
        history = load_json_metrics(args.adail_history)
        adail_row = extract_adail_metrics(history, args.dataset, args.scenario, source=args.adail_history)
        exp = get_lo2ln_experiment(args.dataset, args.scenario)
        lo2ln_acc = args.lo2ln_average_accuracy
        if lo2ln_acc is None:
            lo2ln_acc = exp.official_reported_avg_inc_accuracy
        lo2ln_row = extract_lo2ln_metrics(
            dataset=args.dataset,
            scenario=args.scenario,
            average_accuracy=lo2ln_acc,
            average_forgetting=args.lo2ln_average_forgetting,
            source="official README reported value" if args.lo2ln_average_accuracy is None else "user supplied",
        )
        rows = [adail_row, lo2ln_row]
        print(make_comparison_table(rows))
        write_comparison_report(rows, args.output)
        print(f"Wrote {args.output}")
        return

    if args.command == "train":
        cfg = _config_from_args(args)
        if cfg.data.root is None:
            raise SystemExit("Dataset root is missing. Pass --data or set data.root in the YAML config.")

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(out / "config.yaml")

        loaders = build_imagefolder_loaders(
            root=cfg.data.root,
            image_size=cfg.data.image_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
        scenario = ClassIncrementalScenario.from_string(
            cfg.num_classes,
            cfg.data.scenario,
            class_order=cfg.data.class_order,
        )
        model = ADAILNet(
            num_classes=cfg.num_classes,
            pretrained_backbone=not args.no_pretrained,
            prototype_momentum=cfg.prototype_momentum,
            prototype_margin=cfg.prototype_margin,
        )
        trainer = ADAILTrainer(model=model, config=cfg, device=args.device)
        history = trainer.fit_scenario(
            train_dataset=loaders.train.dataset,
            test_dataset=loaders.test.dataset,
            scenario=scenario,
            epochs_per_task=args.epochs,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
        trainer.save_checkpoint(out / "adail_last.pt")
        with (out / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(json.dumps(history, indent=2))
        return

    print("Run `adail --help` for usage.")


if __name__ == "__main__":
    main()
