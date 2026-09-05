from __future__ import annotations

import argparse

import torch

from adail import ADAILConfig, ADAILNet, ADAILTrainer
from adail.data import ClassIncrementalScenario, build_imagefolder_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train ADAIL on an ImageFolder dataset")
    parser.add_argument("--data", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--scenario", default="3-2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=2000)
    parser.add_argument("--ordinal", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaders = build_imagefolder_loaders(args.data, batch_size=args.batch_size)
    cfg = ADAILConfig(
        num_classes=args.num_classes, memory_size=args.memory_size, use_ordinal=args.ordinal
    )
    model = ADAILNet.from_config(cfg)
    trainer = ADAILTrainer(model, cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    scenario = ClassIncrementalScenario.from_string(args.num_classes, args.scenario)
    history = trainer.fit_scenario(
        loaders.train.dataset,
        loaders.test.dataset,
        scenario,
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
    )
    print(history)
