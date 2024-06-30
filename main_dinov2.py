from inject import INJECT, INJECTEnsemble
from utils import Backbone, CachedDataset, log_metrics, DEFAULT_TRANSFORMS
from pytorch_lightning import Trainer
import mlflow
import argparse
import os
import numpy as np
from data import DATASETS
import torch
from torchvision import transforms as T

def main():
    parser = argparse.ArgumentParser()

    try:
        default_root = os.getenv("DATA_ROOT")
    except:
        raise ValueError("Please set the environment variable DATA_ROOT to the folder where all datasets are stored")
    CACHED_FEATURES = "cached-features"

    parser.add_argument("dataset-identifier", type=str)
    parser.add_argument("dinov2-model", type=str)
    parser.add_argument("n-shot", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)
    parser.add_argument("--use-cached-data", default="True", type=str)
    parser.add_argument("--test-ema", action="store_true", default=False)
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--epoch-multiplier", type=float, default=1.)

    args = parser.parse_args()

    run_name = f"{args.dataset_identifier}-{args.dinov2_model}-{args.n_shot}"
    with mlflow.start_run(run_name=run_name):
        torch.autograd.set_detect_anomaly(False)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mlflow.log_params(vars(args))
        mlflow.pytorch.autolog(log_models=False)

        cache_dir = args.cache_dir
        cache_dir = os.path.join(cache_dir, args.dinov2_model)

        val_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-val-features.npz")

        backbone = Backbone(args.dinov2_model)
        test_flags = ["val", "imagenet-r", "imagenet-a", "v2", "sketch"] if args.dataset_identifier == "imagenet" else ["val", "test"]

        train_transforms = T.Compose([
            DEFAULT_TRANSFORMS,
            backbone.preprocess
        ])

        if args.use_cached_data == "True":
            val_dataset = CachedDataset(val_cached)
        else:
            val_dataset = DATASETS.get(args.dataset_identifier)(args.root, "val", transform=backbone.preprocess)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, num_workers=0)

        test_dataloaders = []
        for test_flag in test_flags:
            test_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-{test_flag}-features.npz")
            if args.use_cached_data == "True":
                test_dataset = CachedDataset(test_cached)
            else:
                test_dataset = DATASETS.get(args.dataset_identifier)(args.root, test_flag, transform=backbone.preprocess)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=0)
            test_dataloaders.append(test_dataloader)

        prompts = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.n_shot}.npz")
        prompts = np.load(prompts)
        p1, p2 = prompts["emb1"], prompts["emb2"]
        ds1 = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot, start_shot=args.n_shot//2, transform=train_transforms)
        ds2 = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot//2, start_shot=0, transform=train_transforms)

        # need to use leave use 50% as prompts and 50% as training samples, switching roles and ensembling improves performance

        print("Training model 1")
        model1 = INJECT(backbone=backbone, text_features=p1, test_flags=test_flags)
        train_loader = torch.utils.data.DataLoader(ds2, batch_size=args.batch_size, num_workers=2, shuffle=True, drop_last=True, persistent_workers=True)
        trainer = Trainer(max_epochs=int(args.epochs * args.epoch_multiplier), precision=32, enable_checkpointing=False, logger=False)
        trainer.fit(model1, train_loader, val_loader)
        if args.test_ema:
            model = model1.ema

        print("Training model 2")
        model2 = INJECT(backbone=backbone, text_features=p2, test_flags=test_flags)
        train_loader = torch.utils.data.DataLoader(ds1, batch_size=args.batch_size, num_workers=2, shuffle=True, drop_last=True, persistent_workers=True)
        trainer = Trainer(max_epochs=int(args.epochs * args.epoch_multiplier), precision=32, enable_checkpointing=False, logger=False)
        trainer.fit(model2, train_loader, val_loader)

        model = INJECTEnsemble(model1, model2)

        results = trainer.validate(model, test_dataloaders)

        log_metrics(results, test_flags)
        if args.save_weights:
            mlflow.pytorch.log_model(model, "models")


if __name__ == "__main__":
    main()

