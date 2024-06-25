import clip
from inject import INJECT
from utils import CLIPBackbone, CachedDataset, log_metrics
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

    parser.add_argument("dataset_identifier", type=str)
    parser.add_argument("clip_model", type=str)
    parser.add_argument("n_shot", type=int)
    parser.add_argument("templates", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)
    parser.add_argument("--use-cached-data", default="True", type=str)
    parser.add_argument("--test-ema", action="store_true", default=False)
    parser.add_argument("--weighing-strategy", type=str, default="linear")
    parser.add_argument("--save_weights", action="store_true", default=False)

    args = parser.parse_args()

    run_name = f"{args.dataset_identifier}-{args.clip_model}-{args.templates}-{args.n_shot}"
    with mlflow.start_run(run_name=run_name):
        torch.autograd.set_detect_anomaly(False)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mlflow.log_params(vars(args))
        mlflow.pytorch.autolog(log_models=False)

        cache_dir = args.cache_dir
        cache_dir = os.path.join(cache_dir, args.clip_model)

        prompts = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.templates}.npy")
        prompts = np.load(prompts)
        val_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-val-features.npz")

        preprocess = clip.load(args.clip_model)[1]
        train_transforms = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.4),
            preprocess
        ])

        if args.use_cached_data == "True":
            val_dataset = CachedDataset(val_cached)
        else:
            val_dataset = DATASETS.get(args.dataset_identifier)(args.root, "val", transform=preprocess)

        test_flags = ["val", "imagenet-r", "imagenet-a", "v2", "sketch"] if args.dataset_identifier == "imagenet" else ["val", "test"]
        test_dataloaders = []
        for test_flag in test_flags:
            test_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-{test_flag}-features.npz")
            if args.use_cached_data == "True":
                test_dataset = CachedDataset(test_cached)
            else:
                test_dataset = DATASETS.get(args.dataset_identifier)(args.root, test_flag, transform=preprocess)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=0)
            test_dataloaders.append(test_dataloader)

        train_dataset = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot, seed=args.seed, transform=train_transforms)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, num_workers=0)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True, drop_last=True, persistent_workers=True)


        backbone = CLIPBackbone(args.clip_model)
        model = INJECT(backbone, prompts, test_flags=test_flags, weighing_strategy=args.weighing_strategy)


        print("Starting training on", args.dataset_identifier)
        trainer = Trainer(max_epochs=args.epochs, precision=32, enable_checkpointing=False, logger=False)
        trainer.fit(model, train_loader, val_loader)
        if args.test_ema:
            model = model.ema
        results = trainer.validate(model, test_dataloaders)

        log_metrics(results, test_flags)
        if args.save_weights:
            mlflow.pytorch.log_model(model, "models")


if __name__ == "__main__":
    main()


