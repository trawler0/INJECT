from inject import INJECT, BaselineEvaluator
from utils import Backbone, CachedDataset, log_metrics, DEFAULT_TRANSFORMS
from pytorch_lightning import Trainer
import mlflow
import argparse
import os
import numpy as np
from data import DATASETS, IdxDataset
import torch
from torchvision import transforms as T
from lora import lora_dinov2
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()

    default_root = os.getenv("DATA_ROOT")
    if default_root is None:
        default_root = "/home/marco/data"
    print(default_root)
    CACHED_FEATURES = "cached-features"

    parser.add_argument("dataset_identifier", type=str)
    parser.add_argument("dinov2_model", type=str)
    parser.add_argument("n_shot", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)
    parser.add_argument("--use-cached-data", action="store_false", default=True)
    parser.add_argument("--test-ema", action="store_true", default=False)
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--epoch-multiplier", type=float, default=1.)
    parser.add_argument("--lora-strategy", type=str, default="none")
    parser.add_argument("--val-frequency", type=int, default=10)
    parser.add_argument("--experiment", type=str, default=None)

    args = parser.parse_args()

    try:
        experiment = mlflow.get_experiment_by_name(args.experiment)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(args.experiment)
    except Exception as e:
        print(f"Error fetching or creating experiment: {e}")

    run_name = f"{args.dataset_identifier}-{args.dinov2_model}-{args.n_shot}"
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
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
        backbone.model = lora_dinov2(backbone.model, 10, 8, strategy=args.lora_strategy)
        test_flags = ["val", "imagenet-r", "imagenet-a", "v2", "sketch"] if args.dataset_identifier == "imagenet" else ["val", "test"]

        train_transforms = T.Compose([
            DEFAULT_TRANSFORMS,
            backbone.preprocess
        ])

        if args.use_cached_data:
            assert args.lora_strategy == "none", "Caching is only supported for the 'none' strategy"
            val_dataset = CachedDataset(val_cached)
        else:
            val_dataset = DATASETS.get(args.dataset_identifier)(args.root, "val", transform=backbone.preprocess)
        baseline_dataset = DATASETS.get(args.dataset_identifier)(args.root, "train", transform=backbone.preprocess, n_shot=args.n_shot, seed=args.seed)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=0)
        baseline_loader = torch.utils.data.DataLoader(baseline_dataset, batch_size=32, num_workers=0)

        test_dataloaders = []
        for test_flag in test_flags:
            test_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-{test_flag}-features.npz")
            if args.use_cached_data:
                test_dataset = CachedDataset(test_cached)
            else:
                test_dataset = DATASETS.get(args.dataset_identifier)(args.root, test_flag, transform=backbone.preprocess)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0)
            test_dataloaders.append(test_dataloader)

        def baseline_eval():
            feats = []
            labels = []
            backbone.eval()
            backbone.cuda()
            for x, y in tqdm(baseline_loader):
                x = x.cuda()
                with torch.no_grad():
                    feat = backbone(x)
                feats.append(feat.detach().cpu())
                labels.append(y)
            feats = torch.cat(feats)
            labels = torch.cat(labels)
            evaluator = BaselineEvaluator(backbone, feats, labels)
            trainer = Trainer(max_epochs=1, precision=32, enable_checkpointing=False, logger=False)
            results = trainer.validate(evaluator, test_dataloaders)
            log_metrics(results, test_flags)
        baseline_eval()


        prompts = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.n_shot}.npz")
        prompts = np.load(prompts)
        p, idxs = prompts["emb"], prompts["idxs"]
        ds = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot, transform=train_transforms, seed=args.seed)
        ds = IdxDataset(ds)

        # need to use leave use 50% as prompts and 50% as training samples, switching roles and ensembling improves performance

        backbone = Backbone(args.dinov2_model)
        backbone.model = lora_dinov2(backbone.model, 10, 8, strategy=args.lora_strategy)

        print("Training model")
        model = INJECT(backbone=backbone, text_features=p, idxs=idxs, test_flags=test_flags)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=2, shuffle=True, drop_last=True, persistent_workers=True)
        trainer = Trainer(max_epochs=int(args.epochs * args.epoch_multiplier), precision=32, enable_checkpointing=False, logger=False, check_val_every_n_epoch=args.val_frequency)
        trainer.fit(model, train_loader, val_loader)
        if args.test_ema:
            model = model.ema

        results = trainer.validate(model, test_dataloaders)

        log_metrics(results, test_flags)
        if args.save_weights:
            mlflow.pytorch.log_model(model, "models")


if __name__ == "__main__":
    main()

