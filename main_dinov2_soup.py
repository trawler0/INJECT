from model import Adapter,  BaselineEvaluator, Soup
from utils import Backbone, CachedDataset, log_metrics, default_transforms, IMAGENET_MEAN, IMAGENET_STD
from pytorch_lightning import Trainer
import mlflow
import argparse
import os
import numpy as np
from data import DATASETS, IdxDataset
import torch
from torchvision import transforms as T


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)
    parser.add_argument("--use-cached-data", action="store_false", default=True)
    parser.add_argument("--test-ema", action="store_true", default=False)
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--epoch-multiplier", type=float, default=1.)
    parser.add_argument("--lora-strategy", type=str, default="none")
    parser.add_argument("--val-frequency", type=int, default=40)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--save", default=None, type=str)
    parser.add_argument("--eval-continuously", action="store_true", default=False)
    parser.add_argument("--no-mask", action="store_true", default=False)

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        mlflow.log_params(vars(args))
        mlflow.pytorch.autolog(log_models=False)

        cache_dir = args.cache_dir
        cache_dir = os.path.join(cache_dir, args.dinov2_model)

        val_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-val-features.npz")

        backbone = Backbone(args.dinov2_model)
        test_flags = ["val", "imagenet-r", "imagenet-a", "v2", "sketch"] if args.dataset_identifier == "imagenet" else ["val", "test"]


        if args.use_cached_data:
            assert args.lora_strategy == "none", "Caching is only supported for the 'none' strategy"
            val_dataset = CachedDataset(val_cached)
        else:
            val_dataset = DATASETS.get(args.dataset_identifier)(args.root, "val", transform=backbone.preprocess)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=0)

        test_dataloaders = []
        for test_flag in test_flags:
            test_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-{test_flag}-features.npz")
            if args.use_cached_data:
                test_dataset = CachedDataset(test_cached)
            else:
                test_dataset = DATASETS.get(args.dataset_identifier)(args.root, test_flag, transform=backbone.preprocess)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0)
            test_dataloaders.append(test_dataloader)

        prompts = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.n_shot}.npz")
        prompts = np.load(prompts)
        p, idxs = prompts["emb"], prompts["idxs"]
        if args.no_mask:
            N, L, D = p.shape
            idxs = -np.ones((N, L), dtype=np.int32)
        C, N, D = p.shape
        y = torch.arange(C).unsqueeze(1).expand(C, N).reshape(C*N)

        def baseline_eval():
            feats = torch.tensor(p).float().view(C*N, D)
            labels = y.reshape(C*N)
            evaluator = BaselineEvaluator(backbone, feats, labels, test_flags=test_flags)
            trainer = Trainer(max_epochs=1, precision=32, enable_checkpointing=False, logger=False)
            results = trainer.validate(evaluator, test_dataloaders)
            log_metrics(results)
        baseline_eval()

        models = []
        for j in range(args.n_runs):
            torch.manual_seed(j)
            reduction = np.random.randint(2, 10)
            lr = np.random.choice([5e-3, 2e-3, 1e-3])
            weight_decay = np.random.choice([1e-3, 1e-2, 5e-2])
            augmentation_strength = np.random.choice([.25, .5, .75, 1])
            epochs = int(args.epochs * args.epoch_multiplier * (np.random.rand() * .5 + .5))

            mlflow.log_param(f"reduction_{j}", reduction)
            mlflow.log_param(f"lr_{j}", lr)
            mlflow.log_param(f"weight_decay_{j}", weight_decay)
            mlflow.log_param(f"augmentation_strength_{j}", augmentation_strength)
            mlflow.log_param(f"epochs_{j}", epochs)

            train_transforms = T.Compose([
                default_transforms(augmentation_strength),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

            model = Adapter(reduction=reduction, backbone=backbone, text_features=p, idxs=idxs,
                            test_flags=test_flags, lr=lr, weight_decay=weight_decay)

            ds = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot, transform=train_transforms, seed=args.seed)
            ds = IdxDataset(ds)

            backbone = Backbone(args.dinov2_model)

            train_loader = torch.utils.data.DataLoader(ds, batch_size=min(args.batch_size, len(ds)), num_workers=4, shuffle=True, drop_last=True, persistent_workers=True)
            trainer = Trainer(max_epochs=epochs, precision=32, enable_checkpointing=False, logger=False, check_val_every_n_epoch=args.val_frequency)
            trainer.fit(model, train_loader, val_loader)
            models.append(model)


            if args.eval_continuously:
                ensemble = Soup(models, flag="uniform", test_flags=test_flags)
                trainer = Trainer(logger=False)
                results = trainer.validate(ensemble, test_dataloaders)
                for i in range(len(results)):
                    results[i] = {k.replace("/", "-"): v for k, v in results[i].items()}
                    results[i] = {k+f"_iteration_{j}": v for k, v in results[i].items()}
                log_metrics(results)
            else:
                results = trainer.validate(model, test_dataloaders)
                for i in range(len(results)):
                    results[i] = {f"{k}_{j}".replace("/", "-"): v for k, v in results[i].items()}
                log_metrics(results)

        ensemble = Soup(models, test_flags=test_flags)
        if args.save:
            checkpoint = torch.nn.ModuleList([model.adapter_layer for model in models])
            torch.save(checkpoint, args.save)
        trainer = Trainer()
        results = trainer.validate(ensemble, test_dataloaders)
        for i in range(len(results)):
            results[i] = {k.replace("/", "-"): v for k, v in results[i].items()}
        log_metrics(results)


if __name__ == "__main__":
    main()
