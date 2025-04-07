import clip
from model import Adapter, Soup
from utils import Backbone, CachedDataset, log_metrics, DEFAULT_TRANSFORMS
from pytorch_lightning import Trainer
import mlflow
import argparse
import os
import numpy as np
from data import DATASETS
import torch
from torchvision import transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
import tempfile
from data import IdxDataset

def main():
    parser = argparse.ArgumentParser()

    default_root = os.getenv("DATA_ROOT")
    if default_root is None:
        default_root = "/home/marco/data"
    print(default_root)
    CACHED_FEATURES = "cached-features"

    parser.add_argument("dataset_identifier", type=str)
    parser.add_argument("clip_model", type=str)
    parser.add_argument("n_shot", type=int)
    parser.add_argument("templates", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)
    parser.add_argument("--use-cached-data", default="True", type=str)
    parser.add_argument("--test-ema", action="store_true", default=False)
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--epoch-multiplier", type=float, default=1.)
    parser.add_argument("--return-best", action="store_true", default=False)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--val-frequency", type=int, default=40)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--greedy", action="store_true", default=False)



    args = parser.parse_args()

    try:
        experiment = mlflow.get_experiment_by_name(args.experiment)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(args.experiment)
    except Exception as e:
        print(f"Error fetching or creating experiment: {e}")

    run_name = f"{args.dataset_identifier}-{args.clip_model}-{args.templates}-{args.n_shot}"
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        mlflow.log_params(vars(args))
        mlflow.pytorch.autolog(log_models=False)

        cache_dir = args.cache_dir
        cache_dir = os.path.join(cache_dir, args.clip_model)

        prompts = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.templates}.npy")
        prompts = np.load(prompts)
        N, L, D = prompts.shape
        idxs = -np.ones((N, L), dtype=np.int32)
        val_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-val-features.npz")

        backbone = Backbone(args.clip_model)
        test_flags = ["val", "imagenet-r", "imagenet-a", "v2", "sketch"] if args.dataset_identifier == "imagenet" else ["val", "test"]

        models = []
        scores = []
        for j in range(args.n_runs):
            torch.manual_seed(j)
            reduction = np.random.randint(2, 10)
            lr = np.random.choice([2e-3, 1e-3, 5e-4])
            weight_decay = np.random.choice([1e-3, 1e-2, 5e-2])
            mlflow.log_param(f"reduction_{j}", reduction)
            mlflow.log_param(f"lr_{j}", lr)
            mlflow.log_param(f"weight_decay_{j}", weight_decay)
            model = Adapter(reduction=reduction, backbone=backbone, text_features=prompts, idxs=idxs, test_flags=test_flags, lr=lr, weight_decay=weight_decay)

            train_transforms = T.Compose([
                DEFAULT_TRANSFORMS,
                backbone.preprocess
            ])

            if args.use_cached_data == "True":
                val_dataset = CachedDataset(val_cached)
            else:
                val_dataset = DATASETS.get(args.dataset_identifier)(args.root, "val", transform=backbone.preprocess)

            test_dataloaders = []
            for test_flag in test_flags:
                test_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-{test_flag}-features.npz")
                if args.use_cached_data == "True":
                    test_dataset = CachedDataset(test_cached)
                else:
                    test_dataset = DATASETS.get(args.dataset_identifier)(args.root, test_flag, transform=backbone.preprocess)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0)
                test_dataloaders.append(test_dataloader)

            train_dataset = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot, seed=args.seed, transform=train_transforms)
            train_dataset = IdxDataset(train_dataset)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=0)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), num_workers=8, shuffle=True, drop_last=True, persistent_workers=True)


            print("Starting training on", args.dataset_identifier)
            if args.return_best:
                checkpoint_callback = ModelCheckpoint(filename=os.path.join(tempfile.gettempdir(), "best.ckpt"), monitor="val_acc_0.9", mode="max")
                callbacks = [checkpoint_callback]
                enable_checkpointing = True
            else:
                callbacks = None
                enable_checkpointing = False
            print(args.epochs, args.epoch_multiplier)
            trainer = Trainer(max_epochs=int(args.epochs * args.epoch_multiplier), precision=32, enable_checkpointing=enable_checkpointing, logger=False, callbacks=callbacks, check_val_every_n_epoch=args.val_frequency)
            trainer.fit(model, train_loader, val_loader)

            # delete best model, no need to save for benchmarking
            if args.return_best and os.path.exists(checkpoint_callback.best_model_path):
                os.remove(checkpoint_callback.best_model_path)

            if args.test_ema:
                model = model.ema
            results = trainer.validate(model, test_dataloaders)
            for i in range(len(results)):
                results[i] = {f"{k}_{j}".replace("/", "-"): v for k, v in results[i].items()}
            log_metrics(results, test_flags)
            models.append(model)
            if args.greedy:
                score = results[0][f"val_acc_1-dataloader_idx_0_{j}"]
                scores.append(score)
            if args.save_weights:
                mlflow.pytorch.log_model(model, "models")
        ensemble = Soup(models, flag="uniform", test_flags=test_flags)
        trainer = Trainer(logger=False)
        results = trainer.validate(ensemble, test_dataloaders)
        for i in range(len(results)):
            results[i] = {k.replace("/", "-"): v for k, v in results[i].items()}
        log_metrics(results, test_flags)
        if args.greedy:
            idx = np.argsort(scores)
            models = [models[i] for i in reversed(idx)]
            scores = [scores[i] for i in reversed(idx)]
            current_soup = [models[0]]
            current_score = 0.
            for j in range(0, len(models)):
                ensemble = Soup(models[:j+1], flag="search", test_flags=test_flags)
                trainer = Trainer(logger=False)
                results = trainer.validate(ensemble, test_dataloaders)
                all_scores = [results[0][f"acc_{thresh}_search-dataloader_idx_0"] for thresh in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]]
                score = np.max(all_scores)
                if score > current_score:
                    current_score = score
                    current_soup.append(models[j])

            ensemble = Soup(current_soup, flag="greedy", test_flags=test_flags)
            trainer = Trainer(logger=False)
            results = trainer.validate(ensemble, test_dataloaders)
            log_metrics(results, test_flags)

if __name__ == "__main__":
    main()

