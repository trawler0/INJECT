import clip
from model import Adapter, Soup
from utils import Backbone, CachedDataset, log_metrics, default_transforms
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
    parser.add_argument("--save", default=None, type=str)
    parser.add_argument("--eval-continuously", action="store_true", default=False)
    parser.add_argument("--save-example-feats", action="store_true", default=False)

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
        for j in range(args.n_runs):
            torch.manual_seed(j)
            reduction = np.random.randint(2, 10)
            lr = np.random.choice([2e-3, 1e-3, 5e-4])
            weight_decay = np.random.choice([1e-3, 1e-2, 5e-2])
            augmentation_strength = np.random.rand()
            epochs = int(args.epochs * args.epoch_multiplier * (np.random.rand() * .75 + .25))


            mlflow.log_param(f"reduction_{j}", reduction)
            mlflow.log_param(f"lr_{j}", lr)
            mlflow.log_param(f"weight_decay_{j}", weight_decay)
            mlflow.log_param(f"augmentation_strength_{j}", augmentation_strength)
            mlflow.log_param(f"epochs_{j}", epochs)

            train_transforms = T.Compose([
                default_transforms(augmentation_strength),
                backbone.preprocess
            ])

            model = Adapter(reduction=reduction, backbone=backbone, text_features=prompts, idxs=idxs, test_flags=test_flags, lr=lr, weight_decay=weight_decay)

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
            trainer = Trainer(max_epochs=epochs, precision=32, enable_checkpointing=enable_checkpointing, logger=False, callbacks=callbacks, check_val_every_n_epoch=args.val_frequency)
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

        ensemble = Soup(models, flag="uniform", test_flags=test_flags)
        trainer = Trainer(logger=False)
        results = trainer.validate(ensemble, test_dataloaders)
        for i in range(len(results)):
            results[i] = {k.replace("/", "-"): v for k, v in results[i].items()}
        log_metrics(results)
        with torch.no_grad():
            if args.save_example_feats:
                X = []
                labels = []
                ensemble.to("cuda")
                for (x, y) in val_loader:
                    x = x.to("cuda")
                    x = torch.nn.functional.normalize(x, p=2, dim=-1)
                    out = [x] + [models[j].adapter_layer(x) for j in range(args.n_runs)]
                    out = torch.stack(out, 1)
                    out = out.cpu().numpy()
                    X.append(out)
                    labels.append(y.cpu().numpy())
                X = np.concatenate(X, axis=0)
                labels = np.concatenate(labels, axis=0)
                tempdir = tempfile.gettempdir()
                np.savez(os.path.join(tempdir, "examples.npz"), X=X, y=labels)
                mlflow.log_artifact(os.path.join(tempdir, "examples.npz"))



if __name__ == "__main__":
    main()

