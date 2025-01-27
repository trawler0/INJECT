import clip
from inject import INJECT
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
    parser.add_argument("--use-cached-images", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)
    parser.add_argument("--use-cached-data", default="True", type=str)
    parser.add_argument("--test-ema", action="store_true", default=False)
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--epoch-multiplier", type=float, default=1.)
    parser.add_argument("--return-best", action="store_true", default=False)
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

    run_name = f"{args.dataset_identifier}-{args.clip_model}-{args.templates}-{args.n_shot}"
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
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
        N_class, L, D = prompts.shape
        idxs = -np.ones((N_class, L))
        if args.use_cached_images:
            images = os.path.join(cache_dir, f"{args.dataset_identifier}-features.npz")
            images, idxs_images = np.load(images)["emb"], np.load(images)["idxs"]
            prompts = np.concatenate([prompts, images], axis=1)
            idxs = np.concatenate([idxs, idxs_images], axis=1)

        val_cached = os.path.join(cache_dir, f"{args.dataset_identifier}-val-features.npz")

        backbone = Backbone(args.clip_model)
        test_flags = ["val", "imagenet-r", "imagenet-a", "v2", "sketch"] if args.dataset_identifier == "imagenet" else ["val", "test"]

        model = INJECT(backbone=backbone, text_features=prompts, idxs=idxs, test_flags=test_flags)

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
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=0)
            test_dataloaders.append(test_dataloader)

        train_dataset = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot, seed=args.seed, transform=train_transforms)
        train_dataset = IdxDataset(train_dataset)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, num_workers=0)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True, persistent_workers=True)


        print("Starting training on", args.dataset_identifier)
        if args.return_best:
            checkpoint_callback = ModelCheckpoint(filename=os.path.join(tempfile.gettempdir(), "best.ckpt"), monitor="val_acc_0.9", mode="max")
            callbacks = [checkpoint_callback]
            enable_checkpointing = True
        else:
            callbacks = None
            enable_checkpointing = False
        trainer = Trainer(max_epochs=int(args.epochs * args.epoch_multiplier), precision=32, enable_checkpointing=enable_checkpointing, logger=False, callbacks=callbacks)
        trainer.fit(model, train_loader, val_loader)

        if args.return_best:
            model = INJECT.load_from_checkpoint(checkpoint_callback.best_model_path, backbone=backbone, text_features=prompts, test_flags=test_flags)
        # delete best model, no need to save for benchmarking
        if args.return_best and os.path.exists(checkpoint_callback.best_model_path):
            os.remove(checkpoint_callback.best_model_path)

        if args.test_ema:
            model = model.ema
        results = trainer.validate(model, test_dataloaders)

        log_metrics(results, test_flags)
        if args.save_weights:
            mlflow.pytorch.log_model(model, "models")


if __name__ == "__main__":
    main()


