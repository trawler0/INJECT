import mlflow
import numpy as np
from matplotlib import pyplot as plt

def plot_scaling(name, n_shots):
    for n_shot in n_shots:
        experiment = mlflow.get_experiment_by_name(f"{name} {n_shot}")
        runs = mlflow.search_runs(experiment.experiment_id)
        for i, run in runs.iterrows():
            run_id = run["run_id"]
            run = mlflow.get_run(run_id)
            params = run.data.params
            metrics = run.data.metrics
            print(params, metrics)

plot_scaling("dinov2 dinov2_vits14", [2, 4, 8, 16])