import mlflow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import os
from matplotlib import cm
from data import DATASETS
import pandas as pd

N = 8

def pick(architecture, model, dataset_identifier, score, n_shot, format=True, use_idxs=True):

    experiment = fr"{architecture}_soup_v3 {model} {n_shot}"
    if dataset_identifier == "imagenet":
        experiment = fr"{architecture}_soup_imagenet_v3 {model}"
    data = pd.read_csv(os.path.join("results", f"{experiment.replace(' ', '_').replace('/', '_')}.csv"))
    data = data[data["params.dataset_identifier"] == dataset_identifier]
    if dataset_identifier == "imagenet":
        data = data[data["params.n_shot"] == n_shot]
    if format:
        thresholds = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
        if use_idxs:
            idx = list(range(N))
            metrics = [[data["metrics." + score.format(thresh, idx)] for idx in idx] for thresh in thresholds]
            metrics = np.array(metrics)
        else:
            metrics = [data["metrics." + score.format(thresh)] for thresh in thresholds]
            metrics = np.array(metrics)

    else:
        metrics = data["metrics." + score]
    return metrics


datasets = ("fgvc_aircraft", "eurosat", "caltech-101", "dtd", "food-101", "oxford_flowers", "oxford_pets", "standford_cars", "ucf101", "imagenet", "sun397")


def clip(model, dataset_identifiers=datasets):
    metric_ = {}
    all_metric_ = {}
    zero_ = {}
    soup_ = {}
    avg_soup_ = {}
    avg_ = {}
    for dataset_identifier in dataset_identifiers:
        metric_[dataset_identifier] = []
        all_metric_[dataset_identifier] = []
        soup_[dataset_identifier] = []
        avg_[dataset_identifier] = []
        avg_soup_[dataset_identifier] = []
        for n_shot in [2, 4, 8, 16]:
            metric_name = "test_acc_{}-dataloader_idx_1_{}"
            soup_name = "acc_{}_uniform-dataloader_idx_1"
            if dataset_identifier == "imagenet":
                metric_name = "val_acc_{}-dataloader_idx_0_{}"
                soup_name = "acc_{}_uniform-dataloader_idx_0"
            scores = pick("clip", model, dataset_identifier, metric_name, n_shot)
            soup_scores = pick("clip", model, dataset_identifier, soup_name, n_shot, use_idxs=False)
            val_scores = pick("clip", model, dataset_identifier, "val_acc_{}-dataloader_idx_0_{}", n_shot)
            soup_val_scores = pick("clip", model, dataset_identifier, "acc_{}_uniform-dataloader_idx_0", n_shot, use_idxs=False)

            mean = scores[5:].mean()
            mean_soup = soup_scores[5:].mean()

            top_idx = np.argmax(val_scores, axis=0)
            zero_shot = scores[0, 0]
            avg = np.array([scores[top_idx[j], j] for j in range(N)]).mean() #scores[top_idx].mean()
            score = np.array([scores[top_idx[j], j] for j in range(N)])

            top_idx_soup = np.argmax(soup_val_scores)

            score_soup = soup_scores[top_idx_soup]

            metric_[dataset_identifier].append(avg)
            all_metric_[dataset_identifier].append(score)
            soup_[dataset_identifier].append(score_soup)
            avg_[dataset_identifier].append(mean)
            avg_soup_[dataset_identifier].append(mean_soup)
        all_metric_[dataset_identifier] = np.array(all_metric_[dataset_identifier])
        zero_[dataset_identifier] = zero_shot

    L = len(metric_.keys())
    rows = 4
    cols = 3
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    n_shots = [2, 4, 8, 16]
    metric_["average"] = np.mean(list(metric_.values()), axis=0)
    all_metric_["average"] = np.mean(np.stack(all_metric_.values(), 0), axis=0)
    zero_["average"] = np.mean(list(zero_.values()), axis=0)
    soup_["average"] = np.mean(list(soup_.values()), axis=0)
    avg_["average"] = np.mean(list(avg_.values()), axis=0)
    avg_soup_["average"] = np.mean(list(avg_soup_.values()), axis=0)

    for i, (dataset_identifier, scores) in enumerate(metric_.items()):
        ax = axs[i // cols, i % cols]
        ax.plot(n_shots, np.array(scores)*100, marker='o', linestyle='-', linewidth=2, markersize=2, color='g', label="Adapter Average")
        ax.plot(n_shots, np.array(soup_[dataset_identifier])*100, marker='o', linestyle='-', linewidth=2, markersize=4, label="CAT-Adapter",
                color='b')
        """ax.plot(n_shots, avg_[dataset_identifier], marker='o', linestyle='-', linewidth=2, markersize=4, label="avg",
                color='r')
        ax.plot(n_shots, avg_soup_[dataset_identifier], marker='o', linestyle='-', linewidth=2, markersize=4, label="avg soup",
                color='orange')"""
        for j in range(N):
            ax.plot(n_shots, np.array(all_metric_[dataset_identifier][:, j, 0, 0])*100, marker='x', linestyle='--', linewidth=0.5,
                    markersize=3, color='y', alpha=1, label="Individual Adapters" if j == 0 else None)
        ax.plot(0, np.array(zero_[dataset_identifier])*100, marker='*', linestyle='--', linewidth=2, markersize=6,
                label="Zero-Shot", color='purple')

        ax.set_title(dataset_identifier, fontsize=16, fontweight='bold')
        if i % 1 == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        else:
            ax.set_yticklabels([])  # remove y-ticks

            # Only label x-axis on last row
        if i // cols == rows - 1:
            ax.set_xlabel("Number of labeled training examples per class", fontsize=12)
        else:
            ax.set_xticklabels([])  # remove x-ticks

        # Turn on minor ticks and finer grid
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)

        # Comment this out if you want a single legend
        # ax.legend(fontsize=12)

    # Get handles and labels from the first axis
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
    plt.savefig("figures/clip.png", bbox_inches='tight')

def dinov2(model, dataset_identifiers=datasets):
    metric_ = {}
    all_metric_ = {}
    knn_ = {}
    proto_ = {}
    soup_ = {}
    fixed_ = {}
    soup_fixed = {}
    for dataset_identifier in dataset_identifiers:
        metric_[dataset_identifier] = []
        all_metric_[dataset_identifier] = []
        soup_[dataset_identifier] = []
        fixed_[dataset_identifier] = []
        soup_fixed[dataset_identifier] = []
        knn_[dataset_identifier] = []
        proto_[dataset_identifier] = []
        for n_shot in [2, 4, 8, 16]:
            metric_name = "test_acc_{}-dataloader_idx_1_{}"
            soup_name = "acc_{}_uniform-dataloader_idx_1"
            if dataset_identifier == "imagenet":
                metric_name = "val_acc_{}-dataloader_idx_0_{}"
                soup_name = "acc_{}_uniform-dataloader_idx_0"
            scores = pick("dinov2", model, dataset_identifier, metric_name, n_shot)
            soup_scores = pick("dinov2", model, dataset_identifier, soup_name, n_shot, use_idxs=False)
            val_scores = pick("dinov2", model, dataset_identifier, "val_acc_{}-dataloader_idx_0_{}", n_shot)
            soup_val_scores = pick("dinov2", model, dataset_identifier, "acc_{}_uniform-dataloader_idx_0", n_shot, use_idxs=False)

            fixed_score = scores[6].mean()
            fixed_soup_score = soup_scores[6]

            top_idx = np.argmax(val_scores, axis=0)
            avg = np.array([scores[top_idx[j], j] for j in range(N)]).mean() #scores[top_idx].mean()
            score = np.array([scores[top_idx[j], j] for j in range(N)])

            top_idx_soup = np.argmax(soup_val_scores)
            score_soup = soup_scores[top_idx_soup]

            metric_[dataset_identifier].append(avg)
            all_metric_[dataset_identifier].append(score)
            soup_[dataset_identifier].append(score_soup)
            fixed_[dataset_identifier].append(fixed_score)
            soup_fixed[dataset_identifier].append(fixed_soup_score)
            knn_[dataset_identifier].append(pick("dinov2", model, dataset_identifier, "knn_acc/dataloader_idx_1", n_shot, use_idxs=False)[0])
            proto_[dataset_identifier].append(pick("dinov2", model, dataset_identifier, "proto_acc/dataloader_idx_1", n_shot, use_idxs=False)[0])
        all_metric_[dataset_identifier] = np.array(all_metric_[dataset_identifier])

    L = len(metric_.keys())
    rows = 4
    cols = 3
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    n_shots = [2, 4, 8, 16]
    metric_["average"] = np.mean(list(metric_.values()), axis=0)
    all_metric_["average"] = np.mean(np.stack(all_metric_.values(), 0), axis=0)
    knn_["average"] = np.mean(list(knn_.values()), axis=0)
    proto_["average"] = np.mean(list(proto_.values()), axis=0)
    soup_["average"] = np.mean(list(soup_.values()), axis=0)
    fixed_["average"] = np.mean(list(fixed_.values()), axis=0)
    soup_fixed["average"] = np.mean(list(soup_fixed.values()), axis=0)

    for i, (dataset_identifier, scores) in enumerate(metric_.items()):
        ax = axs[i // cols, i % cols]
        ax.plot(n_shots, np.array(scores)*100, marker='o', linestyle='-', linewidth=2, markersize=2, color='g', label="Adapter Average")
        ax.plot(n_shots, np.array(soup_[dataset_identifier])*100, marker='o', linestyle='-', linewidth=2, markersize=4, label="Soup-Adapter",
                color='b')
        for j in range(N):
            ax.plot(n_shots, all_metric_[dataset_identifier][:, j, 0, 0]*100, marker='x', linestyle='--', linewidth=0.5,
                    markersize=3, color='y', alpha=1, label="Individual Adapters" if j == 0 else None)
        ax.plot(n_shots, np.array(knn_[dataset_identifier])*100, marker='*', linestyle='--', linewidth=2, markersize=4, label="KNN",
                color='r')
        ax.plot(n_shots, np.array(proto_[dataset_identifier])*100, marker='*', linestyle='--', linewidth=2, markersize=4,
                label="Proto", color='purple')

        ax.set_title(dataset_identifier, fontsize=16, fontweight='bold')
        if i % 1 == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        else:
            ax.set_yticklabels([])  # remove y-ticks

            # Only label x-axis on last row
        if i // cols == rows - 1:
            ax.set_xlabel("Number of labeled training examples per class", fontsize=12)
        else:
            ax.set_xticklabels([])  # remove x-ticks

        # Turn on minor ticks and finer grid
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)

        # Comment this out if you want a single legend
        # ax.legend(fontsize=12)

    # Get handles and labels from the first axis
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=18)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
    plt.savefig("figures/dinov2.png", bbox_inches='tight')

def robustness_clip(id=0):

    fig, ax = plt.subplots(figsize=(15, 15))

    experiment_name = ["clip_soup_imagenet_v3 ViT-B/32", "clip_soup_imagenet_v3 ViT-B/16"][id]
    # experiment = mlflow.get_experiment_by_name(experiment_name)
    data = pd.read_csv(os.path.join("results", f"{experiment_name.replace(' ', '_').replace('/', '_')}.csv"))
    for n in range(2):
        for run in data.iloc:
            n_shot = run["params.n_shot"]
            thresholds = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
            idx = [0, 1, 2, 3, 4, 5, 6, 7]
            val_scores = [[run["metrics." + f"val_acc_{thresh}-dataloader_idx_0_{i}"] for i in idx] for thresh in thresholds]
            val_scores = np.array(val_scores)
            top = np.argmax(val_scores, axis=0)
            #val_scores = val_scores[5]
            # val_scores = [val_scores[top[i], i] for i in range(10)]
            shifts = ["imagenet-r", "imagenet-a", "v2", "sketch"]
            shift_scores = []
            for j, shift in enumerate(shifts):
                scores = [[run["metrics." + f"{shift}_acc_{thresh}-dataloader_idx_{j+1}_{i}"] for i in idx] for thresh in thresholds]
                scores = np.array(scores)
                # scores = scores[5]
                # scores = [scores[top[i], i] for i in range(10)]
                shift_scores.append(scores)
            shift_scores = np.stack(shift_scores, axis=0)
            shift_scores = np.mean(shift_scores, axis=0)

            val_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{0}"] for thresh in thresholds])
            top_uniform = np.argmax(val_uniform)
            val_zero = val_uniform[0]
            # val_uniform = val_uniform[top_uniform]

            r_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{1}"] for thresh in thresholds])
            r_zero = r_uniform[0]
            #r_uniform = r_uniform[top_uniform]

            a_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{2}"] for thresh in thresholds])
            a_zero = a_uniform[0]
            #a_uniform = a_uniform[top_uniform]

            v2_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{3}"] for thresh in thresholds])
            v2_zero = v2_uniform[0]
            #v2_uniform = v2_uniform[top_uniform]

            sketch_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{4}"] for thresh in thresholds])
            sketch_zero = sketch_uniform[0]
            #sketch_uniform = sketch_uniform[top_uniform]

            shift_uniform = .25 * (r_uniform + a_uniform + v2_uniform + sketch_uniform)
            shift_zero = .25 * (r_zero + a_zero + v2_zero + sketch_zero)

            # Generate color smoothly from blue to red
            cmap = cm.get_cmap('coolwarm')
            cmap_soup = cm.get_cmap("coolwarm")
            curve_colors = {
                "2": cmap(0.0),
                "4": cmap(0.33),
                "8": cmap(0.66),
                "16": cmap(1.),
                "32": cmap(1.0)
            }
            def darken_color(color, factor=0.9):
                return tuple(np.array(color[:3]) * factor) + (color[3],)
            soup_colors = {
                "2": darken_color(cmap_soup(0.0)),
                "4": darken_color(cmap_soup(0.33)),
                "8": darken_color(cmap_soup(0.66)),
                "16": darken_color(cmap_soup(1.)),
                "32": darken_color(cmap_soup(1.0))
            }
            curve_color = curve_colors[str(n_shot)]
            soup_color = soup_colors[str(n_shot)]

            # Marker sizes increase from left to right
            marker_sizes = np.linspace(50, 200, val_scores.shape[0])

            # Plot curves with increasing marker sizes
            for i in range(val_scores.shape[0]):
                for j in range(val_scores.shape[1]):
                    if n == 0:
                        ax.plot(
                            val_scores[i:i + 2, j] * 100,
                            shift_scores[i:i + 2, j] * 100,
                            color=curve_color,
                            linestyle="-",
                            linewidth=0.5,
                            marker='o',
                            markersize=marker_sizes[i] / 40,  # scale down for better appearance
                            markeredgecolor='k',
                            alpha=.5
                        )
                if n == 1:
                    ax.plot(
                        val_uniform[i:i+2] * 100, shift_uniform[i:i+2] * 100,
                        marker='D', alpha=1, linestyle='-', linewidth=2,
                        markeredgecolor='k', color=soup_color, markersize=marker_sizes[i] / 20
                    )

        ax.scatter(val_zero * 100, shift_zero * 100, marker='*', color='purple', s=300, zorder=10)
        # Title and labels
        # ax.set_title("Comparison of Validation vs Shift Scores", fontsize=20, fontweight='bold', pad=20)
        # ax.set_xlabel("Validation accuracy (%)", fontsize=16, labelpad=15)
        if id == 0:
            ax.set_ylabel("Average accuracy over four distribution shifts (%)", fontsize=24, labelpad=15)
        if id == 1:
            ax.set_xlim(0.63 * 100, 0.74 * 100)
            ax.set_ylim(0.51 * 100, 0.61 * 100)
        else:
            ax.set_xlim(0.56 * 100, 0.69 * 100)
            ax.set_ylim(0.40 * 100, 0.51 * 100)

        # Grid lines
        ax.grid(True, linestyle='--', alpha=0.5)

        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=28)

        # Tight layout
        plt.tight_layout()
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

        # Display the plot
        #plt.show()
    # Define custom legend elements
    custom_lines = [
        Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=8, label='Individual Adapters'),
        Line2D([0], [0], color='gray', marker='D', linestyle='None', markersize=10, label='CAT Adapter'),
        Line2D([0], [0], color='purple', marker='*', linestyle='None', markersize=14, label='Zero Shot'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.0), lw=2, label='2-shot'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.33), lw=2, label='4-shot'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.66), lw=2, label='8-shot'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(1.0), lw=2, label='16-shot'),
        Line2D([], [], linestyle='None', label=["CLIP ViT-B/32", "CLIP ViT-B/16"][id], linewidth=0)
    ]

    ax.legend(
        handles=custom_lines,
        loc='lower right',
        fontsize=24,
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='gray'
    )
    #ax.text(70, 61, "CLIP ViT-B/16", fontsize=12, color='black', weight='bold')
    #ax.text(65, 51, "CLIP ViT-B/32", fontsize=12, color='black', weight='bold')
    plt.savefig(f"figures/clip_robustness_{id}.png")

def robustness_dinov2(id=0):

    fig, ax = plt.subplots(figsize=(15, 15))
    experiment_name = ["dinov2_soup_imagenet_v3 dinov2_vits14", "dinov2_soup_imagenet_v3 dinov2_vitb14_reg"][id]
    data = pd.read_csv(os.path.join("results", f"{experiment_name.replace(' ', '_').replace('/', '_')}.csv"))
    proto_val_ = []
    knn_val_ = []
    proto_shift_ = []
    knn_shift_ = []
    for n in range(2):
        for run in data.iloc:
            run_data = mlflow.get_run(run["run_id"])
            metrics = run_data.data.metrics
            n_shot = run_data.data.params["n_shot"]
            thresholds = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
            idx = [0, 1, 2, 3, 4, 5, 6, 7]
            val_scores = [[run["metrics." + f"val_acc_{thresh}-dataloader_idx_0_{i}"] for i in idx] for thresh in thresholds]
            val_scores = np.array(val_scores)
            top = np.argmax(val_scores, axis=0)
            #val_scores = val_scores[5]
            # val_scores = [val_scores[top[i], i] for i in range(10)]
            shifts = ["imagenet-r", "imagenet-a", "v2", "sketch"]
            shift_scores = []
            for j, shift in enumerate(shifts):
                scores = [[run["metrics." + f"{shift}_acc_{thresh}-dataloader_idx_{j+1}_{i}"] for i in idx] for thresh in thresholds]
                scores = np.array(scores)
                # scores = scores[5]
                # scores = [scores[top[i], i] for i in range(10)]
                shift_scores.append(scores)
            shift_scores = np.stack(shift_scores, axis=0)
            shift_scores = np.mean(shift_scores, axis=0)

            val_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{0}"] for thresh in thresholds])
            top_uniform = np.argmax(val_uniform)
            val_proto = run["metrics." + f"proto_acc/dataloader_idx_0"]
            val_knn = run["metrics." + f"knn_acc/dataloader_idx_0"]
            # val_uniform = val_uniform[top_uniform]

            r_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{1}"] for thresh in thresholds])
            r_proto = run["metrics." + f"proto_acc/dataloader_idx_1"]
            r_knn = run["metrics." + f"knn_acc/dataloader_idx_1"]
            #r_uniform = r_uniform[top_uniform]

            a_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{2}"] for thresh in thresholds])
            a_proto = run["metrics." + f"proto_acc/dataloader_idx_2"]
            a_knn = run["metrics." + f"knn_acc/dataloader_idx_2"]
            #a_uniform = a_uniform[top_uniform]

            v2_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{3}"] for thresh in thresholds])
            v2_proto = run["metrics." + f"proto_acc/dataloader_idx_3"]
            v2_knn = run["metrics." + f"knn_acc/dataloader_idx_3"]
            #v2_uniform = v2_uniform[top_uniform]

            sketch_uniform = np.array([run["metrics." + f"acc_{thresh}_uniform-dataloader_idx_{4}"] for thresh in thresholds])
            sketch_proto = run["metrics." + f"proto_acc/dataloader_idx_4"]
            sketch_knn = run["metrics." + f"knn_acc/dataloader_idx_4"]
            #sketch_uniform = sketch_uniform[top_uniform]

            shift_uniform = .25 * (r_uniform + a_uniform + v2_uniform + sketch_uniform)
            shift_proto = .25 * (r_proto + a_proto + v2_proto + sketch_proto)
            shift_knn = .25 * (r_knn + a_knn + v2_knn + sketch_knn)

            if n == 0:
                proto_val_.append(val_proto)
                knn_val_.append(val_knn)
                proto_shift_.append(shift_proto)
                knn_shift_.append(shift_knn)


            # Generate color smoothly from blue to red
            cmap = cm.get_cmap('coolwarm')
            cmap_soup = cm.get_cmap("coolwarm")
            curve_colors = {
                "2": cmap(0.0),
                "4": cmap(0.33),
                "8": cmap(0.66),
                "16": cmap(1.),
            }
            def darken_color(color, factor=0.9):
                return tuple(np.array(color[:3]) * factor) + (color[3],)
            soup_colors = {
                "2": darken_color(cmap_soup(0.0)),
                "4": darken_color(cmap_soup(0.33)),
                "8": darken_color(cmap_soup(0.66)),
                "16": darken_color(cmap_soup(1.)),
            }
            curve_color = curve_colors[str(n_shot)]
            soup_color = soup_colors[str(n_shot)]

            # Marker sizes increase from left to right
            marker_sizes = np.linspace(50, 200, val_scores.shape[0])

            # Plot curves with increasing marker sizes
            for i in range(val_scores.shape[0]):
                for j in range(val_scores.shape[1]):
                    if n == 0:
                        ax.plot(
                            val_scores[i:i + 2, j] * 100,
                            shift_scores[i:i + 2, j] * 100,
                            color=curve_color,
                            linestyle="-",
                            linewidth=0.5,
                            marker='o',
                            markersize=marker_sizes[i] / 40,  # scale down for better appearance
                            markeredgecolor='k',
                            alpha=1
                        )
                if n == 1:
                    ax.plot(
                        val_uniform[i:i+2] * 100, shift_uniform[i:i+2] * 100,
                        marker='D', alpha=1, linestyle='-', linewidth=2,
                        markeredgecolor='k', color=soup_color, markersize=marker_sizes[i] / 30
                    )

    #ax.plot(proto_val_, proto_shift_, marker='*', color='green', markersize=15, linestyle='-', linewidth=.5, label="proto")
    #ax.plot(knn_val_, knn_shift_, marker='*', color='purple', markersize=15, linestyle='-', linewidth=.5, label="knn")
    for j in range(len(proto_val_)):
        ax.scatter(proto_val_[j] * 100, proto_shift_[j] * 100, marker='*', color=curve_colors[str(2**(j+1))], s=300, zorder=10)
        ax.scatter(knn_val_[j] * 100, knn_shift_[j] * 100, marker='h', color=curve_colors[str((2**(j+1)))], s=200, zorder=10)

    # Title and labels
    # ax.set_title("Comparison of Validation vs Shift Scores", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Validation accuracy (%)", fontsize=24, labelpad=15)
    if id == 0:
        ax.set_ylabel("Average accuracy over four distribution shifts (%)", fontsize=24, labelpad=15)
    if id == 1:
        ax.set_xlim(0.65*100, 0.80*100)
        ax.set_ylim(0.58*100, 0.69*100)
        #ax.text(70, 66, "DINOv2 ViT-B/14 Reg", fontsize=12, color='black', weight='bold')
    else:
        ax.set_xlim(0.52 * 100, 0.75 * 100)
        ax.set_ylim(0.34 * 100, 0.525 * 100)
        #ax.text(63, 50, "DINOv2 ViT-S/14", fontsize=12, color='black', weight='bold')


    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.5)

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=28)

    # Tight layout
    plt.tight_layout()
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

    # Display the plot
    # Define custom legend elements
    custom_lines = [
        Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=8, label='Individual Adapters'),
        Line2D([0], [0], color='gray', marker='D', linestyle='None', markersize=10, label='CAT Adapter'),
        Line2D([0], [0], color='gray', marker='*', linestyle='None', markersize=14, label='Prototypical'),
        Line2D([0], [0], color='gray', marker='h', linestyle='None', markersize=14, label='KNN'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.0), lw=2, label='2-shot'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.33), lw=2, label='4-shot'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.66), lw=2, label='8-shot'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(1.0), lw=2, label='16-shot'),
        Line2D([], [], linestyle='None', label=["DINOv2 ViT-S/14", "DINOv2 ViT-B/14-Register"][id], linewidth=0)
    ]

    ax.legend(
        handles=custom_lines,
        loc='lower right',
        fontsize=24,
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='gray'
    )
    plt.savefig(f"figures/dinov2_robustness_{id}.png")



def ablate_K(clip="ViT-B/32", dinov2="dinov2_vits14", id=0):

    experiment = fr"ablate_k {clip} {dinov2}"
    data = pd.read_csv(os.path.join("results", f"{experiment.replace(' ', '_').replace('/', '_')}.csv"))

    for i, run in data.iterrows():
        # hard code
        if i == 1:
            name = "ViT-B/32"
        else:
            name = "dinov2_vits14"
        scores = [[[run["metrics." + f"acc_{j}_uniform-dataloader_idx_{d}_iteration_{k}"] for k in range(10)] for j in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]] for d in range(5)]
        scores = np.array(scores) * 100
        val_scores = scores[0]
        shift_scores = np.mean(scores[1:], axis=0)
        if i == id:
            break

    # Transpose to (10, 11) to iterate over iterations
    x = val_scores.T  # shape (10, 11)
    y = shift_scores.T  # shape (10, 11)

    fig, ax = plt.subplots(figsize=(15, 10))

    # Colormap and normalization
    cmap = cm.coolwarm
    norm = Normalize(vmin=0, vmax=x.shape[0] - 1)
    colors = cmap(np.linspace(0, 1, x.shape[0]))

    # Plot each iteration line
    for i in range(x.shape[0]):
        points = np.array([x[i], y[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=[colors[i]] * len(segments), linewidths=2)
        ax.add_collection(lc)

    # Enlarge labels and title ONLY
    ax.set_xlabel("Validation accuracy (%)", fontsize=20)
    ax.set_ylabel("Average accuracy over four distribution shifts (%)", fontsize=20)
    prefix = "CLIP" if name.startswith("ViT") else "DINOv2"
    name = "ViT-B/32" if prefix == "CLIP" else "ViT-S/14"  # hard coded
    ax.set_title(f"{prefix} {name}", fontsize=20)

    ax.autoscale()
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=16)

    cax = fig.add_axes([0.245, 0.2, 0.02, 0.3])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('K (Number of adapters)', fontsize=20)
    cb.set_ticks([0, x.shape[0] - 1])
    cb.set_ticklabels(['1', '10'])
    cb.ax.tick_params(labelsize=12)

    #plt.show()
    plt.savefig(f"figures/ablate_k_{name.replace('/', '-')}.png", bbox_inches='tight')

def ablate_mask():
    experiment = fr"ablate_mask"
    data = pd.read_csv(os.path.join("results", f"{experiment.replace(' ', '_').replace('/', '_')}.csv"))
    for i, run in data.iterrows():
        no_mask = run["params.no_mask"]
        avg_3 = np.mean([run["metrics." + f"test_acc_0.3-dataloader_idx_1_{j}"] for j in range(5)]) * 100
        avg_5 = np.mean([run["metrics." + f"test_acc_0.5-dataloader_idx_1_{j}"] for j in range(5)]) * 100
        avg_7 = np.mean([run["metrics." + f"test_acc_0.7-dataloader_idx_1_{j}"] for j in range(5)]) * 100
        avg_10 = np.mean([run["metrics." + f"test_acc_1-dataloader_idx_1_{j}"] for j in range(5)]) * 100

        soup_3 = run["metrics." + "acc_0.3_uniform-dataloader_idx_1"] * 100
        soup_5 = run["metrics." + "acc_0.5_uniform-dataloader_idx_1"] * 100
        soup_7 = run["metrics." + "acc_0.7_uniform-dataloader_idx_1"] * 100
        soup_10 = run["metrics." + "acc_1_uniform-dataloader_idx_1"] * 100

        print(f"no_mask: {no_mask}, n-shot: {run['params.n_shot']}")
        print({
            "avg r=0.3": avg_3,
            "avg r=0.5": avg_5,
            "avg r=0.7": avg_7,
            "avg r=1.0": avg_10,
            "soup r=0.3": soup_3,
            "soup r=0.5": soup_5,
            "soup r=0.7": soup_7,
            "soup r=1.0": soup_10,
        })
        print("---------------------------------")


def ratio(model, dataset_identifiers=datasets[:-1]):
    metric_ = {}
    metric_3 = {}
    metric_6 = {}
    metric_9 = {}
    soup_ = {}
    soup_3 = {}
    soup_6 = {}
    soup_9 = {}
    for dataset_identifier in dataset_identifiers:
        metric_[dataset_identifier] = []
        soup_[dataset_identifier] = []
        metric_9[dataset_identifier] = []
        metric_6[dataset_identifier] = []
        metric_3[dataset_identifier] = []
        soup_9[dataset_identifier] = []
        soup_6[dataset_identifier] = []
        soup_3[dataset_identifier] = []
        for n_shot in [2, 4, 8, 16]:
            metric_name = "test_acc_{}-dataloader_idx_1_{}"
            soup_name = "acc_{}_uniform-dataloader_idx_1"
            if dataset_identifier == "imagenet":
                metric_name = "val_acc_{}-dataloader_idx_0_{}"
                soup_name = "acc_{}_uniform-dataloader_idx_0"
            scores = pick("clip", model, dataset_identifier, metric_name, n_shot)
            soup_scores = pick("clip", model, dataset_identifier, soup_name, n_shot, use_idxs=False)
            val_scores = pick("clip", model, dataset_identifier, "val_acc_{}-dataloader_idx_0_{}", n_shot)
            soup_val_scores = pick("clip", model, dataset_identifier, "acc_{}_uniform-dataloader_idx_0", n_shot, use_idxs=False)

            top_idx = np.argmax(val_scores, axis=0)
            idx_9 = np.array([9] * N)
            idx_6 = np.array([6] * N)
            idx_3 = np.array([3] * N)

            avg = np.array([scores[top_idx[j], j] for j in range(N)]).mean()  #scores[top_idx].mean()
            avg_9 = np.array([scores[idx_9[j], j] for j in range(N)]).mean()
            avg_6 = np.array([scores[idx_6[j], j] for j in range(N)]).mean()
            avg_3 = np.array([scores[idx_3[j], j] for j in range(N)]).mean()

            top_idx_soup = np.argmax(soup_val_scores)
            soup = soup_scores[top_idx_soup]
            score_soup_9 = soup_scores[9]
            score_soup_6 = soup_scores[6]
            score_soup_3 = soup_scores[3]


            metric_[dataset_identifier].append(avg)
            metric_3[dataset_identifier].append(avg_3)
            metric_6[dataset_identifier].append(avg_6)
            metric_9[dataset_identifier].append(avg_9)

            soup_[dataset_identifier].append(soup)
            soup_3[dataset_identifier].append(score_soup_3)
            soup_6[dataset_identifier].append(score_soup_6)
            soup_9[dataset_identifier].append(score_soup_9)

    avg = np.mean(list(metric_.values()), axis=0)
    avg_3 = np.mean(list(metric_3.values()), axis=0)
    avg_6 = np.mean(list(metric_6.values()), axis=0)
    avg_9 = np.mean(list(metric_9.values()), axis=0)

    soup_avg = np.mean(list(soup_.values()), axis=0)
    soup_avg_3 = np.mean(list(soup_3.values()), axis=0)
    soup_avg_6 = np.mean(list(soup_6.values()), axis=0)
    soup_avg_9 = np.mean(list(soup_9.values()), axis=0)

    out = {}
    for i, n_shot in enumerate([2, 4, 8, 16]):
        out[n_shot] = {}
        out[n_shot]["best_avg"] = avg[i]
        out[n_shot]["best_avg_3"] = avg_3[i]
        out[n_shot]["best_avg_6"] = avg_6[i]
        out[n_shot]["best_avg_9"] = avg_9[i]
        out[n_shot]["soup_avg"] = soup_avg[i]
        out[n_shot]["soup_avg_3"] = soup_avg_3[i]
        out[n_shot]["soup_avg_6"] = soup_avg_6[i]
        out[n_shot]["soup_avg_9"] = soup_avg_9[i]

    print(out)

os.makedirs("figures", exist_ok=True)
ratio("ViT-B/32", datasets)
dinov2("dinov2_vits14", datasets)
clip("ViT-B/32", datasets)
robustness_dinov2(0)
robustness_dinov2(1)
robustness_clip(0)
robustness_clip(1)
ablate_K(id=1)
ablate_K(id=0)
ablate_mask()

"""experiment_names = [
    "clip_soup_v3 ViT-B/32 16",
    "clip_soup_v3 ViT-B/32 8",
    "clip_soup_v3 ViT-B/32 4",
    "clip_soup_v3 ViT-B/32 2",
    "clip_soup_v3 ViT-B/32 16",
    "dinov2_soup_v3 dinov2_vits14 16",
    "dinov2_soup_v3 dinov2_vits14 8",
    "dinov2_soup_v3 dinov2_vits14 4",
    "dinov2_soup_v3 dinov2_vits14 2",
    "ablate_mask",
    "ablate_k ViT-B/32 dinov2_vits14",
    "dinov2_soup_imagenet_v3 dinov2_vitb14_reg",
    "dinov2_soup_imagenet_v3 dinov2_vits14",
    "clip_soup_imagenet_v3 ViT-B/16",
    "clip_soup_imagenet_v3 ViT-B/32",
    "ablate_dino_arch"
]
os.makedirs("results", exist_ok=True)
for experiment_name in experiment_names:
    runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="pandas")
    if 'run_id' not in runs.columns:
        print(f"No run_id in {experiment_name}, skipping.")
        continue

    dataset_ids = []
    dinov2_models = []
    clip_models = []

    for run_id in runs["run_id"]:
        run = mlflow.get_run(run_id)
        params = run.data.params
        dataset_ids.append(params.get("dataset_identifier", None))
        dinov2_models.append(params.get("dinov2_model", None))
        clip_models.append(params.get("clip_model", None))

    runs["params.dataset_identifier"] = dataset_ids
    runs["params.dinov2_model"] = dinov2_models
    runs["params.clip_model"] = clip_models

    save = os.path.join("results", f"{experiment_name.replace(' ', '_').replace('/', '_')}.csv")
    runs.to_csv(save, index=False)"""