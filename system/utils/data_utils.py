import numpy as np
import os
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime


def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join(os.curdir, 'dataset', dataset, 'train')
    else:
        data_dir = os.path.join(os.curdir, 'dataset', dataset, 'test')

    # file = data_dir + str(idx) + '.npz'
    file = os.path.join(data_dir, str(idx) + '.npz')
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def save_metrics_plots(server, run_id=None, dataset=None, algorithm=None):
    # Use datetime, dataset, and algorithm as run_id if not provided
    if run_id is None:
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ds_str = dataset if dataset is not None else getattr(server, 'dataset', 'unknown')
        algo_str = algorithm if algorithm is not None else getattr(server, 'algorithm', 'unknown')
        run_id = f"{dt_str}_{ds_str}_{algo_str}"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Plot Train Loss
    plt.figure()
    plt.plot(server.rs_train_loss, label="Train Loss")
    plt.xlabel("Evaluation Round")
    plt.ylabel("Loss")
    plt.title("Averaged Train Loss")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_train_loss.png"))
    plt.close()

    # Plot Test Accuracy
    plt.figure()
    plt.plot(server.rs_test_acc, label="Test Accuracy")
    plt.xlabel("Evaluation Round")
    plt.ylabel("Accuracy")
    plt.title("Averaged Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_test_acc.png"))
    plt.close()

    # Plot Test AUC
    plt.figure()
    plt.plot(server.rs_test_auc, label="Test AUC")
    plt.xlabel("Evaluation Round")
    plt.ylabel("AUC")
    plt.title("Averaged Test AUC")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{run_id}_test_auc.png"))
    plt.close()

    # Plot selected clients as a heatmap (optional)
    if server.selected_clients_per_round:
        # Create a 2D array: rows=rounds, cols=clients, 1 if selected, 0 otherwise
        num_rounds = len(server.selected_clients_per_round)
        num_clients = server.num_clients
        selection_matrix = np.zeros((num_rounds, num_clients))
        for r, selected in enumerate(server.selected_clients_per_round):
            for cid in selected:
                selection_matrix[r, cid] = 1
        plt.figure(figsize=(12, 6))
        plt.imshow(selection_matrix, aspect='auto', cmap='Greys')
        plt.xlabel("Client ID")
        plt.ylabel("Round")
        plt.title("Client Selection per Round")
        plt.colorbar(label="Selected (1) / Not Selected (0)")
        # Set integer ticks for axes, but limit number for readability
        max_xticks = 20
        max_yticks = 20
        xtick_step = max(1, num_clients // max_xticks)
        ytick_step = max(1, num_rounds // max_yticks)
        plt.xticks(np.arange(0, num_clients, xtick_step))
        plt.yticks(np.arange(0, num_rounds, ytick_step))
        plt.savefig(os.path.join(plot_dir, f"{run_id}_client_selection.png"))
        plt.close()

