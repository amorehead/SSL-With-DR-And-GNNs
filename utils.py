# System-level utilities
import os
# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# Plotting and numeric libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Lightning utilities
import pytorch_lightning as pl
import seaborn as sns
# PyTorch Geometric
import torch
import torch_geometric
# Dimensionality reduction utilities
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Project utilities
from constants import DATASET_PATH, CHECKPOINT_BASE_PATH, RAND_SEED, CLASS_NAMES, REDUCE_METHODS, COLOR_PALETTE, RESULT_DIR
from models import NodeLevelGNN


def get_experiment_name(dataset_name, model_name, reduce_method):
    exp_name = f'{dataset_name}-{model_name}'
    if reduce_method[0] and reduce_method[1]:
        exp_name += f'-{reduce_method[0]}_{reduce_method[1]}'
    return exp_name


def get_dataset(dataset_name: str, reduce_method: tuple):
    """Apply dimensionality reduction to the given PyTorch Geometric dataset, if requested."""
    if dataset_name == 'cora':
        dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Cora')
    elif dataset_name == 'citeseer':
        dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Citeseer')
    if reduce_method[0] in REDUCE_METHODS:
        # Reduce dimensionality of input dataset a priori
        reduced_x = project_2D(reduce_method[0], dataset.data.x, n_components=reduce_method[1])
        reduced_x = torch.Tensor(reduced_x)
        dataset.data.__setattr__('x', reduced_x)
    return dataset


def extract_hidden_features(dataset_name, model_name, reduce_method, **model_kwargs):
    pl.seed_everything(RAND_SEED)
    dataset = get_dataset(dataset_name, reduce_method)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Check whether pretrained model exists.
    experiment_name = get_experiment_name(dataset_name, model_name, reduce_method)
    pretrained_filename = os.path.join(CHECKPOINT_BASE_PATH, f'{experiment_name}.ckpt')
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        raise IOError("NOT found the pretrained model", pretrained_filename)

    # Test best model on the test set
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    hidden_features = model.extract_features(batch).detach().numpy()
    labels = batch.y.detach().numpy()
    return hidden_features, labels


def project_2D(method, data, **kwargs):
    n_components = kwargs.get('n_components', 2)
    if method == 'tsne':
        tsne = TSNE(n_components=n_components, init='pca', perplexity=40, random_state=RAND_SEED)
        embedding = tsne.fit_transform(data)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=RAND_SEED)
        embedding = reducer.fit_transform(data)
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(data)
    else:
        raise ValueError('invalid method', method)
    return embedding


def plot_hidden_features(method, data, labels, dataset_name, title, save_path, **kwargs):
    embedding = project_2D(method, data, **kwargs)
    fig = plot_embedding_2D(data, labels, embedding, dataset_name, title, save_path)


def plot_embedding_2D(data, labels, embedding, dataset_name, title, save_path):
    v = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    v['y'] = labels
    v['label'] = v['y'].apply(lambda i: CLASS_NAMES[dataset_name][i])
    v["t1"] = embedding[:, 0]
    v["t2"] = embedding[:, 1]
    num_classes = len(np.unique(labels))
    palette = sns.color_palette(COLOR_PALETTE)[:num_classes]

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="t1", y="t2",
        hue="label",
        palette=palette,
        legend='full',
        data=v,
        ax=ax,
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def print_results(result_dict):
    for set_name in list(result_dict):
        for metric_name in list(result_dict[set_name]):
            print(f'{set_name} {metric_name}: {result_dict[set_name][metric_name] * 100:4.2f}')


def write_results(dataset_name, model_name, reduce_method, result_dict):
    test = result_dict['test']
    t = f'{dataset_name},{model_name}'
    t += ',' + f'{reduce_method[0]}_{reduce_method[1]}'
    t += ',' + f'{test["accuracy"] * 100:.2f}'
    t += ',' + f'{test["precision"] * 100:.2f}'
    t += ',' + f'{test["recall"] * 100:.2f}'
    t += ',' + f'{test["f1"] * 100:.2f}'
    with open(os.path.join(RESULT_DIR, 'test_results.txt'), 'a') as f:
        f.write(f'{t}\n')


def download_pretrained_weights(exist_ok, checkpoint_path, base_url, pretrained_files):
    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(checkpoint_path, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=exist_ok)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print("Downloading %s..." % file_url)
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(
                    "Something went wrong. Please try to download the file from the GDrive folder,"
                    " or contact the author with the full output including the following error:\n",
                    e,
                )
