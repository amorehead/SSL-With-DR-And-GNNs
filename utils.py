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

from autoencoder import AutoEncoder
# Project utilities
from constants import (CHECKPOINT_BASE_PATH, CLASS_NAMES, COLOR_PALETTE,
                       DATASET_PATH, RAND_SEED, REDUCE_METHODS, RESULT_DIR)
from models import NodeLevelGNN

os.makedirs(DATASET_PATH, exist_ok=True)


def get_experiment_name(dataset_name, model_name, reduce_method, seed, model_kwargs):
    c_hidden = model_kwargs['c_hidden']
    num_layers = model_kwargs['num_layers']

    exp_name = f'{dataset_name}-{model_name}-{c_hidden}x{num_layers}'
    if reduce_method[0] and reduce_method[1]:
        exp_name += f'-{reduce_method[0]}_{reduce_method[1]}'
    if seed is not None:
        exp_name += f'-seed_{seed}'
    return exp_name


def get_dataset(dataset_name: str, reduce_method: tuple = ('', 0)):
    """Apply dimensionality reduction to the given PyTorch Geometric dataset, if requested."""
    if dataset_name == 'cora':
        dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Cora')
    elif dataset_name == 'citeseer':
        dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Citeseer')
    if reduce_method[0] in REDUCE_METHODS:
        # Reduce dimensionality of input dataset a priori
        reduced_x = project_nd(
            reduce_method[0],
            dataset.data.x,
            n_components=reduce_method[1],
            dataset_name=dataset_name)
        reduced_x = torch.Tensor(reduced_x).detach()
        dataset.data.__setattr__('x', reduced_x)
    return dataset


def extract_hidden_features(dataset_name, model_name, reduce_method, seed, **model_kwargs):
    pl.seed_everything(RAND_SEED)
    dataset = get_dataset(dataset_name, reduce_method)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Check whether pretrained model exists.
    experiment_name = get_experiment_name(dataset_name, model_name, reduce_method, seed, model_kwargs)
    pretrained_filename = os.path.join(CHECKPOINT_BASE_PATH, f'{experiment_name}.ckpt')
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        raise IOError("NOT found the pretrained model", pretrained_filename)

    # Test best model on the test set
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    model.eval()
    with torch.no_grad():
        hidden_features = model.extract_features(batch).detach().numpy()
    labels = batch.y.detach().numpy()
    return hidden_features, labels


def print_number_of_parameters(dataset_name, model_name, reduce_method, seed, **model_kwargs):
    experiment_name = get_experiment_name(dataset_name, model_name, reduce_method, seed, model_kwargs)
    pretrained_filename = os.path.join(CHECKPOINT_BASE_PATH, f'{experiment_name}.ckpt')
    if os.path.isfile(pretrained_filename):
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        raise IOError("NOT found the pretrained model", pretrained_filename)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{dataset_name}] # of trainable params of {model_name} {reduce_method} = {num_params}')


def project_nd(method, data, **kwargs):
    n_components = kwargs.get('n_components', 2)
    dataset_name = kwargs.get('dataset_name', '')
    seed = kwargs.get('seed', RAND_SEED)
    if method == 'tsne':
        tsne = TSNE(n_components=n_components, init='pca', perplexity=40, random_state=RAND_SEED)
        embedding = tsne.fit_transform(data)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=RAND_SEED)
        embedding = reducer.fit_transform(data)
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(data)
    elif method == 'ae':
        checkpoint_path = os.path.join(CHECKPOINT_BASE_PATH, f'{dataset_name}-ae-{n_components}-seed_{seed}.ckpt')
        ae = AutoEncoder.load_from_checkpoint(checkpoint_path)
        ae.eval()
        with torch.no_grad():
            embedding = ae.encoder(data)
    else:
        raise ValueError('invalid method', method)
    return embedding


def plot_hidden_features(ax, method, data, labels, dataset_name, title, save_path, **kwargs):
    embedding = project_nd(method, data, **kwargs)
    plot_embedding_2D(ax, data, labels, embedding, dataset_name, title, save_path)


def plot_embedding_2D(ax, data, labels, embedding, dataset_name, title, save_path):
    v = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    v['y'] = labels
    v['label'] = v['y'].apply(lambda i: CLASS_NAMES[dataset_name][i])
    v["t1"] = embedding[:, 0]
    v["t2"] = embedding[:, 1]
    num_classes = len(np.unique(labels))
    palette = sns.color_palette(COLOR_PALETTE)[:num_classes]

    sns.scatterplot(
        x="t1", y="t2",
        hue="label",
        palette=palette,
        legend=None,
        data=v,
        ax=ax,
        size=0,
        edgecolor=None,
        linewidth=0,
    )
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=20)


def print_last_result(result_dict):
    for set_name in list(result_dict):
        for metric_name in list(result_dict[set_name]):
            print(f'{set_name} {metric_name}: {result_dict[set_name][metric_name][-1] * 100:4.2f}')


def write_last_result(dataset_name, model_name, reduce_method, seed, result_dict):
    test = result_dict['test']
    t = f'{dataset_name},{model_name}'
    t += ',' + f'{reduce_method[0]}_{reduce_method[1]}'
    t += ',' + f'{seed}'
    t += ',' + f'{test["accuracy"][-1] * 100:.2f}'
    t += ',' + f'{test["precision"][-1] * 100:.2f}'
    t += ',' + f'{test["recall"][-1] * 100:.2f}'
    t += ',' + f'{test["f1"][-1] * 100:.2f}'
    file_path = os.path.join(RESULT_DIR, 'test_results-individuals.csv')
    with open(file_path, 'a') as f:
        if os.stat(file_path).st_size == 0:
            header = f'dataset,model,reduce_method,seed,accuracy,precision,recall,f1\n'
            f.write(header)
        f.write(f'{t}\n')


def get_mean_std(numbers):
    mean = np.mean(numbers) * 100
    std = np.std(numbers) * 100
    return f'{mean:.2f} ({std:.2f})'


def write_summary_results(dataset_name, model_name, reduce_method, seeds, results, res_format='csv'):
    delimiter = ' & ' if res_format == 'latex' else ','

    seeds_text = '-'.join(map(str, seeds))
    test = results['test']
    t = f'{dataset_name}{delimiter}{model_name}'
    t += delimiter + f'{reduce_method[0]}_{reduce_method[1]}'
    t += delimiter + f'{seeds}'
    t += delimiter + get_mean_std(test['accuracy'])
    t += delimiter + get_mean_std(test['precision'])
    t += delimiter + get_mean_std(test['recall'])
    t += delimiter + get_mean_std(test['f1'])
    file_path = os.path.join(RESULT_DIR, f'test_results-summary.{res_format}')
    with open(file_path, 'a') as f:
        if os.stat(file_path).st_size == 0:
            header = f'dataset{delimiter}model{delimiter}reduce_method{delimiter}seeds{delimiter}accuracy{delimiter}precision{delimiter}recall{delimiter}f1\n'
            f.write(header)
        f.write(f'{t}\n')


def write_ae_error(dataset_name, seed, error):
    t = f'{dataset_name}'
    t += ',' + f'{seed}'
    t += ',' + f'{error:.4f}'
    file_path = os.path.join(RESULT_DIR, 'test_results-AEs.csv')
    with open(file_path, 'a') as f:
        if os.stat(file_path).st_size == 0:
            header = f'dataset,seed,error\n'
            f.write(header)
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
