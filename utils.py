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
import torch_geometric
# Dimensionality reduction utilities
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Project utilities
from constants import DATASETS, CHECKPOINT_BASE_PATH, RAND_SEED, CLASS_NAMES
from models import NodeLevelGNN


def extract_hidden_features(dataset_name, model_name, **model_kwargs):
    pl.seed_everything(RAND_SEED)
    dataset = DATASETS[dataset_name]
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Check whether pretrained model exists.
    experiment_name = f'{dataset_name}-{model_name}'
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
    if method == 'tsne':
        tsne = TSNE(n_components=kwargs.get('n_components', 2), init='pca', perplexity=40, random_state=0)
        embedding = tsne.fit_transform(data)
    elif method == 'umap':
        reducer = umap.UMAP(random_state=8735)
        embedding = reducer.fit_transform(data)
    elif method == 'pca':
        pca = PCA(n_components=kwargs.get('n_components', 2))
        embedding = pca.fit_transform(data)
    else:
        raise ValueError('invalid method', method)
    return embedding


def plot_hidden_features(method, data, labels, dataset_name, title, save_path, **kwargs):
    embedding = project_2D(method, data, **kwargs)
    print('embedding', embedding.shape)
    # fig = plot_embedding_2D(data, labels, embedding, dataset_name, title, save_path)
    plot_embedding_2D(data, labels, embedding, dataset_name, title, save_path)


def plot_embedding_2D(data, labels, embedding, dataset_name, title, save_path):
    v = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    v['y'] = labels
    v['label'] = v['y'].apply(lambda i: CLASS_NAMES[dataset_name][i])
    v["t1"] = embedding[:, 0]
    v["t2"] = embedding[:, 1]
    num_classes = len(np.unique(labels))
    palette = sns.color_palette(
        ["#52D1DC", "#8D0004", "#845218", "#563EAA", "#E44658", "#63C100", "#FF7800"]
    )[:num_classes]

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
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))


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

