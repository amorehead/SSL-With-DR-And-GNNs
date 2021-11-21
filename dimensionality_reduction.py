import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ssl_with_dr_and_gnns import extract_hidden_features

RESULT_DIR = os.path.join('results')
os.makedirs(RESULT_DIR, exist_ok=True)

CLASS_NAMES = {
    'cora': ['CB', 'GA', 'NN', 'PM', 'RL', 'RLL', 'Theory'],
    'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
}


def plot_hidden_features(method, data, labels, dataset_name, title, save_path, **kwargs):
    embedding = project_2d(method, data, **kwargs)
    print('embedding', embedding.shape)
    # fig = plot_embedding_2d(data, labels, embedding, dataset_name, title, save_path)
    plot_embedding_2d(data, labels, embedding, dataset_name, title, save_path)


def project_2d(method, data, **kwargs):
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


def plot_embedding_2d(data, labels, embedding, dataset_name, title, save_path):
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


def main(dataset_name, model_name, methods, **kwargs):
    hidden_features, labels = extract_hidden_features(
        dataset_name=dataset_name, model_name=model_name,
        c_hidden=16, num_layers=2, dp_rate=0.1
    )
    for method_name in methods:
        kwargs = methods[method_name]
        title = f'{method_name} projection of {model_name} on {dataset_name}'
        save_path = os.path.join(RESULT_DIR, f'{dataset_name}-{model_name}-{method_name}.png')
        plot_hidden_features(method_name, hidden_features, labels,
                             dataset_name, title, save_path, **kwargs)
        print(f'Visualizing hidden features of the {model_name} on the {dataset_name} dataset: {method_name}')


if __name__ == '__main__':
    methods = {
        'tsne': {
            'n_components': 2,
        },
        'umap': {},
        'pca': {},
    }
    # dataset_name = 'citeseer' # 'cora', 'citeseer',
    # model_name = 'GCN' # 'MLP', 'GCN', 'GAT', 'GraphConv'
    # main(dataset_name, model_name, methods)
    for model_name in ['MLP', 'GCN', 'GAT', 'GraphConv']:
        for dataset_name in ['cora', 'citeseer']:
            main(dataset_name, model_name, methods)
