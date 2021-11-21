# Standard libraries
import os

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
# PyTorch Geometric
import torch_geometric

from models import NodeLevelGNN
from utils import download_pretrained_weights

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
# Path to the folder where the datasets are/should be downloaded
BASE_URL = '.'
DATASET_PATH = f'{BASE_URL}/data/'
# Path to the folder where the pretrained models are saved
CHECKPOINT_BASE_PATH = f'{BASE_URL}/savedmodels/'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_BASE_PATH, "GNNs/")

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Create checkpoint path if it doesn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Download Cora and Citeseer datasets
datasets = {
    'cora': torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Cora'),
    'citeseer': torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Citeseer'),
}

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
# Files to download
pretrained_files = ["NodeLevelMLP.ckpt", "NodeLevelGNN.ckpt", "GraphLevelGraphConv.ckpt"]
# e = download_pretrained_weights(exist_ok, CHECKPOINT_PATH, base_url, pretrained_files)


import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RESULT_DIR_NAME = 'results'
result_dir = RESULT_DIR_NAME
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def extract_hidden_features(model_name, dataset, result_dir, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Check whether pretrained model exists.
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        raise IOError("NOT found the pretrained model", pretrained_filename)

    # Test best model on the test set
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    hidden_features = model.extract_features(batch).detach().numpy()
    return model, hidden_features, batch.y.detach().numpy()

def plot_hidden_features(method, data, labels, title, save_path, **kwargs):
    embedding = project_2d(method, data, **kwargs)
    fig = plot_embedding_2d(data, labels, embedding, title, save_path)

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

def plot_embedding_2d(data, labels, embedding, title, save_path):
    v = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    v['y'] = labels
    v['label'] = v['y'].apply(lambda i: str(i))
    v["t1"] = embedding[:,0]
    v["t2"] = embedding[:,1]

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="t1", y="t2",
        hue="y",
        palette=sns.color_palette(["#52D1DC", "#8D0004", "#845218","#563EAA", "#E44658", "#63C100", "#FF7800"]),
        legend=True,
        data=v,
        ax=ax,
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('') 
    plt.ylabel('')
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    return fig

def main(model_name, dataset_name, methods, **kwargs):
    node_gnn_model, hidden_features, labels = extract_hidden_features(
        model_name=model_name, layer_name="GCN", dataset=datasets[dataset_name],
        result_dir=result_dir,
        c_hidden=16, num_layers=2, dp_rate=0.1
    )
    for method_name in methods:
        kwargs = methods[method_name]
        title = f'{method_name} projection of {model_name} on {dataset_name}'
        save_path = os.path.join(result_dir, f'{dataset_name}-{model_name}-{method_name}.png')
        viz_result = plot_hidden_features(method_name, hidden_features, labels, title, save_path, **kwargs)
        print(f'Visualizing hidden features of a GCN on the {dataset_name} dataset: {method_name}')

if __name__ == '__main__':
    methods = {
        'tsne': {
            'n_components': 2,
        },
        'umap': {},
        'pca': {},
    }
    main('GNN', 'cora', methods)