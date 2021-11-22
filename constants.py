# System utilities
import os

# PyTorch
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
# Dimensionality reduction utilities
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Random seed
RAND_SEED = 8735

# GPU count
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

# Populate list of possible GNN models to use
GNN_LAYER_BY_NAME = {"MLP": nn.Linear, "GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

# Path to the folder where the DATASETS are/should be downloaded
BASE_URL = '.'
DATASET_PATH = f'{BASE_URL}/data/'

# Path to the folder where the pretrained models are saved
CHECKPOINT_BASE_PATH = f'{BASE_URL}/savedmodels/'

# Make results directory
RESULT_DIR = os.path.join('results')
os.makedirs(RESULT_DIR, exist_ok=True)

# Names of dataset classes
CLASS_NAMES = {
    'cora': ['CB', 'GA', 'NN', 'PM', 'RL', 'RLL', 'Theory'],
    'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
}

# Whether to apply dimensionality reduction to the selected input dataset before training of any models
DR_TRAIN_DATA = True
DR_TRAIN_DATA_METHOD = 'tsne'


# Need to make a copy of this function to avoid circular imports in this Python project...for now....
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


def prep_dataset(dataset: torch_geometric.data.Data, reduce_training_data_dim: bool, reduce_method: str):
    """Apply dimensionality reduction to the given PyTorch Geometric dataset, if requested."""
    if reduce_training_data_dim:
        # Reduce dimensionality of input dataset a priori
        dataset.data.__setattr__('x', project_2D(reduce_method, dataset.data.x))
    return dataset


# Datasets
DATASETS = {
    'cora': prep_dataset(
        torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Cora'),
        DR_TRAIN_DATA,
        DR_TRAIN_DATA_METHOD
    ),
    'citeseer': prep_dataset(
        torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Citeseer'),
        DR_TRAIN_DATA,
        DR_TRAIN_DATA_METHOD)
}
