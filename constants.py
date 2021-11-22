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

REDUCE_METHODS = ['pca', 'tsne', 'umap']
COLOR_PALETTE = ["#52D1DC", "#8D0004", "#845218", "#563EAA", "#E44658", "#63C100", "#FF7800"]