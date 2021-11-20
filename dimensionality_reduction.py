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
cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Cora')
citeseer_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Citeseer')

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
# Files to download
pretrained_files = ["NodeLevelMLP.ckpt", "NodeLevelGNN.ckpt", "GraphLevelGraphConv.ckpt"]
# e = download_pretrained_weights(exist_ok, CHECKPOINT_PATH, base_url, pretrained_files)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RESULT_DIR_NAME = 'results'
result_dir = RESULT_DIR_NAME
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def plot_tsne_2d(data, labels, save_path, n_components=2):
    tsne = TSNE(n_components=n_components, init='pca', perplexity=40, random_state=0)
    tsne_res = tsne.fit_transform(data)

    v = pd.DataFrame(data,columns=[str(i) for i in range(data.shape[1])])
    v['y'] = labels
    v['label'] = v['y'].apply(lambda i: str(i))
    v["t1"] = tsne_res[:,0]
    v["t2"] = tsne_res[:,1]

    sns.scatterplot(
        x="t1", y="t2",
        hue="y",
        palette=sns.color_palette(["#52D1DC", "#8D0004", "#845218","#563EAA", "#E44658", "#63C100", "#FF7800"]),
        legend=False,
        data=v,
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('') 
    plt.ylabel('')
    plt.savefig(save_path, dpi=300)


def visualize_hidden_space(model_name, dataset, result_dir, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Check whether pretrained model exists.
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading to begin fine-tuning (e.g., for Cora)...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        raise IOError("NOT found the pretrained model", pretrained_filename)

    # Test best model on the test set
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    hidden_features = model.extract_features(batch).detach().numpy()
    save_path = os.path.join(result_dir, model_name + '-tsne.png')
    viz_result = plot_tsne_2d(hidden_features, batch.y.detach().numpy(), save_path)
    return model, viz_result


node_gnn_model, node_gnn_result = visualize_hidden_space(
    model_name="GNN", layer_name="GCN", dataset=cora_dataset,
    result_dir=result_dir,
    c_hidden=16, num_layers=2, dp_rate=0.1
)
print(f'\nSemi-supervised node classification results on the Cora dataset using a GCN:')