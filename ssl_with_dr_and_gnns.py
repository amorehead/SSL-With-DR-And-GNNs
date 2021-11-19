# Standard libraries
import os
# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
# PyTorch Geometric
import torch_geometric
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint

from models import GNNModel, MLPModel, NodeLevelGNN
from utils import print_results

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

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
# Files to download
pretrained_files = ["NodeLevelMLP.ckpt", "NodeLevelGNN.ckpt", "GraphLevelGraphConv.ckpt"]

# Create checkpoint path if it doesn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Download Cora and Citeseer datasets
cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Cora')
citeseer_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Citeseer')

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
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


def train_node_classifier(model_name, dataset, fine_tune, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=500,
        progress_bar_refresh_rate=0
    )  # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    if os.path.isfile(fine_tune and pretrained_filename):
        # Currently, fine-tuning is only supported for the Cora dataset since pre-trained models were trained on Cora
        print("Found pretrained model, loading to begin fine-tuning (e.g., for Cora)...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(
            model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result


# Train, validate, and test on the Cora dataset
node_mlp_model, node_mlp_result = train_node_classifier(
    model_name="MLP", dataset=cora_dataset, fine_tune=True,
    c_hidden=16, num_layers=2, dp_rate=0.1
)

print(f'\nSemi-supervised node classification results on the Cora dataset using an MLP:')
print_results(node_mlp_result)

node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GCN", dataset=cora_dataset,
    fine_tune=True, c_hidden=16, num_layers=2, dp_rate=0.1
)
print(f'\nSemi-supervised node classification results on the Cora dataset using a GCN:')
print_results(node_gnn_result)

node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GAT", dataset=cora_dataset,
    fine_tune=True, c_hidden=16, num_layers=2, dp_rate=0.1
)
print(f'\nSemi-supervised node classification results on the Cora dataset using a GAT:')
print_results(node_gnn_result)

node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GraphConv", dataset=cora_dataset,
    fine_tune=True, c_hidden=16, num_layers=2, dp_rate=0.1
)
print(f'\nSemi-supervised node classification results on the Cora dataset using a GraphConv:')
print_results(node_gnn_result)

# Train, validate, and test on the Citeseer dataset
node_mlp_model, node_mlp_result = train_node_classifier(
    model_name="MLP", dataset=citeseer_dataset,
    fine_tune=False, c_hidden=16, num_layers=2, dp_rate=0.1
)

print(f'\nSemi-supervised node classification results on the CiteSeer dataset using an MLP:')
print_results(node_mlp_result)

node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GCN", dataset=citeseer_dataset,
    fine_tune=False, c_hidden=16, num_layers=2, dp_rate=0.1
)
print(f'\nSemi-supervised node classification results on the CiteSeer dataset using a GCN:')
print_results(node_gnn_result)

node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GAT", dataset=citeseer_dataset,
    fine_tune=False, c_hidden=16, num_layers=2, dp_rate=0.1
)
print(f'\nSemi-supervised node classification results on the CiteSeer dataset using a GAT:')
print_results(node_gnn_result)

node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GraphConv", dataset=citeseer_dataset,
    fine_tune=False, c_hidden=16, num_layers=2, dp_rate=0.1
)
print(f'\nSemi-supervised node classification results on the CiteSeer dataset using a GraphConv:')
print_results(node_gnn_result)
