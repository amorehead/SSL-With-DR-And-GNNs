# Standard libraries
import os
# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
import torch.nn as nn
# PyTorch Geometric
import torch_geometric
import torch_geometric.nn as geom_nn
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = '/home/alexm/data/'
# Path to the folder where the pretrained models are saved
CHECKPOINT_BASE_PATH = '/home/alexm/savedmodels/'
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

# Populate list of possible GNN models to use
gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

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


class GNNModel(nn.Module):
    def __init__(
            self,
            c_in,
            c_hidden,
            c_out,
            num_layers=2,
            layer_name="GCN",
            dp_rate=0.1,
            **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of hidden layers
            dp_rate: Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True), nn.Dropout(dp_rate)]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input features per node
        """
        return self.layers(x)


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        transfer_learning = True  # If transfer learning, use a much lower initial learning rate
        lr = 1e-4 if transfer_learning else 1e-1
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)


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


# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))


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
