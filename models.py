# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
import torch.nn as nn
# PyTorch Geometric
import torch_geometric.nn as geom_nn
# TorchMetrics
import torchmetrics
from torchmetrics import Precision, Recall, F1

from constants import GNN_LAYER_BY_NAME


class GNNModel(nn.Module):
    def __init__(
            self,
            c_in,
            c_hidden,
            c_out,
            num_layers=2,
            model_name="GCN",
            dp_rate=0.1,
            **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            model_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = GNN_LAYER_BY_NAME[model_name]

        hidden_layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            hidden_layers += [
                gnn_layer(in_channels, out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.out_layers = nn.ModuleList([gnn_layer(in_channels, c_out, **kwargs)])

    def extract_features(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.hidden_layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

    def forward(self, x, edge_index):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        x = self.extract_features(x, edge_index)
        for layer in self.out_layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()
        self.metric_precision = Precision(num_classes=model_kwargs['c_out'], average='macro')
        self.metric_recall = Recall(num_classes=model_kwargs['c_out'], average='macro')
        self.metric_f1 = F1(num_classes=model_kwargs['c_out'], average='macro')
        self.lr = model_kwargs.get('learning_rate', 1e-1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x

    def extract_features(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model.extract_features(x, edge_index)
        return x

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = self.compute_loss(x, batch, 'train')
        acc = self.compute_metric(x, batch, 'train', 'accuracy')
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = self.compute_loss(x, batch, 'val')
        acc = self.compute_metric(x, batch, 'val', 'accuracy')
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x = self.forward(batch)
        acc = self.compute_metric(x, batch, 'test', 'accuracy')
        self.log("test_acc", acc)

    def get_data_mask(self, data, mode):
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode
        return mask

    def compute_loss(self, x, data, mode):
        mask = self.get_data_mask(data, mode)
        loss = self.loss_module(x[mask], data.y[mask])
        return loss

    def compute_metric(self, x, data, mode, metric_name):
        mask = self.get_data_mask(data, mode)
        preds = x[mask].argmax(dim=-1)
        target = data.y[mask]
        if metric_name == 'accuracy':
            metric = torchmetrics.functional.accuracy(preds, target)
        elif metric_name == 'precision':
            metric = self.metric_precision(preds, target)
        elif metric_name == 'recall':
            metric = self.metric_recall(preds, target)
        elif metric_name == 'f1':
            metric = self.metric_f1(preds, target)
        return metric
