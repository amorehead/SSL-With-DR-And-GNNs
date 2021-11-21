# Standard libraries
import os

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
# PyTorch Geometric
import torch_geometric
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint

from models import NodeLevelGNN
from utils import print_results

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
# Path to the folder where the datasets are/should be downloaded
BASE_URL = '.'
DATASET_PATH = f'{BASE_URL}/data/'
# Path to the folder where the pretrained models are saved
CHECKPOINT_BASE_PATH = f'{BASE_URL}/savedmodels/'

# Setting the seed
pl.seed_everything(8735)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Create checkpoint path if it doesn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)

datasets = {
    'cora': torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Cora'),
    'citeseer': torch_geometric.datasets.Planetoid(root=DATASET_PATH, name='Citeseer'),
}


def train_node_classifier(model_name, dataset_name, fine_tune, max_epochs, **model_kwargs):
    pl.seed_everything(8735)
    dataset = datasets[dataset_name]
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer
    experiment_name = f'{dataset_name}-{model_name}'
    root_dir = os.path.join(CHECKPOINT_BASE_PATH, experiment_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=1
    )  # 0 because epoch size is 1
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_BASE_PATH, f'{experiment_name}.ckpt')
    if os.path.isfile(fine_tune and pretrained_filename):
        # Currently, fine-tuning is only supported for the Cora dataset since pre-trained models were trained on Cora
        print("Found pretrained model, loading to begin fine-tuning")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(8735)
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


def extract_hidden_features(dataset_name, model_name, **model_kwargs):
    pl.seed_everything(8735)
    dataset = datasets[dataset_name]
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


def train(dataset_name, model_name, max_epochs=500):
    node_mlp_model, node_mlp_result = train_node_classifier(
        model_name=model_name, dataset_name=dataset_name, fine_tune=True, max_epochs=max_epochs,
        c_hidden=16, num_layers=2, dp_rate=0.1
    )

    print(f'\nSemi-supervised node classification results on the {dataset_name} dataset using an {model_name}:')
    print_results(node_mlp_result)


if __name__ == '__main__':
    # dataset_name = 'citeseer' # 'cora', 'citeseer',
    # model_name = 'GCN' # 'MLP', 'GCN', 'GAT', 'GraphConv'
    # train(dataset_name, model_name, max_epochs=15000)
    for model_name in ['MLP', 'GCN', 'GAT', 'GraphConv']:
        for dataset_name in ['cora', 'citeseer']:
            train(dataset_name, model_name, max_epochs=15000)
