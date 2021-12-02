# Standard libraries
import os
import shutil

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
# PyTorch Geometric
import torch_geometric
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Project utilities
from constants import DATASET_PATH, CHECKPOINT_BASE_PATH, AVAIL_GPUS, RAND_SEED
from models import NodeLevelGNN
from utils import get_experiment_name, get_dataset, print_results

# Setting the seed
pl.seed_everything(RAND_SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Create checkpoint path if it doesn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)


def train_node_classifier(
        model_name, dataset_name, reduce_method, fine_tune, max_epochs, learning_rate, **model_kwargs):
    dataset = get_dataset(dataset_name, reduce_method)
    print(dataset.data.x.shape)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer
    experiment_name = get_experiment_name(dataset_name, model_name, reduce_method)
    root_dir = os.path.join(CHECKPOINT_BASE_PATH, experiment_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
            EarlyStopping('val_loss', patience=5),
        ],
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
        model = NodeLevelGNN(learning_rate=learning_rate, model_name=model_name,
                             c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs)
    trainer.fit(model, node_data_loader, node_data_loader)
    model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    saved_best_model_path = os.path.join(CHECKPOINT_BASE_PATH, f'{experiment_name}.ckpt')
    shutil.copy(trainer.checkpoint_callback.best_model_path, saved_best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    predicted_y = model.forward(batch)

    result = {
        "train": {},
        "val": {},
        "test": {},
    }
    print('test acc', test_result[0]["test_acc"])
    for set_name in ['train', 'val', 'test']:
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            result[set_name][metric_name] = model.compute_metric(predicted_y, batch, set_name, metric_name)

    return model, result


def train(dataset_name, model_name, reduce_method, fine_tune=False, max_epochs=500, learning_rate=1e-1):
    node_mlp_model, node_mlp_result = train_node_classifier(
        model_name=model_name, dataset_name=dataset_name, reduce_method=reduce_method, fine_tune=fine_tune,
        max_epochs=max_epochs, learning_rate=learning_rate,
        c_hidden=16, num_layers=2, dp_rate=0.1
    )

    print(
        f'\nSemi-supervised node classification results on the {dataset_name} dataset using an {model_name}, {reduce_method[0]} {reduce_method[1]}:')
    print_results(node_mlp_result)


if __name__ == '__main__':
    dataset_name = 'citeseer'  # 'cora', 'citeseer',
    model_name = 'GCN'  # 'MLP', 'GCN', 'GAT', 'GraphConv'
    reduce_method = ('', None)
    fine_tune = False
    lr = 1e-1
    train(dataset_name, model_name, reduce_method, fine_tune=fine_tune, max_epochs=500, learning_rate=lr)
    # for model_name in ['MLP', 'GCN', 'GAT', 'GraphConv']:
    #     for dataset_name in ['cora', 'citeseer']:
    #         train(dataset_name, model_name, reduced_method max_epochs=15000)
