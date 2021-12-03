import os
import shutil

import pytorch_lightning as pl
import torch_geometric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from autoencoder import AutoEncoder
from constants import AVAIL_GPUS, CHECKPOINT_BASE_PATH, RAND_SEED
from utils import get_dataset

pl.seed_everything(RAND_SEED)
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)


def train(dataset_name, latent_dim, **kwargs):
    max_epochs = kwargs.get('max_epochs', 500)
    learning_rate = kwargs.get('learning_rate', 1e-3)

    dataset = get_dataset(dataset_name)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    experiment_name = f'{dataset_name}-ae-{latent_dim}'
    root_dir = os.path.join(CHECKPOINT_BASE_PATH, experiment_name)
    os.makedirs(root_dir, exist_ok=True)

    model = AutoEncoder(input_size=dataset.num_node_features, hidden_size=latent_dim, learning_rate=learning_rate)

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

    trainer.fit(model, node_data_loader, node_data_loader)
    model = AutoEncoder.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    saved_best_model_path = os.path.join(CHECKPOINT_BASE_PATH, f'{experiment_name}.ckpt')
    shutil.copy(trainer.checkpoint_callback.best_model_path, saved_best_model_path)


if __name__ == '__main__':
    for dataset_name in ['cora', 'citeseer']:
        for latent_dim in [100]:
            train(dataset_name, latent_dim, max_epochs=2500, learning_rate=1e-3)
