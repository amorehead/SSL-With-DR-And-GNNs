import os
import shutil

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch_geometric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from autoencoder import AutoEncoder
from constants import AVAIL_GPUS, CHECKPOINT_BASE_PATH, RAND_SEED, RESULT_DIR
from utils import get_dataset, write_ae_error

os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)


def train(dataset_name, latent_dim, seed, **model_kwargs):
    pl.seed_everything(seed)

    max_epochs = model_kwargs.get('max_epochs', 500)
    learning_rate = model_kwargs.get('learning_rate', 1e-3)

    dataset = get_dataset(dataset_name)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    experiment_name = f'{dataset_name}-ae-{latent_dim}-seed_{seed}'
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

    error = compute_test_error(model, node_data_loader)
    write_ae_error(dataset_name, seed, error)


def compute_test_error(model, node_data_loader):
    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=1
    )
    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    return test_result[0]['test_loss_epoch']


def plot_errors(dataset_name, latent_dims):
    dataset = get_dataset(dataset_name)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    errors = []
    for latent_dim in latent_dims:
        checkpoint_path = os.path.join(CHECKPOINT_BASE_PATH, f'{dataset_name}-ae-{latent_dim}.ckpt')
        model = AutoEncoder.load_from_checkpoint(checkpoint_path)
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('# of params =', num_trainable_params)
        error = compute_test_error(model, node_data_loader)
        errors.append(error)

    fig, ax = plt.subplots()
    ax.plot(latent_dims, errors, 'ko-')
    ax.set_xticks(latent_dims)
    ax.set_xlabel('Latent space')
    ax.set_ylabel('MSE')
    ae_errors_path = os.path.join(RESULT_DIR, f'{dataset_name}-ae-errors.png')
    plt.savefig(ae_errors_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    for dataset_name in ['cora', 'citeseer']:
        latent_dims = [100]
        seeds = [8735, 2021, 5555, 25, 888]
        for latent_dim in latent_dims:
            for seed in seeds:
                train(dataset_name, latent_dim, max_epochs=2500, learning_rate=1e-3, seed=seed)
        # plot_errors(dataset_name, latent_dims)
