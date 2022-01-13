# Standard libraries
import os
import shutil
from collections import defaultdict

# PyTorch Lightning
import pytorch_lightning as pl
# PyTorch
import torch
# PyTorch Geometric
import torch_geometric
# PL callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Project utilities
from constants import CHECKPOINT_BASE_PATH, AVAIL_GPUS, RAND_SEED
from models import NodeLevelGNN
from utils import get_experiment_name, get_dataset, print_last_result, write_last_result, write_summary_results


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)


def train_node_classifier(
        results, model_name, dataset_name, reduce_method, fine_tune, max_epochs, learning_rate, seed, **model_kwargs):
    pl.seed_everything(seed)

    dataset = get_dataset(dataset_name, reduce_method)
    print(dataset.data.x.shape)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer
    experiment_name = get_experiment_name(dataset_name, model_name, reduce_method, seed, model_kwargs)
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

    print('test acc', test_result[0]["test_acc"])
    for set_name in ['train', 'val', 'test']:
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            metric_value = model.compute_metric(predicted_y, batch, set_name, metric_name)
            results[set_name][metric_name].append(metric_value)

    return results


def train(
        dataset_name, model_name, reduce_method, fine_tune=False, max_epochs=500, learning_rate=1e-1, c_hidden=16,
        num_layers=2, seeds=[RAND_SEED]):
    results = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    for seed in seeds:
        results = train_node_classifier(
            results=results, model_name=model_name, dataset_name=dataset_name, reduce_method=reduce_method,
            fine_tune=fine_tune, max_epochs=max_epochs, learning_rate=learning_rate, seed=seed,
            c_hidden=c_hidden, num_layers=num_layers, dp_rate=0.1)

        print(
            f'\nSemi-supervised node classification results on the {dataset_name} dataset using an {model_name}, {reduce_method[0]} {reduce_method[1]}:')
        print_last_result(results)
        write_last_result(dataset_name, model_name, reduce_method, seed, results)

    write_summary_results(dataset_name, model_name, reduce_method, seeds, results, res_format='latex')


if __name__ == '__main__':
    dataset_names = ['cora', 'citeseer']
    model_names = ['MLP', 'GCN', 'GAT', 'GraphConv']
    reduce_methods = [('', 0), ('pca', 100), ('tsne', 100), ('umap', 100), ('ae', 100)]
    settings = {
        'fine_tune': False,
        'max_epochs': 500,
        'c_hidden': 64,
        'seeds': [8735, 2021, 5555, 25, 888],
    }
    for dataset_name in dataset_names:
        for reduce_method in reduce_methods:
            for model_name in model_names:
                settings['reduce_method'] = reduce_method
                settings['dataset_name'] = dataset_name
                settings['model_name'] = model_name
                if model_name == 'MLP':
                    settings['num_layers'] = 1
                else:
                    settings['num_layers'] = 1

                if model_name == 'GraphConv':
                    settings['learning_rate'] = 1e-3
                else:
                    settings['learning_rate'] = 1e-1

                train(**settings)
