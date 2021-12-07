import numpy as np
import torch_geometric

from utils import get_dataset


def display_masks(dataset_name, **model_kwargs):
    dataset = get_dataset(dataset_name)
    node_data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=1)
    batch = next(iter(node_data_loader))
    print(batch.train_mask.shape)
    print(batch.val_mask.shape)
    print(batch.test_mask.shape)
    print(np.unique(batch.train_mask, return_counts=True))
    print(np.unique(batch.val_mask, return_counts=True))
    print(np.unique(batch.test_mask, return_counts=True))

if __name__ == '__main__':
    display_masks('citeseer')
