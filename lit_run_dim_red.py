import os

from constants import RESULT_DIR
from utils import extract_hidden_features, plot_hidden_features


def main(dataset_name, model_name, methods, **kwargs):
    hidden_features, labels = extract_hidden_features(
        dataset_name=dataset_name, model_name=model_name,
        c_hidden=16, num_layers=2, dp_rate=0.1
    )
    for method_name in methods:
        kwargs = methods[method_name]
        title = f'{method_name} projection of {model_name} on {dataset_name}'
        save_path = os.path.join(RESULT_DIR, f'{dataset_name}-{model_name}-{method_name}.png')
        plot_hidden_features(method_name, hidden_features, labels,
                             dataset_name, title, save_path, **kwargs)
        print(f'Visualizing hidden features of the {model_name} on the {dataset_name} dataset: {method_name}')


if __name__ == '__main__':
    methods = {
        'tsne': {
            'n_components': 2,
        },
        'umap': {},
        'pca': {},
    }
    # dataset_name = 'citeseer' # 'cora', 'citeseer',
    # model_name = 'GCN' # 'MLP', 'GCN', 'GAT', 'GraphConv'
    # main(dataset_name, model_name, methods)
    for model_name in ['MLP', 'GCN', 'GAT', 'GraphConv']:
        for dataset_name in ['cora', 'citeseer']:
            main(dataset_name, model_name, methods)
