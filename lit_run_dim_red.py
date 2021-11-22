import os

from constants import RESULT_DIR
from utils import get_experiment_name, extract_hidden_features, plot_hidden_features


def main(dataset_name, model_name, reduce_method, viz_methods, **kwargs):
    hidden_features, labels = extract_hidden_features(
        dataset_name=dataset_name, model_name=model_name, reduce_method=reduce_method,
        c_hidden=16, num_layers=2, dp_rate=0.1
    )
    for method_name in viz_methods:
        kwargs = viz_methods[method_name]
        title = f'{method_name} projection of {model_name} on {dataset_name}'
        plot_name = f'{get_experiment_name(dataset_name, model_name, reduce_method)}-viz_{method_name}.png'
        save_dir = os.path.join(RESULT_DIR, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, plot_name)
        plot_hidden_features(method_name, hidden_features, labels,
                             dataset_name, title, save_path, **kwargs)
        print(f'Visualizing hidden features of the {model_name} on the {dataset_name} dataset: {method_name}')


if __name__ == '__main__':
    viz_methods = {
        'tsne': {
            'n_components': 2,
        },
        'umap': {},
        'pca': {},
    }
    dataset_name = 'citeseer'
    model_name = 'GCN'
    reduce_method = ('pca', 100)
    main(dataset_name, model_name, reduce_method, viz_methods)
    # for model_name in ['MLP', 'GCN', 'GAT', 'GraphConv']:
    #     for dataset_name in ['cora', 'citeseer']:
    #         main(dataset_name, model_name, reduce_method, viz_methods)
