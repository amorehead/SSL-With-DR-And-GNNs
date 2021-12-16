import os

import matplotlib.pyplot as plt
from matplotlib import rcParams

from constants import RAND_SEED, REDUCE_METHOD_READABLE_NAMES, RESULT_DIR
from utils import (extract_hidden_features, get_experiment_name,
                   plot_hidden_features, print_number_of_parameters)

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']


def main(dataset_name, model_name, reduce_method, viz_methods, seed, **model_kwargs):
    c_hidden = model_kwargs.get('c_hidden', 16)
    num_layers = model_kwargs.get('num_layers', 2)
    hidden_features, labels = extract_hidden_features(
        dataset_name=dataset_name, model_name=model_name, reduce_method=reduce_method,
        seed=seed, c_hidden=c_hidden, num_layers=num_layers, dp_rate=0.1
    )

    fig, axs = plt.subplots(1, len(viz_methods), figsize=(12, 4), constrained_layout=True)

    for i, method_name in enumerate(viz_methods):
        kwargs = viz_methods[method_name]
        title = REDUCE_METHOD_READABLE_NAMES[method_name]
        experiment_name = get_experiment_name(dataset_name, model_name, reduce_method, seed, model_kwargs)
        plot_name = f'{experiment_name}-viz_{method_name}.png'
        save_dir = os.path.join(RESULT_DIR, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, plot_name)
        plot_hidden_features(axs[i], method_name, hidden_features, labels, dataset_name, title, save_path, **kwargs)
        print(f'Visualizing hidden features of the {model_name} on the {dataset_name} dataset: {method_name}')

    experiment_name = get_experiment_name(dataset_name, model_name, reduce_method, None, model_kwargs)

    def clean_title(t):
        t = t.replace('-', ' ')
        t = t.replace('pca_', 'PCA ')
        t = t.replace('ae_', 'AE ')
        t = t[0].upper() + t[1:]
        return t
    fig.suptitle(clean_title(experiment_name), fontsize=22)
    figures_path = os.path.join(RESULT_DIR, f'{experiment_name}.png')
    plt.savefig(figures_path, dpi=300, bbox_inches="tight")
    figures_path = os.path.join(RESULT_DIR, f'{experiment_name}.pdf')
    plt.savefig(figures_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    viz_methods = {
        'pca': {},
        'tsne': {},
        'umap': {},
    }
    seed = RAND_SEED
    reduce_methods = [('', 0), ('pca', 100), ('ae', 100)]
    settings = {
        'c_hidden': 64,
    }
    for dataset_name in ['cora', 'citeseer']:
        for reduce_method in reduce_methods:
            for model_name in ['MLP', 'GCN', 'GAT', 'GraphConv']:
                if model_name == 'MLP':
                    settings['num_layers'] = 1
                else:
                    settings['num_layers'] = 1
                main(dataset_name, model_name, reduce_method, viz_methods, seed, **settings)
                # print_number_of_parameters(dataset_name, model_name, reduce_method, seed, **settings)
