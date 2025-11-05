import json
import pandas as pd
from tqdm import tqdm
from tabrepo import load_repository

def find_best_config(repo, dataset, fold, model):
    configs = [f'{model}_r{i}_BAG_L1' for i in range(1, 201)]
    metrics = repo.metrics(datasets=[dataset], configs=configs, folds=[fold])
    return metrics.loc[metrics.metric_error_val.idxmin()].name[2]

repo = load_repository("D244_F3_C1530_200")

# Collecting datasets:
print('Collecting binary configs')
binary_datasets, multiclass_datasets = [], []
for dataset in repo.datasets():
    problem_type = repo.dataset_info(dataset)['problem_type']
    if problem_type == 'binary':
        binary_datasets.append(dataset)
    elif problem_type == 'multiclass':
        multiclass_datasets.append(dataset)
print(f'Found {len(binary_datasets)} binary datasets and {len(multiclass_datasets)} binary datasets')

with open("results/binary/datasets.json", 'w') as f:
    json.dump(binary_datasets, f, indent=2)
with open("results/multiclass/datasets.json", 'w') as f:
    json.dump(multiclass_datasets, f, indent=2)
print('Datasets saved!')

# Models:
models = [
    'CatBoost',
    'LightGBM',
    'LinearModel',
    'NeuralNetTorch',
    'RandomForest',
    'XGBoost',
    'NeuralNetFastAI',
]

# For each model-dataset-fold triplet, save default config and best config based on val error:
print('Collecting binary configs')
binary_configs = []
for dataset in tqdm(binary_datasets):
    for fold in [0,1,2]:
        for model in models:
            best_config = find_best_config(repo, dataset, fold, model)
            binary_configs.append({'model': model, 'dataset': dataset, 'fold': fold, 'default_config': model+'_c1_BAG_L1', 'tuned_config': best_config})
binary_configs = pd.DataFrame(binary_configs)
binary_configs.to_csv('results/binary/configs.csv')
print('Configs saved!')

print('Collecting multiclass configs')
multiclass_configs = []
for dataset in tqdm(multiclass_datasets):
    for fold in [0,1,2]:
        for model in models:
            best_config = find_best_config(repo, dataset, fold, model)
            multiclass_configs.append({'model': model, 'dataset': dataset, 'fold': fold, 'default_config': model+'_c1_BAG_L1', 'tuned_config': best_config})
multiclass_configs = pd.DataFrame(multiclass_configs)
multiclass_configs.to_csv('results/multiclass/configs.csv')
print('Configs saved!')
