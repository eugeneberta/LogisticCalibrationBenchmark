import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import test_calibrator
from tabrepo import load_repository
from probmetrics.metrics import Metrics
from probmetrics.calibrators import (
    get_calibrator,
    TorchcalTemperatureScalingCalibrator,
    TorchcalVectorScalingCalibrator,
    TorchcalMatrixScalingCalibrator,
)

import warnings
warnings.simplefilter("ignore")

def prepare_dataset(repo, dataset, fold, config, cache):
    """Load and subsample dataset splits once, reuse for all calibrators."""
    key = (dataset, fold, config)
    if key in cache:
        return cache[key]

    metrics = Metrics.from_names(['logloss', 'brier', 'accuracy', 'ece-15'])

    p_cal = repo.predict_val(dataset=dataset, fold=fold, config=config)
    y_cal = repo.labels_val(dataset=dataset, fold=fold)
    p_test = repo.predict_test(dataset=dataset, fold=fold, config=config)
    y_test = repo.labels_test(dataset=dataset, fold=fold)

    n_cal, k = p_cal.shape
    n_test = len(y_test)

    # subsampling large datasets (done once per dataset/fold/config):
    if n_cal >= 10000:
        np.random.seed(123)
        idx = np.arange(0, n_cal)
        rand_idx = np.random.choice(idx, 10000, replace=False)
        p_cal = p_cal[rand_idx]
        y_cal = y_cal[rand_idx]
        n_cal = len(p_cal)

    base_metrics = metrics.compute_all_from_labels_probs(
        torch.as_tensor(y_test), torch.as_tensor(p_test)
    )

    base_results = {
        'dataset': dataset,
        'fold': fold,
        'config': config,
        'n_classes': k,
        'cal_size': n_cal,
        'test_size': n_test,
    }
    base_results.update({f'base_{key}': value.item() for key, value in base_metrics.items()})

    cache[key] = (p_cal, y_cal, p_test, y_test, base_results, metrics)
    return cache[key]


if __name__ == "__main__":
    repo = load_repository("D244_F3_C1530_200")
    configs = pd.read_csv('results/multiclass/configs.csv')

    calibrator_factories = {
        'guo_ts': lambda: get_calibrator('guo-ts'),
        'prob_ts': lambda: get_calibrator('ts-mix'),
        'torch_ts': lambda: TorchcalTemperatureScalingCalibrator(),
        'svs': lambda: get_calibrator('svs'),
        # 'svs_bfgs': lambda: get_calibrator('svs', svs_opt='bfgs'),
        'torch_vs': lambda: TorchcalVectorScalingCalibrator(),
        'sms': lambda: get_calibrator('sms'),
        # 'sms_bfgs': lambda: get_calibrator('sms', sms_opt='bfgs'),
        'torch_ms': lambda: TorchcalMatrixScalingCalibrator(),
        'dir_ms': lambda: get_calibrator('dircal', dircal_reg_lambda=1e-3, dircal_reg_mu=1e-3),
    }

    data_cache = {}
    aggregated_results = {}

    for cal_name, factory in calibrator_factories.items():
        print(f"\n=== Running benchmark for calibrator: {cal_name} ===")
        for _, row in tqdm(configs.iterrows(), total=len(configs)):
            dataset, fold, config = row['dataset'], row['fold'], row['tuned_config']
            try:
                p_cal, y_cal, p_test, y_test, base_results, metrics = prepare_dataset(
                    repo, dataset, fold, config, data_cache
                )

                key = (dataset, fold, config)
                if key not in aggregated_results:
                    aggregated_results[key] = base_results.copy()

            except Exception:
                pass

            cal = factory()
            aggregated_results[key] = test_calibrator(
                cal, cal_name, metrics, aggregated_results[key], p_cal, y_cal, p_test, y_test
            )

    all_results = list(aggregated_results.values())
    df = pd.DataFrame(all_results)
    df.to_csv('results/multiclass/results.csv', index=False)
