import torch
import pickle
import numpy as np
import pandas as pd
from utils import test_calibrator
from probmetrics.metrics import Metrics
from probmetrics.calibrators import(
    get_calibrator,
    TorchcalVectorScalingCalibrator,
    TorchcalMatrixScalingCalibrator
)

def unpickle_probs(file):
    with open(file, 'rb') as f:
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)
    return ((y_probs_val, y_val), (y_probs_test, y_test))

def softmax(logits):
    l = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(l)
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)

def benchmark_vision_experiment(dataset, model, methods=['ts', 'svs', 'sms', 'torch_vs', 'torch_ms', 'dir_ms']):

    metrics = Metrics.from_names(['logloss', 'brier', 'accuracy', 'ece-15'])

    (p_val, y_val), (p_test, y_test) = unpickle_probs(f'NN_logits/{model}_{dataset}_logits.p')

    p_val = softmax(p_val)
    p_test = softmax(p_test)

    y_val = y_val.flatten()
    y_test = y_test.flatten()

    n_val, k = p_val.shape
    n_test = len(y_test)

    results = {
        'dataset': dataset,
        'model': model,
        'n_classes': k,
        'val_size': n_val,
        'test_size': n_test,
    }

    # Computing initial test metrics
    base_metrics = metrics.compute_all_from_labels_probs(torch.tensor(y_test), torch.tensor(p_test))
    results.update({f'base_{key}': value.item() for key, value in base_metrics.items()})

    if 'iso' in methods:
        iso = get_calibrator('isotonic')
        results = test_calibrator(iso, 'iso', metrics, results, p_val, y_val, p_test, y_test, verbose=True) 

    if 'ts' in methods:
        ts = get_calibrator('ts-mix')
        results = test_calibrator(ts, 'ts', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    if 'svs' in methods:
        svs = get_calibrator('svs')
        results = test_calibrator(svs, 'svs', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    if 'svs_bfgs' in methods:
        svs_bfgs = get_calibrator('svs', svs_opt='bfgs')
        results = test_calibrator(svs_bfgs, 'svs_bfgs', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    if 'torch_vs'in methods:
        torch_vs = TorchcalVectorScalingCalibrator()
        results = test_calibrator(torch_vs, 'torch_vs', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    if 'sms' in methods:
        sms = get_calibrator('sms')
        results = test_calibrator(sms, 'sms', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    if 'sms_bfgs' in methods:
        sms_bfgs = get_calibrator('sms', svs_opt='bfgs')
        results = test_calibrator(sms_bfgs, 'sms_bfgs', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    if 'torch_ms' in methods:
        torch_ms = TorchcalMatrixScalingCalibrator()
        results = test_calibrator(torch_ms, 'torch_ms', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    if 'dir_ms' in methods:
        dir_ms = get_calibrator('dircal', dircal_reg_lambda=1e-3, dircal_reg_mu=1e-3)
        results = test_calibrator(dir_ms, 'dir_ms', metrics, results, p_val, y_val, p_test, y_test, verbose=True)

    return results

if __name__ == "__main__":

    results = []
    for model in ['densenet40', 'lenet5', 'resnet110', 'resnet110_SD', 'resnet_wide32']:
        print(f'### {model} - cifar-10 ###')
        r = benchmark_vision_experiment('c10', model, methods=['ts', 'svs', 'torch_vs', 'sms', 'torch_ms', 'dir_ms'])
        # r = benchmark_vision_experiment('c10', model, methods=['iso', 'ts', 'svs_bfgs', 'sms_bfgs'])
        if r is not None:
            results.append(r)
        print()
        print(f'### {model} - cifar-100 ###')
        r = benchmark_vision_experiment('c100', model, methods=['ts', 'svs', 'torch_vs', 'sms', 'torch_ms', 'dir_ms'])
        # r = benchmark_vision_experiment('c100', model, methods=['iso', 'ts', 'svs_bfgs', 'sms_bfgs'])
        if r is not None:
            results.append(r)
        print()
    results = pd.DataFrame(results)

    results.to_csv('results/ComputerVision/results.csv', index=False)
    # results.to_csv('results/ComputerVision/results_test.csv', index=False)

    results_imagenet = []
    for model in ['densenet161', 'resnet152']:
        print(f'### {model} - imagenet ###')
        r = benchmark_vision_experiment('imgnet', model, methods=['svs_bfgs'])
        # r = benchmark_vision_experiment('imgnet', model, methods=['ts', 'svs', 'torch_vs'])
        if r is not None:
            results_imagenet.append(r)
        print()
    results_imagenet = pd.DataFrame(results_imagenet)

    results_imagenet.to_csv('results/ComputerVision/results_imagenet.csv', index=False)
    # results_imagenet.to_csv('results/ComputerVision/results_imagenet_test.csv', index=False)
