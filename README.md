# LogisticCalibrationBenchmark
Experimental repository for the paper [Structured Matrix Scaling for Multi-Class Calibration](https://arxiv.org/abs/2511.03685).

Uses the [probmetrics](https://github.com/dholzmueller/probmetrics/) package to benchmark new logistic post hoc calibration functions against existing baselines.

## Benchmarks ##
- Post hoc calibration on binary tabular datasets: `benchmark_binary.py` runs the benchmark and figures are generated in `results_binary.ipynb`.
- Post hoc calibration on multiclass tabular datasets: `benchmark_multiclass.py` runs the benchmark and figures are generated in `results_multiclass.ipynb`.
- Post hoc calibration on multiclass computer vision datasets: `benchmark_vision.py` runs the benchmark and figures are generated in `results_vision.ipynb`.

## Hyperparameter search ##
A hyperparameter search to find a default menu of regularization parameters for SVS and SMS is performed in `hyperparameter_search.ipynb`.

## Data ##
Prediction on tabular datasets that we use in our benchmark are from  [TabRepo](https://github.com/autogluon/tabarena/blob/main/tabrepo.md).

Logits for the computer vision benchmark come from [NN_calibration](https://github.com/markus93/NN_calibration).
