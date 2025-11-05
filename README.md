# LogisticCalibrationBenchmark
Experimental repository for the paper "Structured Matrix Scaling for Multi-Class Calibration".
Uses the [probmetrics](https://github.com/dholzmueller/probmetrics/) package to benchmark new logistic post hoc calibration functions against existing baselines.

## Benchmarks ##
- post hoc calibration on binary tabular datasets, `benchmark_binary.py` runs the benchmark and results are displayed in `results_binary.ipynb`.
- post hoc calibration on multiclass tabular datasets, `benchmark_multiclass.py` runs the benchmark and results are displayed in `results_multiclass.ipynb`.
- post hoc calibration on multiclass computer vision datasets, `benchmark_vision.py` runs the benchmark and results are displayed in `results_vision.ipynb`.

## Data ##
Prediction on tabular datasets that we use in our benchmark are from  [TabRepo](https://github.com/autogluon/tabarena/blob/main/tabrepo.md).

Logits for the computer vision benchmark come from [NN_calibration](https://github.com/markus93/NN_calibration).
