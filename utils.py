import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def test_calibrator(cal, name, metrics, results, p_val, y_val, p_test, y_test, verbose=False):
    if verbose:
        print(f'Testing calibrator: {name}', end="")
    start = time.perf_counter()
    cal.fit(p_val, y_val)
    end = time.perf_counter()
    runtime = end - start
    if verbose:
        print(f' - fitted in {runtime:.3f}s.')
    metrics = metrics.compute_all_from_labels_probs(torch.tensor(y_test), torch.tensor(cal.predict_proba(p_test)))
    results.update({f'{name}_{key}': value.item() for key, value in metrics.items()})
    results[f'{name}_time'] = runtime
    return results

def plot_barscatter_ax(ax: plt.Axes, df: pd.DataFrame, title: str, ylabel: str):
    # Helper function for box-plots
    # adapted from https://cduvallet.github.io/posts/2018/03/boxplots-in-python

    hues = df['hue'].unique().tolist()
    colors = sns.color_palette("pastel", len(hues))
    light_colors = colors

    pal = {key: value for key, value in zip(hues, colors)}
    face_pal = {key: value for key, value in zip(hues, light_colors)}

    hue_order = hues

    boxprops = {'edgecolor': 'k', 'linewidth': 1}
    lineprops = {'color': 'k', 'linewidth': 1}

    boxplot_kwargs = {'boxprops': boxprops, 'medianprops': lineprops, 'whis': [10,90],
                      'whiskerprops': lineprops, 'capprops': lineprops,
                      'width': 0.75, 'palette': face_pal, 'hue_order': hue_order, 'dodge': True}

    stripplot_kwargs = {'linewidth': 0.4, 'size': 2.5, 'alpha': 1.0, 'palette': pal, 'hue_order': hue_order, 'dodge': True}

    ax.axhline(y=0, color='#888888', linestyle='--')
    ax.grid(True, which='major')

    sns.boxplot(x='label', y='value', hue='hue', data=df, ax=ax, fliersize=0, **boxplot_kwargs)
    sns.stripplot(x='label', y='value', hue='hue', data=df, ax=ax, jitter=0.18, **stripplot_kwargs)

    ax.set_yscale('symlog', linthresh=1)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.tick_params(axis='x', which='major')
    ax.set_title(title)
    ax.get_legend().remove()

def df_to_latex_table(df: pd.DataFrame, metric: str) -> str:

    # Identify methods
    method_cols = [c for c in df.columns if c.endswith(metric) and not c.startswith("base_")]
    base_col = f"base_{metric}"

    # Compute improvements
    improvements = df[["dataset", "model", base_col]].copy()
    for col in method_cols:
        method = col.replace(f"_{metric}", "")
        improvements[method] = df[col] - df[base_col]

    # Drop base column
    improvements = improvements.drop(columns=[base_col])

    improvements = improvements.sort_values('dataset')

    improvements = improvements.replace('c10', 'CIFAR-10')
    improvements = improvements.replace('c100', 'CIFAR-100')
    improvements = improvements.replace('resnet110_SD', 'resnet110-SD')
    improvements = improvements.replace('resnet_wide32', 'resnet-wide32')
    improvements = improvements.rename(columns={
        'ts': 'TS',
        'svs': 'SVS',
        'torch_vs': 'TorchCal VS',
        'sms': 'SMS',
        'torch_ms': 'TorchCal MS',
        'dir_ms': 'Dirichlet MS'
    })

    # Function to format row with bold best
    def highlight_best(row):
        methods_only = row.drop(["dataset", "model"])
        max_val = methods_only.min()
        return [f"\\textbf{{{v:.3f}}}" if v == max_val else f"{v:.3f}" for v in methods_only]

    # Convert improvements to strings before assigning
    formatted = improvements.astype(str)
    formatted.iloc[:, 2:] = improvements.apply(highlight_best, axis=1, result_type="expand")

    # LaTeX with booktabs
    latex = formatted.to_latex(
        index=False,
        escape=False,
        column_format="ll|" + "c" * (formatted.shape[1] - 2),
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=False
    )

    return latex

import pandas as pd