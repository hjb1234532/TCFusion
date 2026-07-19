"""
Batch Evaluation Utility
========================
Provides directory-based batch evaluation and LaTeX table generation
for rebuttal experiments.
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from .fusion_metrics import compute_all_metrics, DEFAULT_METRICS, get_metric_direction


def load_image(path):
    """Load image as RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_image_pairs(fused_dir, ir_dir, vis_dir):
    """
    Find matching image triplets across three directories.
    Matches by filename stem (without extension).

    Returns:
        list of (fused_path, ir_path, vis_path, stem)
    """
    # Collect stems from fused directory
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    fused_files = {}
    for ext in exts:
        for f in glob.glob(os.path.join(fused_dir, ext)):
            stem = os.path.splitext(os.path.basename(f))[0]
            fused_files[stem] = f

    ir_files = {}
    for ext in exts:
        for f in glob.glob(os.path.join(ir_dir, ext)):
            stem = os.path.splitext(os.path.basename(f))[0]
            ir_files[stem] = f

    vis_files = {}
    for ext in exts:
        for f in glob.glob(os.path.join(vis_dir, ext)):
            stem = os.path.splitext(os.path.basename(f))[0]
            vis_files[stem] = f

    # Find common stems
    common = sorted(set(fused_files.keys()) & set(ir_files.keys()) & set(vis_files.keys()))
    pairs = []
    for stem in common:
        pairs.append((fused_files[stem], ir_files[stem], vis_files[stem], stem))
    return pairs


def batch_evaluate(fused_dir, ir_dir, vis_dir, metrics_list=None, output_csv=None):
    """
    Batch compute fusion metrics for all image pairs in directories.

    Args:
        fused_dir: directory containing fused images
        ir_dir: directory containing IR source images
        vis_dir: directory containing visible source images
        metrics_list: list of metric names (default: DEFAULT_METRICS)
        output_csv: optional path to save results CSV

    Returns:
        pd.DataFrame with columns: image_name, metric1, metric2, ..., mean, std
    """
    if metrics_list is None:
        metrics_list = DEFAULT_METRICS

    pairs = find_image_pairs(fused_dir, ir_dir, vis_dir)
    if len(pairs) == 0:
        print(f"  Warning: No matching image pairs found in {fused_dir}")
        print(f"  IR dir: {ir_dir}, VIS dir: {vis_dir}")
        return pd.DataFrame()

    rows = []
    for fused_path, ir_path, vis_path, stem in tqdm(pairs, desc="  Evaluating"):
        try:
            fused_img = load_image(fused_path)
            ir_img = load_image(ir_path)
            vis_img = load_image(vis_path)

            # Ensure same spatial size
            h = min(fused_img.shape[0], ir_img.shape[0], vis_img.shape[0])
            w = min(fused_img.shape[1], ir_img.shape[1], vis_img.shape[1])
            fused_img = fused_img[:h, :w]
            ir_img = ir_img[:h, :w]
            vis_img = vis_img[:h, :w]

            results = compute_all_metrics(fused_img, ir_img, vis_img, metrics_list)
            results['image_name'] = stem
            rows.append(results)
        except Exception as e:
            print(f"  Warning: Failed to evaluate {stem}: {e}")

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ['image_name'] + metrics_list
    df = df[[c for c in cols if c in df.columns]]

    # Add summary row (mean ± std)
    summary_data = {'image_name': 'mean ± std'}
    for m in metrics_list:
        if m in df.columns:
            vals = df[m].dropna()
            if len(vals) > 0:
                summary_data[m] = f"{vals.mean():.4f} ± {vals.std():.4f}"
            else:
                summary_data[m] = "N/A"
    summary_df = pd.DataFrame([summary_data])
    df_with_summary = pd.concat([df, summary_df], ignore_index=True)

    # Save to CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        df_with_summary.to_csv(output_csv, index=False)
        print(f"  Results saved to: {output_csv}")

    # Print summary
    print(f"\n  Summary ({len(df)} images):")
    for m in metrics_list:
        if m in df.columns:
            vals = df[m].dropna()
            if len(vals) > 0:
                direction = '↑' if get_metric_direction().get(m, 'higher') == 'higher' else '↓'
                print(f"    {m:10s}: {vals.mean():.4f} ± {vals.std():.4f}  {direction}")

    return df_with_summary


def to_latex_table(df, caption="Fusion Metrics Comparison", label="tab:metrics",
                   best_bold=True, second_underline=True, metrics_list=None,
                   experiment_name=""):
    """
    Generate booktabs-format LaTeX table from a DataFrame.

    Args:
        df: pd.DataFrame with columns: method/config + metric columns
        caption: table caption
        label: LaTeX label
        best_bold: bold the best values
        second_underline: underline the second-best values
        metrics_list: which metric columns to include (default: all except image_name)
        experiment_name: experiment identifier for the first column header

    Returns:
        str: LaTeX table code
    """
    if metrics_list is None:
        metrics_list = [c for c in df.columns if c not in ('image_name', 'config', 'method', 'dataset')]

    direction = get_metric_direction()

    # Parse mean ± std from string values
    def parse_val(s):
        if isinstance(s, str) and '±' in s:
            return float(s.split('±')[0].strip())
        try:
            return float(s)
        except (ValueError, TypeError):
            return float('nan')

    # Extract numeric values for ranking
    method_col = None
    for col in ['config', 'method', 'image_name']:
        if col in df.columns:
            method_col = col
            break

    if method_col is None:
        method_col = df.columns[0]

    methods = df[method_col].tolist()
    n_methods = len(methods)
    n_metrics = len(metrics_list)

    # Build numeric matrix for ranking
    matrix = np.zeros((n_methods, n_metrics))
    for i in range(n_methods):
        for j, m in enumerate(metrics_list):
            if m in df.columns:
                matrix[i, j] = parse_val(df.iloc[i][m])
            else:
                matrix[i, j] = float('nan')

    # Find best and second-best for each metric
    best_idx = np.full(n_metrics, -1, dtype=int)
    second_idx = np.full(n_metrics, -1, dtype=int)
    for j in range(n_metrics):
        col_vals = matrix[:, j]
        valid_mask = ~np.isnan(col_vals)
        if valid_mask.sum() == 0:
            continue
        valid_indices = np.where(valid_mask)[0]
        valid_vals = col_vals[valid_mask]
        d = direction.get(metrics_list[j], 'higher')
        sorted_idx = valid_indices[np.argsort(valid_vals)]
        if d == 'lower':
            sorted_idx = sorted_idx[::-1]
        if len(sorted_idx) >= 1:
            best_idx[j] = sorted_idx[0]
        if len(sorted_idx) >= 2:
            second_idx[j] = sorted_idx[1]

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")

    # Column spec: first column left-aligned, rest centered
    col_spec = "l" + "c" * n_metrics
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header = experiment_name if experiment_name else "Method"
    for m in metrics_list:
        header += f" & {m}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for i in range(n_methods):
        row = str(methods[i])
        for j in range(n_metrics):
            val = df.iloc[i][metrics_list[j]] if metrics_list[j] in df.columns else "N/A"
            val_str = str(val)

            if best_bold and i == best_idx[j]:
                val_str = f"\\mathbf{{{val_str}}}"
                row += f" & ${val_str}$"
            elif second_underline and i == second_idx[j]:
                val_str = f"\\underline{{{val_str}}}"
                row += f" & ${val_str}$"
            else:
                row += f" & {val_str}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_summary_csv(results_dir, output_path):
    """
    Aggregate all experiment results into a single summary CSV.

    Args:
        results_dir: root directory containing experiment result CSVs
        output_path: path to save aggregated summary
    """
    all_csvs = glob.glob(os.path.join(results_dir, '**', 'results_*.csv'), recursive=True)
    frames = []
    for csv_path in all_csvs:
        try:
            df = pd.read_csv(csv_path)
            # Extract experiment name from filename
            exp_name = os.path.splitext(os.path.basename(csv_path))[0]
            df.insert(0, 'experiment', exp_name)
            frames.append(df)
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"Summary saved to: {output_path}")
        return combined
    else:
        print("No results found to aggregate.")
        return pd.DataFrame()
