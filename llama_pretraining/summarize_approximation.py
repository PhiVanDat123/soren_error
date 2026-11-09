"""Compact rebuttal tables from one or more rank logs; no torch dependency."""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


HEADER = (
    "| Optimizer | Step | Reference | Matrices | Mean rel. F (%) | Global rel. F (%) | Max mode (%) | Log RMSE | Mean width exact / approx |\n"
    "|---|---:|---|---:|---:|---:|---:|---:|---|\n"
)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['optimizer'], row['step'], row['reference'])].append(row)
    lines = []
    for (optimizer, step, reference), group in sorted(groups.items()):
        count = len(group)
        mean_f = sum(r['relative_frobenius_error'] for r in group) / count
        global_f = 'N/A'
        if all('exact_frobenius_squared' in r for r in group):
            denominator = sum(r['exact_frobenius_squared'] for r in group)
            global_f = f"{100 * math.sqrt(sum(r['error_frobenius_squared'] for r in group) / denominator):.3f}"
        max_mode = max(r['max_per_mode_relative_error'] for r in group)
        log_rmse = math.sqrt(sum(r['num_modes'] * r['log_centered_rmse'] ** 2 for r in group)
                             / sum(r['num_modes'] for r in group))
        exact_width = sum(r['width_exact'] for r in group) / count
        approx_width = sum(r['width_approx'] for r in group) / count
        lines.append(f"| {optimizer} | {step} | {reference} | {count} | {100*mean_f:.3f} | {global_f} | {100*max_mode:.3f} | {log_rmse:.4f} | {exact_width:.4f} / {approx_width:.4f} |\n")
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', help='Rank log paths or wildcard patterns; include ALL ranks')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    files = sorted({filename for pattern in args.logs for filename in glob.glob(pattern)})
    if not files:
        parser.error('No log files matched')
    unique = {}
    for filename in files:
        with open(filename, encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    key = (row['optimizer'], row['step'], row['reference'], row['parameter'])
                    if key in unique and unique[key] != row:
                        parser.error(f'Conflicting records for {key}; use files from one run per optimizer')
                    unique[key] = row
    report = (f"Sources: {len(files)} rank log file(s). Include all ranks for full coverage.\n\n"
              + HEADER + summarize(unique.values())
              + "\nMean rel. F: unweighted mean of per-matrix relative Frobenius errors.\n"
                "Global rel. F: sqrt(sum squared matrix errors / sum squared exact norms); N/A for old logs.\n"
                "Max mode: maximum across all matrices/modes. Log RMSE: mode-weighted, per-matrix median-centered log2 spectra.\n"
                "Widths: unweighted means of per-matrix log2 ranges. References are reported separately.\n")
    Path(args.output).write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
