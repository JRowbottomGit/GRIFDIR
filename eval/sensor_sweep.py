"""
Sensor-count sweep for sparse-sensor reconstruction (paper Table 1).

Runs eval/<task>.py for each (n_sensors, sampler) combination, then aggregates the
per-run {sampler}_summary.json files into a LaTeX table (RMSE / energy score vs number
of sensors, one row per sampler) plus an errorbar plot. Works for either reconstruction
task — both write the same summary schema.

Usage
-----
# Pinball (Table 1) — pass --skip_unconditional through to each run:
python eval/sensor_sweep.py --task pinball_reconstruction --run_dir checkpoints/pinball \
    --n_sensors 25 50 75 100 --samplers dps daps --skip_unconditional

# Re-tabulate without re-running (reads existing summaries):
python eval/sensor_sweep.py --task pinball_reconstruction --run_dir checkpoints/pinball \
    --n_sensors 25 50 75 100 --samplers dps daps --tabulate_only

Output (in <output_dir>, default <run_dir>/sensor_sweep):
    n{N}_{sampler}/            per-run reconstruction + {sampler}_summary.json
    sensor_sweep_table.tex     LaTeX table
    sensor_sweep.png / .pdf    errorbar plot
Any unrecognised flags are forwarded verbatim to each eval/<task>.py invocation.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Paper names the samplers Fun-DPS / Fun-DAPS.
_PRETTY = {"dps": "Fun-DPS", "daps": "Fun-DAPS"}


def _run_one(task, run_dir, n_sensors, sampler, out, n_samples, n_steps, passthrough):
    """Invoke eval/<task>.py for one (n_sensors, sampler) in a clean subprocess."""
    cmd = [sys.executable, os.path.join("eval", f"{task}.py"),
           "--run_dir", run_dir, "--n_sensors", str(n_sensors), "--sampler", sampler,
           "--output_dir", out, "--n_samples", str(n_samples), "--n_steps", str(n_steps),
           *passthrough]
    env = dict(os.environ)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")        # VE logspace falls back on mps
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n>>> {task}  n_sensors={n_sensors}  sampler={sampler}")
    subprocess.run(cmd, check=True, cwd=_ROOT, env=env)


def _collect(output_dir, n_sensors_list, samplers):
    """Read {sampler}_summary.json from each n{N}_{sampler}/ into (n_sensors, sampler) arrays."""
    shape = (len(n_sensors_list), len(samplers))
    rmse_m, rmse_s = np.full(shape, np.nan), np.full(shape, np.nan)
    es_m, es_s = np.full(shape, np.nan), np.full(shape, np.nan)
    for i, n in enumerate(n_sensors_list):
        for j, s in enumerate(samplers):
            path = os.path.join(output_dir, f"n{n}_{s}", f"{s}_summary.json")
            if not os.path.exists(path):
                print(f"  missing summary (skipped): {path}")
                continue
            with open(path) as f:
                d = json.load(f)
            rmse_m[i, j], rmse_s[i, j] = d["posterior_rmse_mean"], d["posterior_rmse_std"]
            es_m[i, j], es_s[i, j] = d["energy_score_mean"], d["energy_score_std"]
    return rmse_m, rmse_s, es_m, es_s


def _latex_table(rmse_m, rmse_s, es_m, es_s, n_sensors_list, samplers):
    def cell(m, s):
        return "$\\mathrm{n/a}$" if np.isnan(m) else f"${m:.3f} \\pm {s:.3f}$"
    head_top = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{n} sensors}}}}" for n in n_sensors_list)
    head_mid = " & ".join(["RMSE $(\\downarrow)$ & ES $(\\downarrow)$"] * len(n_sensors_list))
    lines = [
        "\\begin{table*}[htbp]", "\\centering",
        "\\caption{Performance for varying number of sensors.}", "\\label{tab:sensors}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "cc" * len(n_sensors_list) + "}", "\\toprule",
        f"& {head_top} \\\\",
        f"\\cmidrule(lr){{2-{1 + 2 * len(n_sensors_list)}}}",
        f"& {head_mid} \\\\", "\\midrule",
    ]
    for j, s in enumerate(samplers):
        row = [_PRETTY.get(s, s.upper())]
        for i in range(len(n_sensors_list)):
            row.append(cell(rmse_m[i, j], rmse_s[i, j]))
            row.append(cell(es_m[i, j], es_s[i, j]))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines)


def _plot(rmse_m, rmse_s, es_m, es_s, n_sensors_list, samplers, output_dir):
    n = np.array(n_sensors_list)
    fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(10, 4))
    for j, s in enumerate(samplers):
        label = _PRETTY.get(s, s.upper())
        ax_r.errorbar(n, rmse_m[:, j], yerr=rmse_s[:, j], marker="o", capsize=4, label=label)
        ax_e.errorbar(n, es_m[:, j], yerr=es_s[:, j], marker="s", capsize=4, label=label)
    for ax, ylab, title in ((ax_r, "RMSE", "Posterior-mean RMSE"),
                            (ax_e, "Energy score", "Energy score")):
        ax.set_xlabel("Number of sensors")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"sensor_sweep.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Sensor-count sweep + table/plot for sparse-sensor reconstruction (Table 1).")
    parser.add_argument("--task", required=True,
                        choices=["pinball_reconstruction", "gaussian_blob_reconstruction"])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--n_sensors", type=int, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--samplers", type=str, nargs="+", default=["dps", "daps"])
    parser.add_argument("--output_dir", default=None, help="Default: <run_dir>/sensor_sweep")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--tabulate_only", action="store_true",
                        help="Skip running; aggregate existing summaries only.")
    args, passthrough = parser.parse_known_args()

    output_dir = args.output_dir or os.path.join(args.run_dir, "sensor_sweep")
    os.makedirs(output_dir, exist_ok=True)

    if not args.tabulate_only:
        for n in args.n_sensors:
            for s in args.samplers:
                _run_one(args.task, args.run_dir, n, s,
                         os.path.join(output_dir, f"n{n}_{s}"),
                         args.n_samples, args.n_steps, passthrough)

    rmse_m, rmse_s, es_m, es_s = _collect(output_dir, args.n_sensors, args.samplers)
    tex = _latex_table(rmse_m, rmse_s, es_m, es_s, args.n_sensors, args.samplers)
    tex_path = os.path.join(output_dir, "sensor_sweep_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")
    _plot(rmse_m, rmse_s, es_m, es_s, args.n_sensors, args.samplers, output_dir)

    print("\n" + tex)
    print(f"\nSaved table -> {tex_path}")
    print(f"Saved plot  -> {os.path.join(output_dir, 'sensor_sweep.png')}")


if __name__ == "__main__":
    main()
