#!/usr/bin/env python
"""
Unified evaluation entry point.

Usage:
    python evaluate.py <task> [task-specific args ...]

Tasks:
    resolution_invariance         Resolution-invariance suite (gaussian_blob + cross-domain; same/finer/coarser meshes).
    pinball_reconstruction        Pinball sparse-sensor reconstruction (Fun-DPS / Fun-DAPS).
    gaussian_blob_reconstruction  Gaussian-blob sparse-sensor reconstruction.
    sensor_sweep                  Sensor-count sweep over a reconstruction task -> Table 1 (RMSE / ES vs #sensors).

Remaining arguments are forwarded to the corresponding module under eval/.
For task-specific options, run e.g.:
    python evaluate.py resolution_invariance --help
"""
import os
import sys
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = {
    "resolution_invariance":        os.path.join("eval", "resolution_invariance.py"),
    "pinball_reconstruction":       os.path.join("eval", "pinball_reconstruction.py"),
    "gaussian_blob_reconstruction": os.path.join("eval", "gaussian_blob_reconstruction.py"),
    "sensor_sweep":                 os.path.join("eval", "sensor_sweep.py"),
}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg in ("-h", "--help", None) or arg not in TASKS:
        print(__doc__)
        print("Available tasks:", ", ".join(TASKS))
        sys.exit(0 if arg in ("-h", "--help") else 1)
    script = os.path.join(_HERE, TASKS[arg])
    # Hand the remaining argv to the task script as if invoked directly.
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
