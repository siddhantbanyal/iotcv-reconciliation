# iotcv-reconciliation

Simulation and evaluation code for the IoT sensing layer and the IoT+CV composition study in *A Modular IoT-to-Vision Framework for Inventory Reconciliation via Calibrated Sequential Detection and Resource-Aware Verification* (Banyal and Shantha Raju).

The simulator builds a warehouse digital twin with known ground truth, injects inventory discrepancies, and senses them through three noisy lanes:

- **weight:** heteroscedastic CUSUM on load-cell counts
- **location:** Wald SPRT on BLE/UWB zone reads
- **dwell:** Gamma survival test on staging time

Detector scores are Platt-calibrated and triaged under a per-tick verification budget. The composition study scores IoT-only, CV-only, and IoT-triggered-CV arms on the same simulated world at equal CV budget.

## Layout

```
configs/    experiment parameters (default.yaml is the paper setting)
src/iotcv/  world, sensors, detectors, calibration, triage, composition, metrics
scripts/    entry points
results/    CSVs behind the paper's tables and figures (20 seeds)
tests/      unit tests
```

## Install

Python 3.10+.

```
pip install -r requirements.txt
```

## Reproduce

```
python scripts/run_experiment.py  --config configs/default.yaml --seeds 20 --ablations
python scripts/run_composition.py --config configs/default.yaml --seeds 20
python scripts/make_plots.py --lane all
```

Runs are seeded and deterministic. Outputs go to `results/` (use `--outdir` to write elsewhere) and figures to `figures/`.

| Output | Paper |
|---|---|
| `detector_by_lane_*`, `family_comparison_*` | detector quality, sequential vs fixed-threshold |
| `heteroscedastic_weight.csv` | weight lane vs unit-weight variability |
| `calibration_*`, `reliability_*` | calibration (ECE, reliability) |
| `triage_*` | budgeted VoI triage |
| `ablation_*` | sensitivity to detector thresholds and sensor noise |
| `composition_summary.csv`, `composition_per_seed.csv` | three-arm composition |
| `composition_budget.csv`, `composition_sensitivity.csv` | CV budget and CV accuracy sweeps |

## CV models

The CV models are not included. In the composition study the verifier is a stochastic oracle whose confirm/reject rates are set from the measured per-lane CV results (`cv_oracle` in `configs/default.yaml`).

## Tests

```
python -m unittest discover -s tests
```

## Citation

If you use this code, please cite the paper (citation to be added on publication).

## License

MIT. See `LICENSE`.
