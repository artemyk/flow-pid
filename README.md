# Partial Information Decomposition

This repository contains the code for the following paper on partial information decomposition (PID).

[**Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions**](https://arxiv.org/abs/2510.04417)<br>
Wenyuan Zhao, Adithya Balachandran, Chao Tian, Paul Pu Liang<br>
NeurIPS 2025.

If you find this repository useful, please cite:
```
@inproceedings{zhao2025partial,
  title={Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions},
  author={Zhao, Wenyuan and Balachandran, Adithya and Tian, Chao and Liang, Paul Pu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Usage
### Environment Setup
To install the repository, first clone the repository via Git, and then install the prerequisite packages through Conda (Linux/MacOS):
```
conda env create [-n ENVNAME] -f environment.yml
```

### PID Estimators
#### Gaussian PID solvers
1. Thin-PID: `exact_gauss_thin_pid` in `pid/thin_pid.py`
2. Tilde-PID: `exact_gauss_tilde_pid` in `pid/tilde_pid.py`
3. MMI-PID: `mmi_pid` in `pid/mmi_pid.py`

#### Thin-PID stopping controls (this fork)

Both `exact_thin_pid_minimizer` and `exact_gauss_thin_pid` accept:

- `objective_target=None`: stop when the union objective is at most this
  threshold, in **bits** (before debiasing or clipping).
- `native_stopping=True`: set to `False` to disable the original stagnation
  rule when measuring time to an objective target. The iteration cap still applies.
- `timeout=None`: a positive minimizer time limit in seconds. Expiry raises
  `TimeoutError`, rather than returning an unconverged PID as a successful run.

```python
sig, value, iterations, history = exact_thin_pid_minimizer(
    hx, hy, ret_obj=True, objective_target=target_bits,
    native_stopping=False, timeout=100.0,
)
```

The timeout includes minimizer initialization, but excludes the covariance
wrapper's preprocessing and postprocessing. It is cooperative: checked between
numerical operations, without interrupting an in-progress BLAS/LAPACK call.
Use an external process watchdog if a strict wall-clock deadline is required.
An objective target is a stopping threshold, not an optimality certificate;
the original iteration-limit warning still applies if the target is not reached.
Omitting these options retains the original stopping behavior and return format.

Run the focused tests with `python -m pytest -q tests/test_thin_pid_stopping.py`.

#### Generic PID solvers
1. Flow-PID: `flow_pid` in `pid/flow_pid.py`
2. CVX/BATCH: see [Quantifying & Modeling Multimodal Interactions](https://arxiv.org/abs/2302.12247)



### Experiments
#### Gaussian PID
To compute PIDs on 1D GPID examples, run
```
python examples/gpid/1d_gpid.py
```

To compute PIDs on 2D GPID examples, run
```
python examples/gpid/gain_angle_sweeps.py
```

To compute PIDs on GPID examples at higher dimensionality, run
```
python examples/gpid/multi_dim.py
python examples/gpid/doubling.py
```

Examples of time analysis on GPID algorithms: `examples\gpid\time_analysis.ipynb`

#### Synthetic Data
Examples of multivariate Gaussian with invertible nonlinear transformation: `examples\synthetic\multi_dim_non_gauss.ipynb`

The code for synthetic specialized interactions is located under `synthetic\`.

To generate synthetic data, run 
```
python examples/synthetic/generate_data.py --num-data 20000 --setting redundancy --out-path synthetic/data
```

Example command to solve PIDs with Thin/Flow-PID on synthetic data:
```
python examples/synthetic/pid_solver.py --data-path ./data/experiments/DATA_redundancy.pickle --bs 256 --input-dim 100 --hidden-dim 512 --n-latent 600 --num-classes 2
```

To train a multimodal model on synthetic data, see [CVX/BATCH](https://github.com/pliang279/PID/blob/main/synthetic/baseline.py).

#### Real-world Data
We rely on the original [MultiBench](https://github.com/pliang279/MultiBench) implementations for real-world data experiments. Please refer to the repo for more details on training the encoders.

The code for training normalizing flows and solve Flow-PIDs is located under `examples\MultiBench`.

## Acknowledgments

Some portions of the code were adapted or referenced from the following open-source projects:

- [praveenv253/gpid](https://github.com/praveenv253/gpid): used for the implementation of Gaussian PID solvers.
- [pliang279/PID](https://github.com/pliang279/PID): provided helpful examples for real-world datasets.

We gratefully acknowledge their contributions and great work!
