# Optical PINNs

This repository implements fully backpropagation-free training for physics-informed neural networks (PINNs), supporting both weight-domain and phase-domain training across multiple PDEs.

---

## 📖 About this work

Physics-informed neural networks (PINNs) are a promising surrogate for high-dimensional PDE solving, but their training latency on GPUs is too high for real-time digital-twin and edge-deployment use cases. Photonic computing can close that latency gap, yet on-chip training has been blocked by two long-standing obstacles: photonic chips have no usable photonic memory for storing gradient activations, and the optical building blocks (MZIs) are too large to fit a full back-propagation graph for a real-size PINN.

This project removes both obstacles by making the entire training pipeline **completely back-propagation-free** and **photonics-friendly**. Three contributions work together:

1. **Sparse-grid Stein derivative estimator** — replaces autograd inside the PINN residual with a deterministic, low-variance derivative estimate evaluated on Smolyak sparse-grid quadrature points, eliminating the autograd graph from the loss itself.
2. **Tensor-train (TT) dimension-reduced zeroth-order optimizer** — performs parameter updates through forward-only Monte-Carlo gradient estimates in a TT-factorized weight space, which is small enough for ZO to converge well and large enough to retain the expressivity of a real PINN.
3. **Photonic tensor-core accelerator design** — a scalable on-chip architecture that maps the TT layers onto photonic tensor cores and, because no BP is required, needs no photonic memory for intermediate activations.

The numerical methods are validated on low- and high-dimensional PDE benchmarks (Black-Scholes, Burgers, Darcy, 20-D HJB), and pre-silicon simulation with real device parameters shows large reductions in chip area and end-to-end training latency relative to GPU baselines. The accompanying manuscript is included in this repository: [`Final_Scalable_Back_Propagation_Free_Training_of_Optical_Physics_Informed_Neural_Networks.pdf`](./Final_Scalable_Back_Propagation_Free_Training_of_Optical_Physics_Informed_Neural_Networks.pdf).

---

## 🛠️ Dependencies

- Python 3.10 (this is the version pinned by `requirements.txt` and tested below)
- NVIDIA GPU with a driver compatible with the CUDA 11.5 PyTorch wheels
- Python packages listed in `requirements.txt`
- The two vendored source trees `pyutility/` and `pytorch-onn/`: The code builds upon older version of [Pyutility](https://github.com/JeremieMelo/pyutility) and [pytoch-onn](https://github.com/JeremieMelo/pytorch-onn). We included the source files that are compatible with our codebase. We sincerely thank the developers of these two packages, which have greatly advanced the development of ONN simulations.

---

## 📦 Environment Setup

The repo ships a turnkey setup using a local `.venv/`. The conda alternative is documented further below.

### Option A — local `.venv/` (recommended; what this README is tested against)

Run from the repository root:

```bash
# 1. Create and activate the venv
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Upgrade the build toolchain (older pip drops legacy setup.py installs)
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch 1.11.0 + CUDA 11.5 (must come BEFORE requirements.txt so the cu115 wheel wins)
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 \
    --extra-index-url https://download.pytorch.org/whl/cu115

# 4. Install the rest of the Python deps
pip install -r requirements.txt

# 5. Install the vendored ONN-stack source trees IN ORDER
(cd pyutility   && python setup.py install)
(cd pytorch-onn && python setup.py install)        # NOTE: do NOT pass --user (breaks venv) and do NOT pass `clean` (deletes the build before install)
```

> **Troubleshooting**
>
> - `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6`: install `requirements.txt` (which pins `numpy==1.26.4`) *after* installing the torch wheels.
> - `ModuleNotFoundError: torchonn`: re-run `cd pytorch-onn && python setup.py install` without the `clean` subcommand.
> - `ModuleNotFoundError: pyutils`: the install must run *inside* the active venv; check `which python` first.

### Option B — conda

```bash
conda create -n optical_pinn python=3.10 && conda activate optical_pinn
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 \
    --extra-index-url https://download.pytorch.org/whl/cu115
pip install -r requirements.txt
(cd pyutility   && python setup.py install)
(cd pytorch-onn && python setup.py install)
```

---

## 🚀 Usage

Every PINN entry point (`PINNs/<PDE>/main.py`) prepends `../..` to `sys.path` so that `from core.TFONet import ...` resolves. **You must `cd` into the PDE directory before running** — invoking from the repo root will fail.

### Direct script execution

```bash
source .venv/bin/activate                  # if you used Option A
cd PINNs/Black_Scholes

# Weight-domain training (TT-tensorized DNN, zeroth-order PINN gradient)
python -u main.py -m both -c configs_weight/Ours.ini

# Phase-domain training (tensorized ONN — needs the extra .yml for noise/quantization knobs)
python -u main.py -m both -c configs_phase/Ours.ini -y configs_phase/tonn.yml
```

`-m both` runs training and post-training evaluation in one shot (`train` or `eval` are also accepted).

### Easy-to-use runner script

From the repository root:

```bash
bash ./scripts/run_pinn.sh <PDE_NAME> <DOMAIN>
```

| Argument     | Allowed values                                            |
| ------------ | --------------------------------------------------------- |
| `PDE_NAME` | `Black_Scholes`, `Burgers`, `Darcy`, `HJB_20d_FD` |
| `DOMAIN`   | `weight` or `phase`                                   |

Example:

```bash
bash ./scripts/run_pinn.sh Black_Scholes weight
```

The script picks one specific GPU implicitly through the inherited `CUDA_VISIBLE_DEVICES`; pin it explicitly if you share the box:

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/run_pinn.sh Black_Scholes phase
```

### 📁 Where outputs go

PINN runs land under `PINNs/<PDE>/exp/<model_type>/<...hyperparams.../<PID>/`:

```
exp/<model_type>/<hidden>/<factorization>/<rank>/<grad_func>/<optim>/<scheduler>/<exp_name>/<PID>/
├── config.ini          # resolved config snapshot
├── log.log             # training log (uses logging.CRITICAL as the info channel)
├── checkpoints/
├── figures/
└── events.out.tfevents.*   # TensorBoard summaries
```

## 📊 Reproduce results in the paper

The paper's three ablation tables are reproduced one PDE at a time:

```bash
# Table 1 — FD/Stein/Sparse-grid as the PINN gradient estimator (weight domain)
bash ./scripts/table_1.sh Black_Scholes

# Table 2 — Standard vs. TT model × FO vs. ZO optimizer (weight domain)
bash ./scripts/table_2.sh Black_Scholes

# Table 3 — Phase-domain methods: Ours vs. FLOPS vs. L2IGHT
bash ./scripts/table_3.sh Burgers
```

Each `table_*.sh <PDE>` runs three `python main.py` invocations sequentially and writes their results into `PINNs/<PDE>/exp/`. Inspect the final `test loss` line of each `log.log` to compare configurations.

---

## 🧩 What lives where

| Path                                              | Role                                                                                                                                                                               |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core/TFONet/`                                  | PINN training loop, config parser, model builders, Fourier features, differential operators                                                                                        |
| `core/models/`                                  | ONN/TONN backbones (uses MZI building blocks from `pytorch-onn/torchonn`)                                                                                                        |
| `core/tensor_layers/`, `core/tensor_fwd_bwd/` | TT / Tucker / Block-TT layers (built on `tltorch` + `tensorly`)                                                                                                                |
| `core/ZO_Estim/`                                | Layer/parameter-granularity Monte-Carlo ZO gradient estimation (the engine behind the `[ZO_Estim]` ini section)                                                                  |
| `PINNs/<PDE>/`                                  | Per-PDE entry script +`dataio.py` / `loss_func.py` / `val_func.py` / `eval_func.py` + INI/YAML configs + sparse-grid quadrature point files (`GQN_*.asc`, `KPN_*.asc`) |
| `MNIST/`                                        | Standalone MNIST classification entry (uses the same `core/` library)                                                                                                            |

For an architectural deep dive, see [`CLAUDE.md`](./CLAUDE.md).

---

## 📝 Citation

This work has been accepted by **ACM Transactions on Design Automation of Electronic Systems (TODAES)**. If you use this code or build on the ideas in the paper, please cite:

```bibtex
@article{zhao2025scalable,
  title={Scalable back-propagation-free training of optical physics-informed neural networks},
  author={Zhao, Yequan and Yu, Xinling and Xiao, Xian and Chen, Zhixiong and Liu, Ziyue and Kurczveil, Geza and Beausoleil, Raymond G and Liu, Sijia and Zhang, Zheng},
  journal={arXiv preprint arXiv:2502.12384},
  year={2025}
}
```

The BibTeX entry will be updated to the TODAES version once the journal reference (volume / issue / DOI) is published; please prefer the journal version of the citation after that.

---

## 📜 License

See [`LICENSE`](./LICENSE).
