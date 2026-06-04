# Optical PINNs

This repository implements fully backpropagation-free training for physics-informed neural networks (PINNs), supporting both weight-domain and phase-domain training across multiple PDEs.

---

## 🛠️ Dependencies

- Python ≥ 3.10
- NVIDIA GPU with CUDA ≥ 11.5
- Python packages listed in `requirements.txt`

---

## 📦 Environment Setup

### General packages

```bash
conda create -n optical_pinn python=3.10
conda activate optical_pinn
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115
pip install -r requirements.txt
```

### ONN specific packages

The code builds upon older version of [Pyutility](https://github.com/JeremieMelo/pyutility) and [pytoch-onn](https://github.com/JeremieMelo/pytorch-onn). We included the source files that are compatible with our codebase. We sincerely thank the developers of these two packages, which have greatly advanced the development of ONN simulations.

```bash
# Install local utilities
cd pyutility
python setup.py install

# Install ONN backend
cd ../pytorch-onn
python setup.py install --user clean

cd ..
```

---

## 🚀 Usage

### Direct Script Execution

Run PINNs with your desired PDE and training domain:

#### Black-Scholes Equation

- **Weight-domain training**

  ```bash
  cd ./PINNs/Black_Scholes
  python -u main.py -m both -c configs_weight/Ours.ini
  ```
- **Phase-domain training**

  ```bash
  cd ./PINNs/Black_Scholes
  python -u main.py -m both -c configs_phase/Ours.ini -y configs_phase/tonn.yml
  ```

---

### Easy-to-Use Runner Script

```bash
bash ./scripts/run_pinn.sh [PDE_NAME] [DOMAIN]
```

#### Supported Arguments

| Argument     | Allowed Values                                                  | Description                                    |
| ------------ | --------------------------------------------------------------- | ---------------------------------------------- |
| `PDE_NAME` | `Black_Scholes<br>``HJB_20d_FD<br>``Burgers<br>``Darcy` | Name of the PDE (subdirectory under `PINNs`) |
| `DOMAIN`   | `weight<br>``phase`                                         | Type of training domain                        |

**Example:**

```bash
bash ./scripts/run_pinn.sh Black_Scholes weight
```

---

## 📊 Reproduce Results in Paper

To reproduce results reported in different tables:

```bash
bash ./scripts/table_1.sh [PDE_NAME]
bash ./scripts/table_2.sh [PDE_NAME]
bash ./scripts/table_3.sh [PDE_NAME]
```
