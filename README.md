# Policy-Gradient Flow Matching for Real-Time Ergodic Control

This repository contains an implementation of the Ergodic Control using Model Predictive Path Integral (MPPI) method. The code is designed to facilitate research and experimentation in the field of ergodic control for robotic systems.

This repository provides an implementation of **Model Predictive Path Integral (MPPI) control** with two interchangeable backends:

- **JAX** – functional, JIT-compiled, fast for research and large batch sampling  
- **PyTorch** – imperative, flexible, easier integration with existing Torch pipelines  

Both backends implement the same MPPI logic but are maintained independently.

---

## Requirements

- **Python**: `>= 3.8`
- `pip >= 23` recommended

> ⚠️ **Backend-specific Python versions**
>
> - **JAX backend**: Python **≥ 3.9**
> - **PyTorch backend**: Python **≥ 3.8**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/invidia0/ergodic_control_mppi.git
cd ergodic_control_mppi
```

---

## Install with JAX backend

```bash
pip install .[jax]
```

Make sure your CUDA version matches the available JAX wheels.

---

## Install with PyTorch backend

> ⚠️ Requires **Python ≥ 3.8**

```bash
pip install .[torch]
```

If you are using a virtual environment:

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install .[torch]
```

---

## Installing both backends (for benchmarking)

If your Python version supports both extras:

```bash
pip install .[jax,torch]
```

This is useful for performance comparisons and validation experiments.

---

## Optional dependencies

### Development / testing
```bash
pip install .[dev]
```

Example (full setup):

```bash
pip install .[jax,dev]
```

---

## Usage

Import the backend explicitly:

```python
# JAX
from mppi_control.jax import MPPI

# PyTorch
from mppi_control.torch import MPPI
```

The APIs are intentionally kept as similar as possible across backends.

---

## Project structure

```text
mppi/
├── jax/      # JAX implementation
└── torch/    # PyTorch implementation
```

