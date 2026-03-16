# Ergodic Control via MPPI with Stein Gradient Flow

JAX-based Model Predictive Path Integral (MPPI) controller that drives robots toward a Gaussian Mixture Model (GMM) target density using Stein gradient flows. Supports single-robot and decentralized multi-robot ergodic coverage. Runs on CPU and CUDA.

## How It Works

### MPPI

At each control step, K rollout trajectories are sampled by perturbing the previous control sequence with Gaussian noise. Each trajectory accumulates stage costs (obstacle collision, out-of-bounds penalties, and ergodic objective). Trajectories are importance-weighted and combined into an updated control:

```
w_k ∝ exp(-(S_k - min(S)) / λ)
U_new = U_old + Σ_k w_k · δu_k
```

The temperature λ is auto-tuned each step to hold the Effective Sample Size (ESS) fraction at `ess_target`:

```
λ_new = clip(λ · exp(α · (ESS_target - ESS_frac)), λ_min, λ_max)
```

### Stein Gradient Flow (Ergodic Objective)

The ergodic objective steers the robot's planned trajectory toward the GMM target density via the Stein gradient operator:

```
h_A(x) = Σ_j [ k(xⱼ, x) · A · ∇log p(xⱼ)  +  A · ∇_{xⱼ} k(xⱼ, x) ]
```

where `k(x, y) = exp(-‖x - y‖² / h)` is a radial kernel and `A = R(θ)` is a 2×2 rotation matrix parameterised by mixing angle θ:

- **θ = 0°** — pure score-descent (gradient ascent toward modes)
- **θ = 90°** — pure curl (rotation around modes)
- **θ = 45°** — balanced (recommended default)

The kernel bandwidth `h` is adapted at runtime via the median heuristic applied to the robot's own planning trajectory, with a configurable floor. The own-trajectory particle set is augmented with the last `history_len` executed positions to prevent the robot from revisiting regions it has already covered.

The Stein gradient enters the MPPI cost as:

```
S_flow = -⟨position trajectory, h_target⟩
S_total = S_stage + weight · S_flow + weight_pdf · S_pdf
```

### Multi-Robot Mode

Each robot runs its own MPPI independently (decentralised). Cross-robot repulsion is layered on top:

- `cross_particles` for robot *i* = concatenation of other robots' `spatial_median` trajectories and position histories
- Cross term uses **pure kernel-gradient repulsion** (score term dropped to avoid redundant mode-attraction from foreign particles) with a **fixed** bandwidth `h_cross`
- Combined target: `h_target = h_self + cross_alpha · h_cross`

The fixed `h_cross` acts as a separation radius independent of local trajectory geometry: `√h_cross ≈ desired inter-robot separation in metres`.

---

## Project Structure

| Path | Purpose |
|------|---------|
| `configs/mppi_params.yaml` | All tunable parameters (single source of truth) |
| `configs/params_loader.py` | YAML → validated `MPPIParams` dataclass |
| `models/double_integrator.py` | 6D double-integrator dynamics + control clamping |
| `mppi/core.py` | JIT-compiled functional MPPI core |
| `mppi/stein.py` | GMM log-pdf, Stein gradient operator, `SteinParams` |
| `mppi/mppi_controller.py` | Stateful wrapper with ESS-adaptive λ |
| `scripts/main.py` | Simulation entry point (single + multi-robot) |
| `scripts/diagnose.py` | Parameter diagnostics and coverage validation |

---

## Installation

This repository relies on uv package manager. You can install uv using the following command:

```bash
sudo apt-get install curl  # if curl is not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing uv, you can clone this repository and install the required dependencies using:

```bash
uv sync
```

> [!NOTE]
> Make sure to select the correct Python venv in your IDE after installing the dependencies.

### No uv?

If you do not wish to use uv, you can manually install the required dependencies using pip. First, create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

---

## Usage

All scripts are run from the repository root. The number of robots and all other settings are controlled via `configs/mppi_params.yaml`.

```bash
# Run simulation (single-robot or multi-robot, set by robots.num_robots in YAML)
uv run python scripts/main.py

# Validate config and inspect coverage diagnostics
uv run python scripts/diagnose.py

# Syntax check all modules
uv run python -m compileall configs models mppi scripts
```

---

## Configuration Reference

All parameters live in [`configs/mppi_params.yaml`](configs/mppi_params.yaml).

| Key | Default | Description |
|-----|---------|-------------|
| `robots.num_robots` | 6 | Number of robots; set to 1 for single-robot mode |
| `mppi.K` | 1000 | Rollout samples per MPPI step |
| `mppi.T` | 50 | Planning horizon (timesteps) |
| `mppi.lambda` | 1.0 | Initial temperature λ |
| `mppi.ess_target` | 0.3 | Target ESS fraction; λ is adapted to track this |
| `mppi.lam_min` / `lam_max` | 0.05 / 20.0 | λ clamp bounds |
| `mppi.history_len` | 100 | Position history buffer length (prevents revisits) |
| `mppi.exploration` | 0.1 | Fraction of rollouts that ignore the warm-start control |
| `stein.weight` | 45.0 | Overall Stein flow cost multiplier |
| `stein.weight_pdf` | 0.0 | Log-likelihood cost multiplier (0 = disabled) |
| `stein.h` | 1.0 | Self-bandwidth floor; adapted upward by median heuristic at runtime |
| `stein.h_cross` | 4.0 | Fixed inter-robot bandwidth; `√h_cross` ≈ separation radius in metres |
| `stein.theta` | 45.0 | Mixing angle in degrees: 0° = drift, 90° = curl |
| `stein.cross_alpha` | 50.0 | Inter-robot repulsion weight (0 = disabled) |
| `map.oom_cost` | 1e8 | Out-of-bounds penalty |
| `map.x_limits` / `y_limits` | ±10 m | Map extents |
| `model.delta_t` | 0.02 | Integration timestep (s) |
| `model.double_integrator.max_accel_lin_abs` | 15.0 | Linear acceleration limit (m/s²) |
| `model.double_integrator.max_accel_ang_abs` | 15.0 | Angular acceleration limit (rad/s²) |

The target density is a GMM defined under `density:` (means, covariances, weights). Obstacles are randomly generated each run from the bounds in `map.obstacles`.

---

## State and Control

| | Symbol | Dimension | Description |
|-|--------|-----------|-------------|
| State | `x` | 6 | `[px, py, vx, vy, yaw, yaw_rate]` |
| Control | `u` | 3 | `[ax, ay, α]` — linear and angular accelerations |

The double-integrator model integrates position and velocity over `delta_t`. Controls are clamped to the configured acceleration limits before integration.
