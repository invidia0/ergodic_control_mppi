"""NumPy/JAX implementations of the configured literature baselines."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.experiments.common import Scenario, build_target_grid
from ergodic_control_mppi.models import double_integrator as model
from ergodic_control_mppi.mppi.single import run_single
from ergodic_control_mppi.parameters import GMMParams
from ergodic_control_mppi.simulation import random_state


METHODS = ("mppi", "smc", "hedac", "traj_opt", "dec")


@dataclass(frozen=True)
class _FourierContext:
    k_arr_np: np.ndarray  # (M, 2)
    lambda_k_np: np.ndarray  # (M,)
    phi_k_np: np.ndarray  # (M,)
    k_arr_jax: jnp.ndarray  # (M, 2)
    lambda_k_jax: jnp.ndarray  # (M,)
    phi_k_jax: jnp.ndarray  # (M,)
    x_min: float
    x_max: float
    y_min: float
    y_max: float



def make_no_obstacle_scenario(
    base_scenario: Scenario,
    gmm_spec: GMMSpec,
    *,
    steps: int,
    grid_shape: tuple[int, int] = (80, 80),
) -> Scenario:
    means = jnp.asarray(gmm_spec.means, dtype=jnp.float32)
    covs = jnp.asarray(gmm_spec.covariances, dtype=jnp.float32)
    weights = jnp.asarray(gmm_spec.weights, dtype=jnp.float32)

    cov_inv = jnp.linalg.inv(covs)
    if not bool(jnp.all(jnp.isfinite(cov_inv))):
        raise ValueError(f"scenario '{gmm_spec.name}' covariances are singular or invalid")

    log_weights = jnp.log(weights + 1e-10)
    log_det = jnp.log(jnp.linalg.det(covs))
    d = means.shape[1]
    log_norm = -0.5 * (d * jnp.log(2 * jnp.pi) + log_det)

    gmm = GMMParams(
        means=means,
        covariance_inverse=cov_inv,
        log_weights=log_weights,
        log_normalizers=log_norm,
    )
    workspace = dataclasses.replace(
        base_scenario.params.workspace,
        obstacles=jnp.zeros((0, 3), dtype=jnp.float32),
        obstacle_cost=0.0,
    )
    params = dataclasses.replace(
        base_scenario.params,
        gmm=gmm,
        workspace=workspace,
    )
    target_grid = build_target_grid(params, grid_shape=grid_shape)

    return Scenario(
        name=gmm_spec.name,
        params=params,
        run_config=dataclasses.replace(base_scenario.run_config, steps=int(steps)),
        target_density_grid=target_grid,
        map_x_limits=tuple(map(float, params.workspace.x_limits)),
        map_y_limits=tuple(map(float, params.workspace.y_limits)),
        obstacle_map=np.zeros((0, 3), dtype=np.float64),
        safety_radius=float(base_scenario.safety_radius),
    )



def sample_initial_states(params, team_size: int, seed: int) -> np.ndarray:
    if team_size < 1:
        raise ValueError("team_size must be >= 1")
    key = jax.random.PRNGKey(seed)
    robot_keys = jax.random.split(key, team_size)
    x0_all = jax.vmap(random_state, in_axes=(0, None))(robot_keys, params)
    return np.asarray(x0_all, dtype=np.float32)



def _fourier_context(scenario: Scenario, order: int) -> _FourierContext:
    if order < 1:
        raise ValueError("fourier order must be >= 1")
    k_list = []
    for kx in range(order + 1):
        for ky in range(order + 1):
            if kx == 0 and ky == 0:
                continue
            k_list.append((float(kx), float(ky)))
    k_arr = np.asarray(k_list, dtype=np.float64)
    lambda_k = np.power(1.0 + np.sum(k_arr * k_arr, axis=1), -1.5)

    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    nx = scenario.target_density_grid.shape[1]
    ny = scenario.target_density_grid.shape[0]
    xs = np.linspace(x_min, x_max, nx, dtype=np.float64)
    ys = np.linspace(y_min, y_max, ny, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx, gy], axis=-1)  # (ny, nx, 2)
    vals = _basis_values_np(pts.reshape(-1, 2), k_arr, x_min, x_max, y_min, y_max).reshape(ny, nx, -1)
    target = np.asarray(scenario.target_density_grid, dtype=np.float64)
    target = target / max(float(target.sum()), 1e-12)
    phi_k = np.sum(vals * target[:, :, None], axis=(0, 1))

    return _FourierContext(
        k_arr_np=k_arr,
        lambda_k_np=lambda_k,
        phi_k_np=phi_k,
        k_arr_jax=jnp.asarray(k_arr, dtype=jnp.float32),
        lambda_k_jax=jnp.asarray(lambda_k, dtype=jnp.float32),
        phi_k_jax=jnp.asarray(phi_k, dtype=jnp.float32),
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
    )



def _basis_values_np(
    xy: np.ndarray,
    k_arr: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64)
    kx = k_arr[:, 0][None, :]
    ky = k_arr[:, 1][None, :]
    x_span = max(float(x_max - x_min), 1e-12)
    y_span = max(float(y_max - y_min), 1e-12)

    xn = ((pts[:, 0:1] - x_min) / x_span)
    yn = ((pts[:, 1:2] - y_min) / y_span)
    return np.cos(np.pi * xn * kx) * np.cos(np.pi * yn * ky)



def _basis_and_grad_np(xy: np.ndarray, ctx: _FourierContext) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(xy, dtype=np.float64)
    kx = ctx.k_arr_np[:, 0][None, :]
    ky = ctx.k_arr_np[:, 1][None, :]
    x_span = max(float(ctx.x_max - ctx.x_min), 1e-12)
    y_span = max(float(ctx.y_max - ctx.y_min), 1e-12)

    xn = ((pts[:, 0:1] - ctx.x_min) / x_span)
    yn = ((pts[:, 1:2] - ctx.y_min) / y_span)

    cos_x = np.cos(np.pi * xn * kx)
    cos_y = np.cos(np.pi * yn * ky)
    sin_x = np.sin(np.pi * xn * kx)
    sin_y = np.sin(np.pi * yn * ky)

    basis = cos_x * cos_y
    grad_x = -(np.pi / x_span) * (kx * sin_x * cos_y)
    grad_y = -(np.pi / y_span) * (ky * cos_x * sin_y)
    grad = np.stack([grad_x, grad_y], axis=-1)
    return basis, grad



def _basis_values_jax(xy: jnp.ndarray, ctx: _FourierContext) -> jnp.ndarray:
    kx = ctx.k_arr_jax[:, 0][None, None, :]
    ky = ctx.k_arr_jax[:, 1][None, None, :]
    x_span = jnp.asarray(max(float(ctx.x_max - ctx.x_min), 1e-12), dtype=jnp.float32)
    y_span = jnp.asarray(max(float(ctx.y_max - ctx.y_min), 1e-12), dtype=jnp.float32)

    xn = ((xy[..., 0:1] - ctx.x_min) / x_span)
    yn = ((xy[..., 1:2] - ctx.y_min) / y_span)
    return jnp.cos(jnp.pi * xn * kx) * jnp.cos(jnp.pi * yn * ky)



def _grid_index(x: np.ndarray, low: float, high: float, n: int) -> np.ndarray:
    span = max(high - low, 1e-12)
    idx = (x - low) / span * (n - 1)
    return np.clip(idx, 0.0, n - 1.0)



def _resize_grid_bilinear(grid: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(grid, dtype=np.float64)
    out_h, out_w = out_shape
    src_h, src_w = src.shape

    if (src_h, src_w) == out_shape:
        return src.copy()

    y_src = np.linspace(0.0, 1.0, src_h)
    x_src = np.linspace(0.0, 1.0, src_w)
    y_out = np.linspace(0.0, 1.0, out_h)
    x_out = np.linspace(0.0, 1.0, out_w)

    tmp = np.vstack([np.interp(x_out, x_src, src[i, :]) for i in range(src_h)])
    out = np.vstack([np.interp(y_out, y_src, tmp[:, j]) for j in range(out_w)]).T
    out = np.maximum(out, 0.0)
    s = float(out.sum())
    return out / s if s > 0.0 else out



def _boundary_bias(
    xy: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    margin: float = 0.75,
    gain: float = 1.5,
) -> np.ndarray:
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    out = np.zeros_like(xy, dtype=np.float64)

    out[:, 0] += np.where(xy[:, 0] < x_min + margin, gain * ((x_min + margin) - xy[:, 0]), 0.0)
    out[:, 0] -= np.where(xy[:, 0] > x_max - margin, gain * (xy[:, 0] - (x_max - margin)), 0.0)
    out[:, 1] += np.where(xy[:, 1] < y_min + margin, gain * ((y_min + margin) - xy[:, 1]), 0.0)
    out[:, 1] -= np.where(xy[:, 1] > y_max - margin, gain * (xy[:, 1] - (y_max - margin)), 0.0)
    return out



def _limit_speed(v: np.ndarray, desired_speed: float) -> np.ndarray:
    if desired_speed <= 0.0:
        return np.zeros_like(v)
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.minimum(1.0, desired_speed / np.maximum(speed, 1e-12))
    return v * scale



def _clamp_controls_np(u: np.ndarray, model_params) -> np.ndarray:
    return np.asarray(model.clamp(jnp.asarray(u), model_params), dtype=np.float64)



def _clip_state_to_map_np(
    x: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    out_x = np.clip(out[:, 0], x_limits[0], x_limits[1])
    out_y = np.clip(out[:, 1], y_limits[0], y_limits[1])
    hit_x = out_x != out[:, 0]
    hit_y = out_y != out[:, 1]
    out[:, 0] = out_x
    out[:, 1] = out_y
    out[hit_x, 2] = 0.0
    out[hit_y, 3] = 0.0
    return out



def _double_integrator_step_np(x: np.ndarray, u: np.ndarray, model_params) -> np.ndarray:
    return np.asarray(model.step(jnp.asarray(x), jnp.asarray(u), model_params), dtype=np.float64)



def _tracker_step_np(
    states: np.ndarray,
    desired_vel: np.ndarray,
    scenario: Scenario,
    tracker_gain: float,
) -> np.ndarray:
    accel = tracker_gain * (desired_vel - states[:, 2:4])
    yaw_accel = -tracker_gain * states[:, 5]
    controls = np.zeros((states.shape[0], 3), dtype=np.float64)
    controls[:, 0:2] = accel
    controls[:, 2] = yaw_accel

    controls = _clamp_controls_np(controls, scenario.params.model)
    next_states = _double_integrator_step_np(states, controls, scenario.params.model)
    next_states = _clip_state_to_map_np(next_states, scenario.map_x_limits, scenario.map_y_limits)
    return next_states



def _run_mppi(
    scenario: Scenario,
    x0_all: np.ndarray,
    *,
    steps: int,
    seed: int,
) -> np.ndarray:
    if x0_all.shape[0] != 1:
        raise ValueError("the 'mppi' baseline is single-robot only")
    x0_j = jnp.asarray(x0_all, dtype=jnp.float32)
    params = scenario.params

    key = jax.random.PRNGKey(seed)
    U0 = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
    result = jax.jit(run_single, static_argnames=("steps",))(params, x0_j[0], U0, key, steps)
    return np.asarray(result.path, dtype=np.float64)[:, None, :]



def _run_smc_like(
    scenario: Scenario,
    x0_all: np.ndarray,
    *,
    steps: int,
    cfg: LiteratureComparisonConfig,
    gain: float,
    consensus: bool,
) -> np.ndarray:
    ctx = _fourier_context(scenario, cfg.fourier_order)
    team_size = x0_all.shape[0]

    states = np.asarray(x0_all, dtype=np.float64)
    paths = np.zeros((steps, team_size, 6), dtype=np.float64)

    team_coeff = np.zeros_like(ctx.phi_k_np)
    local_coeff = np.zeros((team_size, ctx.phi_k_np.shape[0]), dtype=np.float64)

    for t in range(steps):
        xy = states[:, :2]
        basis, grad = _basis_and_grad_np(xy, ctx)

        if consensus:
            local_coeff = (t * local_coeff + basis) / float(t + 1)
            coeff = np.mean(local_coeff, axis=0)
        else:
            team_sample = np.mean(basis, axis=0)
            team_coeff = (t * team_coeff + team_sample) / float(t + 1)
            coeff = team_coeff

        err = coeff - ctx.phi_k_np
        weighted_err = ctx.lambda_k_np * err
        grad_obj = np.einsum("m,rmc->rc", weighted_err, grad)

        desired_vel = -gain * grad_obj
        desired_vel += _boundary_bias(xy, scenario.map_x_limits, scenario.map_y_limits)
        desired_vel = _limit_speed(desired_vel, cfg.desired_speed)

        states = _tracker_step_np(states, desired_vel, scenario, cfg.tracker_gain)
        paths[t] = states

    return paths



def _run_hedac(
    scenario: Scenario,
    x0_all: np.ndarray,
    *,
    steps: int,
    cfg: LiteratureComparisonConfig,
) -> np.ndarray:
    team_size = x0_all.shape[0]
    states = np.asarray(x0_all, dtype=np.float64)
    paths = np.zeros((steps, team_size, 6), dtype=np.float64)

    n = int(cfg.hedac.grid_size)
    target = _resize_grid_bilinear(np.asarray(scenario.target_density_grid, dtype=np.float64), (n, n))
    occupancy_counts = np.zeros((n, n), dtype=np.float64)

    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits

    for t in range(steps):
        xy = states[:, :2]
        ix = _grid_index(xy[:, 0], x_min, x_max, n)
        iy = _grid_index(xy[:, 1], y_min, y_max, n)

        ix_i = np.clip(np.round(ix).astype(np.int64), 0, n - 1)
        iy_i = np.clip(np.round(iy).astype(np.int64), 0, n - 1)
        np.add.at(occupancy_counts, (iy_i, ix_i), 1.0)

        total = float(occupancy_counts.sum())
        occupancy = occupancy_counts / total if total > 0.0 else np.zeros_like(occupancy_counts)
        source = target - occupancy

        potential = np.zeros_like(source)
        for _ in range(cfg.hedac.jacobi_iterations):
            new_p = potential.copy()
            new_p[1:-1, 1:-1] = (
                potential[0:-2, 1:-1]
                + potential[2:, 1:-1]
                + potential[1:-1, 0:-2]
                + potential[1:-1, 2:]
                + cfg.hedac.diffusion_gain * source[1:-1, 1:-1]
            ) / (4.0 + cfg.hedac.damping)
            potential = new_p

        grad_y, grad_x = np.gradient(potential)

        x0 = np.floor(ix).astype(np.int64)
        y0 = np.floor(iy).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, n - 1)
        y1 = np.clip(y0 + 1, 0, n - 1)
        x0 = np.clip(x0, 0, n - 1)
        y0 = np.clip(y0, 0, n - 1)

        wx = ix - x0
        wy = iy - y0

        gx = (
            (1.0 - wx) * (1.0 - wy) * grad_x[y0, x0]
            + wx * (1.0 - wy) * grad_x[y0, x1]
            + (1.0 - wx) * wy * grad_x[y1, x0]
            + wx * wy * grad_x[y1, x1]
        )
        gy = (
            (1.0 - wx) * (1.0 - wy) * grad_y[y0, x0]
            + wx * (1.0 - wy) * grad_y[y0, x1]
            + (1.0 - wx) * wy * grad_y[y1, x0]
            + wx * wy * grad_y[y1, x1]
        )

        desired_vel = cfg.hedac.gradient_gain * np.stack([gx, gy], axis=1)
        desired_vel += _boundary_bias(xy, scenario.map_x_limits, scenario.map_y_limits)
        desired_vel = _limit_speed(desired_vel, cfg.desired_speed)

        states = _tracker_step_np(states, desired_vel, scenario, cfg.tracker_gain)
        paths[t] = states

    return paths



def _rollout_controls(x0: jnp.ndarray, controls: jnp.ndarray, model_params) -> jnp.ndarray:
    def one_step(x, u_t):
        x_next = model.step(x, u_t, model_params)
        return x_next, x_next

    _, traj = jax.lax.scan(one_step, x0, controls)
    return traj



def _run_traj_opt(
    scenario: Scenario,
    x0_all: np.ndarray,
    *,
    steps: int,
    seed: int,
    cfg: LiteratureComparisonConfig,
) -> np.ndarray:
    ctx = _fourier_context(scenario, cfg.fourier_order)
    x0 = jnp.asarray(x0_all, dtype=jnp.float32)
    team_size = x0.shape[0]

    model_params = scenario.params.model
    lin_lim = float(model_params.max_accel_lin_abs)
    ang_lim = float(model_params.max_accel_ang_abs)

    x_min = float(scenario.map_x_limits[0])
    x_max = float(scenario.map_x_limits[1])
    y_min = float(scenario.map_y_limits[0])
    y_max = float(scenario.map_y_limits[1])

    key = jax.random.PRNGKey(seed)
    controls = 1e-2 * jax.random.normal(key, shape=(steps, team_size, 3), dtype=jnp.float32)

    def loss_fn(u: jnp.ndarray) -> jnp.ndarray:
        traj = _rollout_controls(x0, u, model_params)
        pos = traj[..., :2]

        basis = _basis_values_jax(pos, ctx)
        coeff = jnp.mean(basis, axis=(0, 1))
        spectral = jnp.mean(ctx.lambda_k_jax * jnp.square(coeff - ctx.phi_k_jax))

        ctrl_pen = jnp.mean(jnp.square(u))
        if steps > 1:
            smooth_pen = jnp.mean(jnp.square(u[1:] - u[:-1]))
        else:
            smooth_pen = jnp.asarray(0.0, dtype=jnp.float32)

        px = pos[..., 0]
        py = pos[..., 1]
        bounds_violation = (
            jax.nn.relu(x_min - px)
            + jax.nn.relu(px - x_max)
            + jax.nn.relu(y_min - py)
            + jax.nn.relu(py - y_max)
        )
        bounds_pen = jnp.mean(jnp.square(bounds_violation))

        return (
            spectral
            + cfg.traj_opt.control_weight * ctrl_pen
            + cfg.traj_opt.smoothness_weight * smooth_pen
            + cfg.traj_opt.bounds_weight * bounds_pen
        )

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    m = jnp.zeros_like(controls)
    v = jnp.zeros_like(controls)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for i in range(cfg.traj_opt.iterations):
        _, grad = loss_and_grad(controls)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * jnp.square(grad)
        m_hat = m / (1.0 - beta1 ** (i + 1))
        v_hat = v / (1.0 - beta2 ** (i + 1))
        controls = controls - cfg.traj_opt.learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        controls = controls.at[..., 0].set(jnp.clip(controls[..., 0], -lin_lim, lin_lim))
        controls = controls.at[..., 1].set(jnp.clip(controls[..., 1], -lin_lim, lin_lim))
        controls = controls.at[..., 2].set(jnp.clip(controls[..., 2], -ang_lim, ang_lim))

    traj = _rollout_controls(x0, controls, model_params)
    traj_np = np.asarray(traj, dtype=np.float64)
    flat = traj_np.reshape(-1, 6)
    clipped = _clip_state_to_map_np(flat, scenario.map_x_limits, scenario.map_y_limits)
    return clipped.reshape(steps, team_size, 6)



def run_literature_method(
    method_name: str,
    scenario: Scenario,
    x0_all: np.ndarray,
    *,
    steps: int,
    seed: int,
    cfg: LiteratureComparisonConfig,
) -> np.ndarray:
    if method_name not in METHODS:
        raise ValueError(f"unknown literature method '{method_name}'")

    team_size = int(x0_all.shape[0])
    if x0_all.shape != (team_size, 6):
        raise ValueError(f"x0_all must have shape (robots, 6), got {x0_all.shape}")

    if method_name == "mppi":
        out = _run_mppi(scenario, x0_all, steps=steps, seed=seed)
    elif method_name == "smc":
        out = _run_smc_like(
            scenario,
            x0_all,
            steps=steps,
            cfg=cfg,
            gain=float(cfg.smc_gain),
            consensus=False,
        )
    elif method_name == "hedac":
        out = _run_hedac(scenario, x0_all, steps=steps, cfg=cfg)
    elif method_name == "traj_opt":
        out = _run_traj_opt(scenario, x0_all, steps=steps, seed=seed, cfg=cfg)
    else:
        out = _run_smc_like(
            scenario,
            x0_all,
            steps=steps,
            cfg=cfg,
            gain=float(cfg.dec_gain),
            consensus=True,
        )

    arr = np.asarray(out, dtype=np.float64)
    if arr.shape != (steps, team_size, 6):
        raise ValueError(
            f"method '{method_name}' returned shape {arr.shape}, expected {(steps, team_size, 6)}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"method '{method_name}' returned non-finite states")
    return arr
