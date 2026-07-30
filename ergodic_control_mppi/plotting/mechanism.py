"""Explanatory figures for Sec. III-C, Fading-Memory Coverage Feedback.

Every field, weight and bandwidth drawn here is produced by calling the
controller's own functions -- ``kernel_gradient``, ``smoothed``, ``pdf``,
``stein_repulsion``, ``multiscale_memory_flow``, ``stein_gradient`` -- on a memory
buffer taken from a real run. Nothing is hand-drawn, so a figure cannot drift
away from the implementation.

    python -m ergodic_control_mppi.plotting.mechanism --output-dir theory/pictures

The source run is cached; delete the .npz to regenerate it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.stein import (
    kernel,
    kernel_gradient,
    multiscale_memory_flow,
    pdf,
    smoothed,
    stein_gradient,
    stein_repulsion,
)
from ergodic_control_mppi.plotting.style import (
    ACCENT,
    NEUTRAL,
    PRIMARY,
    paper_style,
    save,
)

# Enough steps for the buffer to hold a representative mid-run trail rather than
# the initial transient: the shipped tau_M = 10 s gives P = 1500 at dt = 0.02.
SOURCE_STEPS = 4000


def _source(config_path: str, cache: Path, device: str, steps: int) -> dict[str, np.ndarray]:
    """Load or produce the run whose final memory buffer the figures use."""
    if cache.exists():
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}

    import yaml

    from ergodic_control_mppi.simulation import run_simulation

    # Patch only the step count; everything else is the shipped config.
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data["steps"] = steps
    scratch = cache.parent / "mechanism_source_config.yaml"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config = load_config(scratch)

    result = run_simulation(config, device)
    length = config.controller.mppi.memory_length
    path = result.paths[:, 0, :2].astype(np.float32)

    # Illustration instant. The buffer only ever holds the last P steps (30 s at
    # the shipped tau_M), so the *end* of a run is an arbitrary slice of it -- and
    # often a single dwell, which shows the mechanism at its least interesting.
    # Pick instead the instant whose buffer is most spatially spread, i.e. one
    # containing a dwell and a transit. This selects the illustration, not the
    # mechanism: every quantity below is still computed from a genuine buffer.
    cell = 2.0
    best_end, best_spread = length, -1
    for end in range(length, len(path) + 1, max(length // 10, 1)):
        window = path[end - length:end]
        occupied = len(set(map(tuple, np.floor(window / cell).astype(np.int64))))
        if occupied > best_spread:
            best_end, best_spread = end, occupied

    arrays = {
        "memory": path[best_end - length:best_end],
        "path": path[:best_end],
        "surrogate": result.surrogates[0].astype(np.float32),
        "position": path[best_end - 1],
        "buffer_end": np.asarray(best_end, dtype=np.int64),
        "buffer_cells": np.asarray(best_spread, dtype=np.int64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    return arrays


def _context(config, arrays):
    """Per-step derived quantities, exactly as core.py builds them."""
    params = config.controller
    stein, workspace, gmm = params.stein, params.workspace, params.gmm
    memory = jnp.asarray(arrays["memory"])

    ages = jnp.arange(memory.shape[0])[::-1]
    recency = stein.memory_decay ** ages
    density_floor = 1.0 / (
        (workspace.x_limits[1] - workspace.x_limits[0])
        * (workspace.y_limits[1] - workspace.y_limits[0])
    )
    # Attraction source and bandwidth: the median rollout, as core.py:171-176.
    particles = jnp.asarray(arrays["surrogate"])
    differences = particles[:, None, :] - particles[None, :, :]
    bandwidth = jnp.maximum(
        jnp.median(jnp.sum(differences * differences, axis=-1)), stein.self_bandwidth
    )
    scales = np.asarray(
        jnp.geomspace(stein.fine_bandwidth, stein.coarse_bandwidth, stein.memory_scales)
    )
    # Slice through whichever mode the buffer actually occupies, so panel (b)'s
    # three curves are comparable instead of all sitting near zero.
    means = np.asarray(gmm.means)
    buffer_np = np.asarray(arrays["memory"])
    occupancy_per_mode = [
        float(np.mean(np.linalg.norm(buffer_np - mean, axis=-1) < 3.0)) for mean in means
    ]
    return {
        "params": params, "stein": stein, "gmm": gmm, "workspace": workspace,
        "memory": memory, "recency": recency, "density_floor": density_floor,
        "particles": particles, "bandwidth": bandwidth, "scales": scales,
        "full_path": np.asarray(arrays["path"]),
        "position": np.asarray(arrays["position"]),
        "slice_y": float(means[int(np.argmax(occupancy_per_mode))][1]),
        "limits_x": tuple(map(float, workspace.x_limits)),
        "limits_y": tuple(map(float, workspace.y_limits)),
    }


def _excess(ctx, bandwidth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Occupancy, scale-matched target, relative excess at the memory points.

    Mirrors stein.py:174-185 term for term.
    """
    memory, recency = ctx["memory"], ctx["recency"]
    occupancy = (kernel(memory[:, None, :], memory[None, :, :], bandwidth) @ recency) / (
        jnp.sum(recency) * jnp.pi * bandwidth
    )
    target = pdf(memory, smoothed(ctx["gmm"], bandwidth))
    excess = jnp.maximum(occupancy - target, 0.0) / (target + ctx["density_floor"])
    activity = jnp.sum(recency * excess) / jnp.sum(recency)
    return (
        np.asarray(occupancy), np.asarray(target), np.asarray(excess), float(activity)
    )


def _grid(ctx, step: float = 0.9):
    """Query grid for the quiver panels."""
    xs = np.arange(ctx["limits_x"][0] + step, ctx["limits_x"][1], step)
    ys = np.arange(ctx["limits_y"][0] + step, ctx["limits_y"][1], step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)
    return grid_x, grid_y, jnp.asarray(points, dtype=jnp.float32)


def _target_contours(axis, ctx, levels: int = 5) -> None:
    xs = np.linspace(*ctx["limits_x"], 160)
    ys = np.linspace(*ctx["limits_y"], 160)
    grid_x, grid_y = np.meshgrid(xs, ys)
    density = np.asarray(
        pdf(jnp.stack((grid_x, grid_y), axis=-1).astype(jnp.float32), ctx["gmm"])
    )
    axis.contour(grid_x, grid_y, density, levels=levels,
                 colors="#5C6B87", linewidths=0.45, alpha=0.75)


def _square(axis, ctx) -> None:
    axis.set_xlim(*ctx["limits_x"])
    axis.set_ylim(*ctx["limits_y"])
    axis.set_aspect("equal")
    axis.set_xlabel(r"$x$ [m]")


def figure_pipeline(ctx, output: Path) -> Path:
    """Fig. A -- buffer, over-coverage excess, weighted repelling field."""
    stein = ctx["stein"]
    coarse = float(stein.coarse_bandwidth)
    occupancy, target, excess, activity = _excess(ctx, coarse)
    memory = np.asarray(ctx["memory"])
    recency = np.asarray(ctx["recency"])
    masses = recency / recency.sum()

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.45))

        # (a) the fading memory: what the controller retains (coloured by recency
        # weight) against everywhere it has actually been (faint grey) -- the
        # visual definition of "fading".
        axis = axes[0]
        _target_contours(axis, ctx)
        axis.plot(ctx["full_path"][:, 0], ctx["full_path"][:, 1],
                  color="#B9C2D4", linewidth=0.25, alpha=0.8, zorder=1,
                  label="executed path (forgotten)")
        order = np.argsort(recency)
        scatter = axis.scatter(
            memory[order, 0], memory[order, 1], c=recency[order],
            s=2.6, cmap="Blues", norm=Normalize(0.0, 1.0), linewidths=0, zorder=3,
        )
        axis.plot(*ctx["params"].gmm.means.T, "*", color=ACCENT, markersize=5,
                  markeredgewidth=0, zorder=5)
        axis.plot(*ctx["position"], "o", color="#111111", markersize=2.6,
                  markeredgewidth=0, zorder=6)
        _square(axis, ctx)
        axis.set_ylabel(r"$y$ [m]")
        axis.set_title(r"(a) fading memory $\mathcal{M}_t$")
        bar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
        bar.set_label(r"$\omega_i$", labelpad=1)
        bar.ax.tick_params(labelsize=6)
        axis.legend(loc="upper left", fontsize=5.0, handletextpad=0.3,
                    borderpad=0.22, handlelength=1.2)

        # Inset: the recency weights and the 3-tau truncation.
        inset = axis.inset_axes((0.055, 0.055, 0.40, 0.30))
        age_seconds = np.arange(len(recency))[::-1] * float(ctx["params"].model.delta_t)
        inset.plot(age_seconds, recency, color=PRIMARY, linewidth=0.8)
        tau = -float(ctx["params"].model.delta_t) / np.log(float(stein.memory_decay))
        inset.axvline(tau, color=ACCENT, linewidth=0.6, linestyle="--")
        inset.text(tau * 1.15, 0.55, r"$\tau_{\mathcal{M}}$", color=ACCENT, fontsize=5.5)
        inset.set_xlabel("age [s]", fontsize=5.5, labelpad=0)
        inset.set_ylabel(r"$\omega_i$", fontsize=5.5, labelpad=0)
        inset.tick_params(labelsize=4.5, length=1.3, pad=1)
        inset.set_facecolor("#EEF2F8")

        # (b) the hinge: only points above the diagonal act.
        axis = axes[1]
        over = occupancy > target
        axis.scatter(target[~over], occupancy[~over], s=1.4, color=NEUTRAL,
                     linewidths=0, label="under-covered (inert)")
        axis.scatter(target[over], occupancy[over], s=1.6, color=ACCENT,
                     linewidths=0, label="over-covered (acts)")
        span = np.array([min(target.min(), occupancy.min()), max(target.max(), occupancy.max())])
        axis.plot(span, span, color="#33415C", linewidth=0.7, linestyle="--")
        axis.text(0.96, 0.06,
                  f"$[\\cdot]_+$ keeps {100.0 * over.mean():.0f}% of points",
                  transform=axis.transAxes, ha="right", va="bottom", fontsize=6.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$p^\star_h(\mathbf{m}_{t,i})$")
        axis.set_ylabel(r"$o^h_t(\mathbf{m}_{t,i})$")
        axis.set_title(r"(b) over-coverage at $h_c$")
        axis.legend(loc="upper left", fontsize=5.5, handletextpad=0.3,
                    borderpad=0.25, labelspacing=0.25)

        # (c) the weighted repelling field over the attraction.
        axis = axes[2]
        grid_x, grid_y, points = _grid(ctx)
        repulsion = np.asarray(
            multiscale_memory_flow(
                points, ctx["memory"], ctx["recency"], ctx["gmm"], stein, ctx["density_floor"]
            )
        )
        attraction = np.asarray(
            stein_gradient(points, ctx["particles"], ctx["gmm"], stein, ctx["bandwidth"])
        )
        magnitude = np.linalg.norm(repulsion, axis=-1)
        axis.streamplot(
            grid_x, grid_y,
            attraction[:, 0].reshape(grid_x.shape), attraction[:, 1].reshape(grid_x.shape),
            color="#8A97AE", linewidth=0.35, density=0.65, arrowsize=0.35,
        )
        quiver = axis.quiver(
            points[:, 0], points[:, 1], repulsion[:, 0], repulsion[:, 1], magnitude,
            cmap="Reds", scale=None, width=0.006, headwidth=3.4,
        )
        axis.plot(*ctx["params"].gmm.means.T, "*", color=ACCENT, markersize=5,
                  markeredgewidth=0, zorder=5)
        _square(axis, ctx)
        axis.set_title(r"(c) $k_{\mathcal{M}}\boldsymbol{\rho}^{\mathrm{MS}}_t$ on $\widetilde{\mathbf{h}}_t$")
        bar = figure.colorbar(quiver, ax=axis, fraction=0.046, pad=0.03)
        bar.set_label(r"$\|\boldsymbol{\rho}^{\mathrm{MS}}_t\|$", labelpad=1)
        bar.ax.tick_params(labelsize=6)

        figure.tight_layout(pad=0.35, w_pad=0.9)
        path = save(figure, output)
        plt.close(figure)

    print(f"  activity gate S/(S+eps_S) = {activity / (activity + 1e-3):.4f}  (S = {activity:.4g})")
    return path


def figure_scale_bank(ctx, output: Path) -> Path:
    """Fig. B -- the gauge, the matched target, and what each scale asks.

    Layout: two analytic line panels on top, the fine/coarse comparison of the
    *same* buffer adjacent on the bottom row.
    """
    stein = ctx["stein"]
    scales = ctx["scales"]
    memory = np.asarray(ctx["memory"])

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(2, 2, figsize=(6.9, 4.6))

        # (a) raw vs gauged kernel gradient magnitude.
        axis = axes[0, 0]
        radii = np.linspace(1e-3, 4.5, 400)
        probe = jnp.stack((radii, jnp.zeros_like(radii)), axis=-1)
        origin = jnp.zeros_like(probe)
        colors = [PRIMARY, "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2"]
        for index, (bandwidth, color) in enumerate(zip(scales, colors)):
            gradient = np.linalg.norm(
                np.asarray(kernel_gradient(probe, origin, bandwidth)), axis=-1
            )
            gauge = float(np.sqrt(0.5 * np.e * bandwidth))
            axis.plot(radii, gradient, color=color, linewidth=0.7, linestyle="--", alpha=0.75)
            axis.plot(radii, gauge * gradient, color=color, linewidth=1.1,
                      label=rf"$h_{index}={bandwidth:.2f}$")
            axis.axvline(np.sqrt(bandwidth / 2.0), color=color, linewidth=0.45,
                         linestyle=":", alpha=0.8)
        axis.axhline(1.0, color="#33415C", linewidth=0.5, linestyle="-.")
        axis.set_yscale("log")
        # The gauged peaks must all land on 1.0 -- that is the panel's self-check.
        axis.set_ylim(1e-2, 4.0)
        axis.set_xlim(0.0, 4.0)
        axis.set_xlabel(r"$r=\|\mathbf{z}-\mathbf{m}_{t,i}\|$ [m]")
        axis.set_ylabel(r"$\|\nabla_{\mathbf{x}}\kappa_h\|$")
        axis.set_title("(a) per-scale gauge")
        axis.legend(loc="lower left", fontsize=5.5, handletextpad=0.3,
                    borderpad=0.25, labelspacing=0.2)
        axis.text(0.97, 0.94, "dashed: raw\nsolid: gauged",
                  transform=axis.transAxes, ha="right", va="top", fontsize=5.5)

        # (b) the scale-matched target removes a bandwidth-dependent bias.
        axis = axes[0, 1]
        coarse = float(stein.coarse_bandwidth)
        line_y = ctx["slice_y"]
        xs = np.linspace(*ctx["limits_x"], 400)
        slice_points = jnp.stack((xs, jnp.full_like(xs, line_y)), axis=-1)
        raw = np.asarray(pdf(slice_points, ctx["gmm"]))
        matched = np.asarray(pdf(slice_points, smoothed(ctx["gmm"], coarse)))
        occupancy_slice = np.asarray(
            (kernel(ctx["memory"][None, :, :], slice_points[:, None, :], coarse) @ ctx["recency"])
            / (jnp.sum(ctx["recency"]) * jnp.pi * coarse)
        )
        axis.plot(xs, occupancy_slice, color=PRIMARY, linewidth=1.1, label=r"$o^{h_c}_t$")
        axis.plot(xs, matched, color="#33415C", linewidth=1.0, label=r"$p^\star_{h_c}$")
        axis.plot(xs, raw, color=ACCENT, linewidth=0.9, linestyle="--", label=r"$p^\star$")
        axis.fill_between(
            xs, matched, raw, where=raw > matched, color=ACCENT, alpha=0.18,
            linewidth=0, label="bias if unmatched",
        )
        axis.set_xlabel(rf"$x$ [m] at $y={line_y:.0f}$")
        axis.set_ylabel("density")
        axis.set_title(r"(b) scale-matched target $p^\star_{h_c}$")
        axis.legend(loc="upper right", fontsize=5.5, handletextpad=0.3,
                    borderpad=0.25, labelspacing=0.2)

        # (c), (d) the same buffer read at the fine and at the coarse scale.
        for column, (bandwidth, tag, label, question) in enumerate(
            (
                (float(stein.fine_bandwidth), "c", r"$h_f$", "is this track filled?"),
                (float(stein.coarse_bandwidth), "d", r"$h_c$", "has this mode had its share?"),
            )
        ):
            axis = axes[1, column]
            _, _, excess, _ = _excess(ctx, bandwidth)
            _target_contours(axis, ctx, levels=4)
            order = np.argsort(excess)
            scatter = axis.scatter(
                memory[order, 0], memory[order, 1], c=excess[order],
                s=1.8, cmap="Reds", linewidths=0,
            )
            _square(axis, ctx)
            if column == 0:
                axis.set_ylabel(r"$y$ [m]")
            axis.set_title(rf"({tag}) excess at {label} = {bandwidth:.2f}: {question}")
            bar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
            bar.set_label(r"$e^{h}_{t,i}$", labelpad=1)
            bar.ax.tick_params(labelsize=6)

        figure.tight_layout(pad=0.35, w_pad=0.9, h_pad=0.9)
        path = save(figure, output)
        plt.close(figure)
    return path


def figure_extra(ctx, output_dir: Path) -> list[Path]:
    """Rebuttal-only: the activity gate and the effective blend coefficient.

    Deliberately not in the submission -- see the plan. Kept runnable so the
    answer exists if a reviewer asks about eps_S or about blending normalized
    fields instead of blending the weights.
    """
    stein = ctx["stein"]
    paths = []
    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))

        # Gate: field magnitude as total excess vanishes.
        axis = axes[0]
        activities = np.geomspace(1e-6, 1e1, 200)
        axis.plot(activities, activities / (activities + 1e-3), color=PRIMARY,
                  linewidth=1.1, label=r"gated: $S/(S+\varepsilon_S)$")
        axis.axhline(1.0, color=ACCENT, linewidth=0.9, linestyle="--",
                     label="ungated (scale-invariant)")
        axis.axvline(1e-3, color=NEUTRAL, linewidth=0.6, linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel(r"total relative excess $S^h_t$")
        axis.set_ylabel("field scale")
        axis.set_title("activity gate restores continuity at $S=0$")
        axis.legend(loc="lower right", fontsize=6)

        # Effective blend coefficient of the retired weight-blending design.
        axis = axes[1]
        grid_a = np.linspace(0.0, 1.0, 200)
        grid_s = np.geomspace(1e-3, 1e1, 200)
        mesh_a, mesh_s = np.meshgrid(grid_a, grid_s)
        effective = mesh_a * mesh_s / ((1.0 - mesh_a) + mesh_a * mesh_s)
        image = axis.pcolormesh(mesh_a, mesh_s, effective - mesh_a, cmap="RdBu_r",
                                vmin=-0.5, vmax=0.5, shading="auto")
        axis.set_yscale("log")
        axis.set_xlabel(r"requested balance $a$")
        axis.set_ylabel(r"$S^h_t$")
        axis.set_title(r"drift $\alpha^h_t-a$ if the weights are blended")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

        figure.tight_layout(pad=0.35, w_pad=1.1)
        paths.append(save(figure, output_dir / "fig_design_choices.pdf"))
        plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mppi_params.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("theory/pictures"))
    parser.add_argument("--cache", type=Path,
                        default=Path("results/campaign/mechanism_source.npz"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--steps", type=int, default=SOURCE_STEPS)
    parser.add_argument("--extra", action="store_true",
                        help="also render the rebuttal-only design-choices figure")
    args = parser.parse_args()

    arrays = _source(args.config, args.cache, args.device, args.steps)
    config = load_config(args.config)
    ctx = _context(config, arrays)
    print(f"scale bank: {np.round(ctx['scales'], 3).tolist()}  P = {len(arrays['memory'])}")

    written = [
        figure_pipeline(ctx, args.output_dir / "fig_memory_pipeline.pdf"),
        figure_scale_bank(ctx, args.output_dir / "fig_scale_bank.pdf"),
    ]
    if args.extra:
        written.extend(figure_extra(ctx, args.output_dir))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
