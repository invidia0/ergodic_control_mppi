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
    EXCESS_CMAP,
    SEQUENTIAL_CMAP,
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


def _target_contours(axis, ctx, levels: int = 4) -> None:
    """Recessive target-density contours: context, never a foreground mark."""
    xs = np.linspace(*ctx["limits_x"], 160)
    ys = np.linspace(*ctx["limits_y"], 160)
    grid_x, grid_y = np.meshgrid(xs, ys)
    density = np.asarray(
        pdf(jnp.stack((grid_x, grid_y), axis=-1).astype(jnp.float32), ctx["gmm"])
    )
    axis.contour(grid_x, grid_y, density, levels=levels,
                 colors="#9AA7BE", linewidths=0.4, alpha=0.9, zorder=1)


def _map_axes(axis, ctx, *, ylabel: bool = False) -> None:
    """Shared geometry for every workspace panel, so panels align exactly."""
    axis.set_xlim(*ctx["limits_x"])
    axis.set_ylim(*ctx["limits_y"])
    axis.set_aspect("equal")
    axis.set_xticks([-10, -5, 0, 5, 10])
    axis.set_yticks([-10, -5, 0, 5, 10])
    axis.set_xlabel(r"$x$ [m]", labelpad=1)
    if ylabel:
        axis.set_ylabel(r"$y$ [m]", labelpad=1)
    else:
        axis.set_yticklabels([])


def _modes(axis, ctx, label: str | None = None) -> None:
    """Mark the target modes identically in every panel."""
    means = np.asarray(ctx["params"].gmm.means)
    axis.plot(means[:, 0], means[:, 1], marker="x", linestyle="none",
              color="#33415C", markersize=3.4, markeredgewidth=0.9,
              zorder=6, label=label)


def _inline_colorbar(figure, axis, mappable, label: str) -> None:
    """Compact horizontal colorbar *inside* the panel.

    An external colorbar steals ~15% of a panel's width and, when only some
    panels carry one, leaves their titles at different heights. Putting it
    inside keeps every map the same size.
    """
    cax = axis.inset_axes((0.06, 0.055, 0.42, 0.035))
    bar = figure.colorbar(mappable, cax=cax, orientation="horizontal")
    bar.outline.set_linewidth(0.4)
    bar.outline.set_edgecolor("#5C6B87")
    cax.tick_params(labelsize=5.6, length=1.4, pad=1, colors="#33415C")
    cax.set_title(label, fontsize=6.0, pad=2, color="#33415C")


def figure_pipeline(ctx, output: Path) -> Path:
    """Fig. A -- one buffer, three colourings: memory, excess, resulting field.

    Every panel is the same workspace map of the same memory points, so the
    pipeline reads left to right as one object being re-weighted.
    """
    stein = ctx["stein"]
    coarse = float(stein.coarse_bandwidth)
    _, _, excess, activity = _excess(ctx, coarse)
    memory = np.asarray(ctx["memory"])
    recency = np.asarray(ctx["recency"])
    acts = excess > 0.0

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.35))

        # (a) what the controller retains, against everywhere it has been.
        axis = axes[0]
        _target_contours(axis, ctx)
        axis.plot(ctx["full_path"][:, 0], ctx["full_path"][:, 1],
                  color="#B9C2D4", linewidth=0.3, alpha=0.9, zorder=2,
                  label="executed path")
        order = np.argsort(recency)
        scatter = axis.scatter(memory[order, 0], memory[order, 1], c=recency[order],
                               s=2.4, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0,
                               linewidths=0, zorder=3)
        _modes(axis, ctx, label="target modes")
        _map_axes(axis, ctx, ylabel=True)
        axis.set_title(r"(a) fading memory $\mathcal{M}_t$")
        _inline_colorbar(figure, axis, scatter, r"recency $\omega_i$")
        axis.legend(loc="lower right", fontsize=5.8, handletextpad=0.35,
                    borderpad=0.25, handlelength=1.1, labelspacing=0.2,
                    framealpha=0.9)

        # (b) the hinge, shown spatially: inert points simply do not colour.
        axis = axes[1]
        _target_contours(axis, ctx)
        axis.scatter(memory[~acts, 0], memory[~acts, 1], s=1.8, color="#A8B2C6",
                     linewidths=0, zorder=2)
        order = np.argsort(excess[acts])
        active = axis.scatter(memory[acts][order, 0], memory[acts][order, 1],
                              c=excess[acts][order], s=2.6, cmap=EXCESS_CMAP,
                              linewidths=0, zorder=3)
        _modes(axis, ctx)
        _map_axes(axis, ctx)
        axis.set_title(f"(b) over-coverage ({100 * acts.mean():.0f}% acts)")
        _inline_colorbar(figure, axis, active, r"excess $e^{h_c}_{t,i}$")

        # (c) the field those weights produce. Length carries magnitude, so the
        # colour channel stays free rather than repeating it.
        axis = axes[2]
        grid_x, grid_y, points = _grid(ctx, step=1.15)
        repulsion = np.asarray(
            multiscale_memory_flow(
                points, ctx["memory"], ctx["recency"], ctx["gmm"], stein, ctx["density_floor"]
            )
        )
        attraction = np.asarray(
            stein_gradient(points, ctx["particles"], ctx["gmm"], stein, ctx["bandwidth"])
        )
        axis.streamplot(
            grid_x, grid_y,
            attraction[:, 0].reshape(grid_x.shape), attraction[:, 1].reshape(grid_x.shape),
            color="#B9C2D4", linewidth=0.3, density=0.55, arrowsize=0.0, zorder=2,
        )
        axis.quiver(points[:, 0], points[:, 1], repulsion[:, 0], repulsion[:, 1],
                    color="#C0392F", width=0.0055, headwidth=3.4, headlength=3.8,
                    zorder=4)
        _modes(axis, ctx)
        _map_axes(axis, ctx)
        axis.set_title(r"(c) repelling field")
        axis.text(0.055, 0.055,
                  "grey: attraction $\\widetilde{\\mathbf{h}}_t$\n"
                  "red: $k_{\\mathcal{M}}\\boldsymbol{\\rho}^{\\mathrm{MS}}_t$",
                  transform=axis.transAxes, fontsize=5.9, va="bottom",
                  color="#33415C", linespacing=1.5)

        figure.tight_layout(pad=0.3, w_pad=0.5)
        path = save(figure, output)
        plt.close(figure)

    print(f"  activity gate S/(S+eps_S) = {activity / (activity + 1e-3):.4f}  (S = {activity:.4g})")
    return path


def figure_scale_bank(ctx, output: Path) -> Path:
    """Fig. B -- why there are Q scales, and why none carries its own gain.

    (a) stacks raw above gauged so "before/after" is spatial rather than encoded
    by dash-vs-solid, which would compete with the hue identifying the scale.

    (b) reads the excess along the buffer at both endpoints. The buffer is a thin
    curve, so plotting the two scales as maps makes them look alike -- the real
    difference is the spatial *frequency* each responds to, which a profile shows
    directly: the fine scale reacts to local retracing, the coarse scale to
    mode-wide over-service. Fig. A already grounds the buffer in the workspace,
    so no third map is needed here.
    """
    stein = ctx["stein"]
    scales = ctx["scales"]
    memory = np.asarray(ctx["memory"])
    # Ordinal ramp: h_0 < h_1 < h_2 is ordered, so it takes one hue light->dark,
    # not three categorical hues.
    colors = [SEQUENTIAL_CMAP(v) for v in np.linspace(0.12, 1.0, len(scales))]

    with plt.rc_context(rc=paper_style("double")):
        figure = plt.figure(figsize=(6.9, 2.55), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.15])
        ax_raw = figure.add_subplot(grid[0, 0])
        ax_gauged = figure.add_subplot(grid[1, 0], sharex=ax_raw)

        radii = np.linspace(1e-3, 3.6, 400)
        probe = jnp.stack((radii, jnp.zeros_like(radii)), axis=-1)
        origin = jnp.zeros_like(probe)
        peaks = []
        for index, (bandwidth, color) in enumerate(zip(scales, colors)):
            raw = np.linalg.norm(np.asarray(kernel_gradient(probe, origin, bandwidth)), axis=-1)
            gauged = float(np.sqrt(0.5 * np.e * bandwidth)) * raw
            peaks.append(gauged.max())
            ax_raw.plot(radii, raw, color=color, linewidth=1.0)
            ax_gauged.plot(radii, gauged, color=color, linewidth=1.0)
            # Direct label at the peak instead of a legend box.
            ax_raw.annotate(rf"$h_{index}\!=\!{bandwidth:.2f}$",
                            xy=(float(radii[int(np.argmax(raw))]), raw.max()),
                            xytext=(2.5, 1.0), textcoords="offset points",
                            fontsize=6.0, color=color)

        # Self-check, not decoration: the gauge is correct iff every scale peaks
        # at exactly 1. Fail loudly rather than ship a wrong figure.
        if not np.allclose(peaks, 1.0, atol=1e-5):
            raise AssertionError(f"gauge broken: peak magnitudes {peaks} != 1")

        ax_gauged.axhline(1.0, color="#5C6B87", linewidth=0.5, zorder=1)
        ax_raw.set_ylabel(r"$\|\nabla\kappa_h\|$", labelpad=2)
        ax_gauged.set_ylabel("gauged", labelpad=2)
        ax_gauged.set_xlabel(r"$r=\|\mathbf{z}-\mathbf{m}_{t,i}\|$ [m]", labelpad=1)
        ax_raw.tick_params(labelbottom=False)
        ax_raw.set_ylim(0, 3.5)
        ax_gauged.set_ylim(0, 1.3)
        ax_raw.set_title("(a) raw: peaks differ 8x", pad=3)
        ax_gauged.set_title(r"$\times\sqrt{he/2}$: peaks coincide at 1", pad=3,
                            fontsize=6.5, fontweight="normal")

        # (b) excess along the buffer at both endpoints, on a shared absolute
        # axis. Normalising each curve to its own max was tried and discarded:
        # it made the two look alike, because on a thin trail the *shape* barely
        # differs. What actually differs is magnitude -- the fine scale sees an
        # order of magnitude more relative over-coverage -- so the axis is shared
        # and logarithmic, which shows that gap and the fine scale's higher
        # spatial frequency at the same time.
        axis = figure.add_subplot(grid[:, 1])
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(memory, axis=0), axis=1))))
        peaks_by_scale = {}
        for bandwidth, color, symbol in (
            (float(stein.fine_bandwidth), colors[0], r"$h_f$"),
            (float(stein.coarse_bandwidth), colors[-1], r"$h_c$"),
        ):
            _, _, excess, _ = _excess(ctx, bandwidth)
            peaks_by_scale[symbol] = float(excess.max())
            axis.plot(arc, np.maximum(excess, 1e-3), color=color, linewidth=0.8,
                      label=rf"{symbol}$={bandwidth:.2f}$")
        axis.set_yscale("log")
        axis.set_xlim(arc[0], arc[-1])
        axis.set_ylim(1e-2, 60)
        axis.set_xlabel("arc length along the buffer [m]  (oldest to newest)", labelpad=1)
        axis.set_ylabel(r"relative excess $e^h_{t,i}$", labelpad=2)
        ratio = peaks_by_scale[r"$h_f$"] / max(peaks_by_scale[r"$h_c$"], 1e-9)
        axis.set_title(rf"(b) the fine scale sees {ratio:.0f}$\times$ more excess")
        axis.legend(loc="lower right", fontsize=6.0, handletextpad=0.4,
                    borderpad=0.3, handlelength=1.4, labelspacing=0.25)

        path = save(figure, output)
        plt.close(figure)
    return path


def figure_extra(ctx, output_dir: Path) -> list[Path]:
    """Rebuttal-only: matched target, activity gate, effective blend.

    Deliberately not in the submission. Kept runnable so the answer exists if a
    reviewer asks why the occupancy is compared against a smoothed target, what
    eps_S is for, or why the two fields are blended after normalization rather
    than blending their weights.
    """
    stein = ctx["stein"]
    paths = []
    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))

        # Why p*_h and not p*: comparing a smoothed occupancy against an
        # unsmoothed target manufactures excess that is pure bandwidth artefact.
        axis = axes[0]
        coarse = float(stein.coarse_bandwidth)
        line_y = ctx["slice_y"]
        xs = np.linspace(*ctx["limits_x"], 400)
        slice_points = jnp.stack((xs, jnp.full_like(xs, line_y)), axis=-1)
        raw = np.asarray(pdf(slice_points, ctx["gmm"]))
        matched = np.asarray(pdf(slice_points, smoothed(ctx["gmm"], coarse)))
        axis.fill_between(xs, matched, raw, where=raw > matched, color=ACCENT,
                          alpha=0.16, linewidth=0)
        axis.plot(xs, matched, color=PRIMARY, linewidth=1.1)
        axis.plot(xs, raw, color=ACCENT, linewidth=1.0)
        axis.annotate(r"$p^\star_{h_c}$", xy=(xs[300], matched[300]),
                      xytext=(3, 3), textcoords="offset points",
                      fontsize=6, color=PRIMARY)
        axis.annotate(r"$p^\star$", xy=(xs[int(np.argmax(raw))], raw.max()),
                      xytext=(3, 1), textcoords="offset points",
                      fontsize=6, color=ACCENT)
        axis.set_xlabel(rf"$x$ [m] at $y={line_y:.0f}$", labelpad=1)
        axis.set_ylabel("density", labelpad=2)
        axis.set_title("shaded: excess that\nan unmatched target invents", fontsize=6.5)

        # Gate: field magnitude as total excess vanishes.
        axis = axes[1]
        activities = np.geomspace(1e-6, 1e1, 200)
        axis.plot(activities, activities / (activities + 1e-3), color=PRIMARY,
                  linewidth=1.1, label=r"gated: $S/(S+\varepsilon_S)$")
        axis.axhline(1.0, color=ACCENT, linewidth=0.9, linestyle="--",
                     label="ungated (scale-invariant)")
        axis.axvline(1e-3, color=NEUTRAL, linewidth=0.6, linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel(r"total relative excess $S^h_t$")
        axis.set_ylabel("field scale")
        axis.set_title("the gate restores continuity\nas over-coverage vanishes", fontsize=6.5)
        axis.legend(loc="lower right", fontsize=6)

        # Effective blend coefficient of the retired weight-blending design.
        axis = axes[2]
        grid_a = np.linspace(0.0, 1.0, 200)
        grid_s = np.geomspace(1e-3, 1e1, 200)
        mesh_a, mesh_s = np.meshgrid(grid_a, grid_s)
        effective = mesh_a * mesh_s / ((1.0 - mesh_a) + mesh_a * mesh_s)
        image = axis.pcolormesh(mesh_a, mesh_s, effective - mesh_a, cmap="RdBu_r",
                                vmin=-0.5, vmax=0.5, shading="auto")
        axis.set_yscale("log")
        axis.set_xlabel(r"requested balance $a$")
        axis.set_ylabel(r"$S^h_t$")
        axis.set_title(r"drift $\alpha^h_t-a$ if the weights" "\n" r"are blended instead", fontsize=6.5)
        bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=5.5)

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
