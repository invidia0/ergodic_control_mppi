#!/usr/bin/env python3
"""Tracking error of each trajectory mission in a flight_recorder.py log.

The command's position is the reference one controller step ahead, so the error compares it
with PX4's position one command later. Reports per axis the mean (a bias), rms, p95 and max,
how often the reference re-anchored (the commanded position jumps instead of advancing by the
commanded velocity), how the z error follows horizontal acceleration and the vertical
feedforward, and how far the commands lead the drone (mullet_core clamps that lead).

    python3 tests/sitl/tracking_report.py flight.npz
"""

import sys

import numpy as np

TRAJECTORY, TRAJECTORY_3D = 5, 6
log = np.load(sys.argv[1])
columns = [str(c) for c in log["columns"]]
data, missions = log["data"], [str(m) for m in log["missions"]]
col = lambda rows, name: rows[:, columns.index(name)]

for index, mission in enumerate(missions):
    rows = data[(col(data, "mission") == index) & (col(data, "running") == 1) & (col(data, "in_mission") == 1)]
    modes = set(col(rows, "mode").astype(int)) if len(rows) else set()
    if len(rows) < 50 or not modes & {TRAJECTORY, TRAJECTORY_3D}:
        continue
    three_d = TRAJECTORY_3D in modes
    t = col(rows, "t")
    dt = float(np.median(np.diff(t)))
    reference = np.stack([col(rows, f"cmd_{a}") for a in "xyz"], axis=1)[:-1]
    measured = np.stack([col(rows, a) for a in "xyz"], axis=1)[1:]
    error = measured - reference  # + = drone past the reference; z + = drone below it (NED)
    axes = "xyz" if three_d else "xy"
    print(f"\n{mission} ({'trajectory_3d' if three_d else 'trajectory'}): {len(rows)} commands over {t[-1] - t[0]:.0f} s")
    for i, a in enumerate(axes):
        e = error[:, i]
        print(f"  {a} error: mean {e.mean():+.3f} m, rms {np.sqrt(np.mean(e ** 2)):.3f}, p95 |e| {np.percentile(np.abs(e), 95):.3f}, max |e| {np.abs(e).max():.3f}")
    norm = np.linalg.norm(error[:, :len(axes)], axis=1)
    print(f"  {len(axes)}D error: rms {np.sqrt(np.mean(norm ** 2)):.3f} m, p95 {np.percentile(norm, 95):.3f}, p99 {np.percentile(norm, 99):.3f}, max {norm.max():.3f}")

    # A reference advances by about v * dt per command; a re-anchor jumps it to the drone.
    velocity = np.stack([col(rows, f"cmd_v{a}") for a in "xyz"], axis=1)
    step = np.diff(np.stack([col(rows, f"cmd_{a}") for a in "xyz"], axis=1)[:, :len(axes)], axis=0)
    jumps = np.linalg.norm(step - velocity[1:, :len(axes)] * dt, axis=1) > 0.05
    print(f"  re-anchors: {int(jumps.sum())} ({jumps.sum() / (t[-1] - t[0]):.2f} per s)")

    if three_d:
        ez = error[:, 2]
        horizontal_accel = np.linalg.norm(np.stack([col(rows, "ax"), col(rows, "ay")], axis=1), axis=1)[1:]
        vz_ff, az_ff = col(rows, "cmd_vz")[:-1], np.nan_to_num(col(rows, "cmd_az"))[:-1]
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        print(f"  z error vs |horizontal accel|: r = {corr(ez, horizontal_accel):+.2f}; "
              f"vs vz feedforward: r = {corr(ez, vz_ff):+.2f}; vs az feedforward: r = {corr(ez, az_ff):+.2f}")
        print(f"  altitude {-col(rows, 'z').max():.2f}..{-col(rows, 'z').min():.2f} m, "
              f"reference altitude {-col(rows, 'cmd_z').max():.2f}..{-col(rows, 'cmd_z').min():.2f} m")
        lead_z = np.abs(col(rows, "cmd_z") - col(rows, "z"))
        print(f"  vertical lead of the command over the drone: p95 {np.percentile(lead_z, 95):.2f} m, max {lead_z.max():.2f} m "
              f"(mullet_core clamps at max_z_position_lead_m)")
    lead_xy = np.hypot(col(rows, "cmd_x") - col(rows, "x"), col(rows, "cmd_y") - col(rows, "y"))
    print(f"  horizontal lead of the command over the drone: p95 {np.percentile(lead_xy, 95):.2f} m, max {lead_xy.max():.2f} m "
          f"(mullet_core clamps at max_xy_position_lead_m)")
