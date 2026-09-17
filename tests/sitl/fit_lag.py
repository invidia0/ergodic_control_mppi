#!/usr/bin/env python3
"""Fit how PX4 follows mullet_core MISSION commands, from lag_test.py's log.

Velocity steps: measured velocity = gain * (the forwarded setpoint delayed by a dead time and
through a first-order lag). Acceleration pulses: through the velocity they produce, which is far
less noisy than PX4's acceleration estimate. Also the altitude change over the test.

    python3 tests/sitl/fit_lag.py lag.npz
"""
import sys, numpy as np
log = np.load(sys.argv[1]); c = [str(x) for x in log["columns"]]; d = log["data"]; col = lambda n: d[:, c.index(n)]
t = col("t"); dt = float(np.median(np.diff(t))); mode = col("mode")

def lagged(u, L, tau):
    shifted = np.concatenate((np.full(L, u[0]), u[:len(u) - L])) if L else u
    y = np.zeros_like(u); a = min(1.0, dt / tau)
    for k in range(1, len(u)): y[k] = y[k - 1] + (shifted[k] - y[k - 1]) * a
    return y

print("velocity steps: y = K * lag(u)")
for axis in "xyz":
    u = np.nan_to_num(np.where(mode == 3, col(f"sp_v{axis}"), 0.0)); y = col(f"v{axis}")
    moving = (mode == 3) & (np.abs(np.nan_to_num(col(f"cmd_v{axis}"))) > 0)
    mask = np.zeros(len(t), bool)
    for k in np.flatnonzero(moving): mask[max(0, k - 25):k + 150] = True
    mask &= (mode == 3) & np.isfinite(y)
    best = (np.inf,)
    for L in range(0, 20):
        for tau in np.arange(0.03, 1.5, 0.01):
            base = lagged(u, L, tau)
            K = float(base[mask] @ y[mask] / max(base[mask] @ base[mask], 1e-9))
            e = np.mean((K * base[mask] - y[mask]) ** 2)
            if e < best[0]: best = (e, L, tau, K)
    e, L, tau, K = best
    print(f"  {axis}: dead time {L*dt*1e3:3.0f} ms, tau {tau:.2f} s, gain {K:.2f}, residual rms {np.sqrt(e):.3f} m/s, explains {100*(1-e/np.var(y[mask])):.0f} %")

print("acceleration pulses: velocity change = integral of lag(a_cmd)")
for axis in "xyz":
    ua = np.nan_to_num(np.where(mode == 4, col(f"sp_a{axis}"), 0.0)); v = col(f"v{axis}")
    best = (np.inf,)
    starts = np.flatnonzero((mode == 4) & (np.abs(np.nan_to_num(col(f"cmd_a{axis}"))) > 0))
    segments = []
    k = 0
    while k < len(starts):
        s = starts[k]; e_ = s
        while k < len(starts) and starts[k] - e_ <= 1: e_ = starts[k]; k += 1
        segments.append((s - 10, min(len(t), e_ + 60)))
    for L in range(0, 25):
        for tau in np.arange(0.02, 0.8, 0.01):
            aeff = lagged(ua, L, tau); err = []
            for s, e_ in segments:
                pred = v[s] + np.cumsum(aeff[s:e_]) * dt
                err.append(pred - v[s:e_])
            e = float(np.mean(np.concatenate(err) ** 2))
            if e < best[0]: best = (e, L, tau)
    e, L, tau = best
    print(f"  {axis}: dead time {L*dt*1e3:3.0f} ms, tau {tau:.2f} s, velocity residual rms {np.sqrt(e):.3f} m/s over {len(segments)} pulses")

rest = (mode == 3) & (np.nan_to_num(col("sp_vz")) == 0) & (np.abs(np.nan_to_num(col("sp_vx"))) + np.abs(np.nan_to_num(col("sp_vy"))) == 0)
print(f"vertical drift while commanded vz = 0 and hovering: mean vz {np.nanmean(col('vz')[rest]):+.3f} m/s (NED, + = sinking), "
      f"altitude {-col('z')[0]:.2f} m at start -> {-col('z')[-1]:.2f} m at end of the test")
