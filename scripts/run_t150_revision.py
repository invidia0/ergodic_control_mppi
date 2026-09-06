"""Run the frozen T150 numerical revision, stopping dependent stages on any failure.

Prepare the bundle's config.yaml, clutter/maps.json and open/maps.json first. Map
manifests must point to copied arrays inside the bundle. `plan` prints the exact
commands; `run` executes them sequentially and checks artifacts before marking success.
Laptop timing and SITL are separate, supervised validation sessions.
"""

import argparse
import csv
import hashlib
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import yaml
from ergodic_control_mppi.experiments.common import artifact_digests, execution_record
from ergodic_control_mppi.experiments.uav_ablation import FINAL_ARMS, _BY_NAME


def stages(bundle: Path):
    """Yield commands, expected CSV row counts and required artifacts in dependency order."""
    python = sys.executable
    config = bundle / "config.yaml"
    clutter, opened = bundle / "clutter/maps.json", bundle / "open/maps.json"
    base = ["--config", str(config), "--device", "gpu"]
    # First, not with the audits it belongs to: the six 750,000-step runs are ~80% of the
    # campaign's wall clock and this 2,000-step run is the only thing that costs them. Run
    # it before the gates so `main`'s duration is known while the GPU is still uncommitted.
    output = bundle / "audit/pilot.csv"
    yield "pilot", [python, "scripts/theory_audit.py", "run", *base, "--maps", str(clutter),
        "--seeds", "12", "--steps", "2000", "--stride", "20", "--output", str(output)], 72, [
        output, output.with_name("pilot_paths.npz")]
    for tier, width, seeds, maps in (("clutter", 36, 6, clutter),
                                    ("clutter", 9, 6, clutter), ("open", 12, 12, opened)):
        output = bundle / f"verification/{tier}_batch{width}.json"
        yield f"branch_{tier}_{width}", [python, "scripts/final_ablation.py", "verify", *base,
            "--maps", str(maps), "--seeds", str(seeds), "--steps", "2000",
            "--verify-lanes", str(width), "--output", str(output)], 0, [output]
    mechanism = [a for a in FINAL_ARMS if a == "baseline" or _BY_NAME[a][0] not in
                 {"T", "K", "alpha", "exploration", "lam_max", "track_weight",
                  "reference_speed", "penalty_scale", "boundary_scale"}]
    for tier, maps, seeds, count in (("clutter", clutter, 6, 1476), ("open", opened, 12, 276)):
        output = bundle / tier / "ablation.csv"
        command = [python, "scripts/final_ablation.py", "run", *base, "--maps", str(maps),
                   "--seeds", str(seeds), "--first-seed", "43", "--steps", "20000",
                   "--output", str(output), "--stop-file", str(bundle / "STOP")]
        if tier == "open":
            command += ["--arms", ",".join(mechanism)]
        yield f"ablation_{tier}", command, count, [output]
    # Historical competitor rows lack run-source provenance. Re-measure all five;
    # no archived row is assigned a new hash or silently counted as a current run.
    for tier, maps, seeds, count in (("clutter", clutter, 6, 180), ("open", opened, 12, 60)):
        output = bundle / tier / "baselines.csv"
        yield f"baselines_{tier}", [python, "-m", "ergodic_control_mppi.experiments.baselines",
            "--tier", tier, "--config", str(config), "--maps", str(maps), "--steps", "20000",
            "--seeds", ",".join(map(str, range(43, 43 + seeds))), "--output", str(output)], count, [output, output.with_suffix(".fidelity.json")]
    audits = [("main", "run", 750000, config, []), ("short", "run", 20000, config, []),
              ("ideal", "ideal", 750000, config, [])]
    audits += [(f"nt_{n}_{t}", "run", 100000, bundle / f"audit/config_{n}_{t}.yaml", [])
               for n, t in ((250, 150), (125, 150), (500, 150), (1000, 150), (250, 75), (250, 350))]
    audits += [(f"inits_start{i}", "run", 750000, config, ["--inits", "4", "--start-index", str(i)])
               for i in range(4)]
    for name, kind, steps, variant, extra in audits:
        output = bundle / f"audit/{name}.csv"
        yield name, [python, "scripts/theory_audit.py", kind, "--config", str(variant),
            "--device", "gpu", "--maps", str(clutter), "--seeds", "12", "--steps", str(steps),
            "--stride", "20", "--output", str(output), *extra], 72, [output, output.with_name(name + "_paths.npz")]
        if name == "main":
            tv = bundle / "audit/tv_resolution.csv"
            yield "tv_resolution", [python, "scripts/theory_audit.py", "sweep", "--config", str(config),
                "--maps", str(clutter), "--paths", str(output.with_name("main_paths.npz")),
                "--output", str(tv)], 4032, [tv, tv.with_name("tv_resolution_split.csv"),
                                              tv.with_suffix(".artifacts.json")]
    captures = bundle / "captures"
    yield "captures", [python, "scripts/mechanism_captures.py", "--config", str(config),
        "--out", str(captures), "--axis", "plan_gain", "--levels", "0,6,10", "--seed", "43",
        "--steps", "20000", "--freeze", "12000"], None, [captures / f"plan_gain_{g}_s43.npz" for g in (0, 6, 10)]


def check_artifacts(paths, count):
    """Require complete finite primary observations or a passing branch-gate JSON."""
    for path in paths:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing or empty artifact: {path}")
    if paths[-1].name.endswith(".artifacts.json"):
        if json.loads(paths[-1].read_text()) != artifact_digests(paths[:-1]):
            raise ValueError(f"artifact receipt does not match outputs: {paths[-1]}")
    if count == 0:
        if json.loads(paths[0].read_text()).get("passed") is not True:
            raise ValueError(f"branch verification failed: {paths[0]}")
    elif count is not None:
        with paths[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != count:
            raise ValueError(f"{paths[0]}: {len(rows)} rows, expected {count}")
        keys = [k for k in ("method", "arm", "obs_num", "map_seed", "map", "seed",
                            "seed_a", "seed_b", "factor", "k", "execution") if k in rows[0]]
        identities = [tuple(r[k] for k in keys) for r in rows]
        if len(set(identities)) != len(rows):
            raise ValueError(f"{paths[0]}: duplicate identities")
    return artifact_digests(paths)


def main():
    """Run one campaign under an exclusive lock, with per-stage logs and checked receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--bundle", type=Path, default=Path("results/uav/T150"))
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if yaml.safe_load((bundle / "config.yaml").read_text())["mppi"]["T"] != 150:
        parser.error("the frozen profile must have T=150")
    frozen = json.loads((bundle / "freeze.json").read_text())
    for name, digest in frozen.items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen input changed: {name}")
    scheduled = list(stages(bundle))
    if args.action == "plan":
        import shlex
        for name, command, count, _ in scheduled:
            print(f"{name}: expected rows={count}\n{shlex.join(command)}")
        return
    log_dir = bundle / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (bundle / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while args.wait_for_pid and Path(f"/proc/{args.wait_for_pid}").exists():
            print(f"waiting for identified prior audit PID {args.wait_for_pid}", flush=True)
            time.sleep(30)
        source = execution_record("scripts/run_t150_revision.py", "gpu")
        for name, command, count, paths in scheduled:
            if (bundle / "STOP").exists():
                raise SystemExit("STOP present; no dependent stage launched")
            # Memory and process identities, not utilization, identify other compute jobs.
            while True:
                query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
                busy = [line for line in query.stdout.splitlines()
                        if len(line.split(",")) == 2 and line.split(",")[1].strip().isdigit()
                        and int(line.split(",")[1]) > 1000]
                if not busy:
                    break
                print(f"waiting for other GPU compute processes: {busy}", flush=True)
                if (bundle / "STOP").exists():
                    raise SystemExit("STOP present")
                time.sleep(30)
            receipt = log_dir / f"{name}.json"
            record = {"host": socket.gethostname(), "source": source, "command": command,
                      "config": str(bundle / "config.yaml"), "started": time.time()}
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous.get("source") == source and previous.get("command") == command and previous.get("status") == "complete":
                    if previous["artifacts"] == check_artifacts(paths, count):
                        print(f"SKIP verified stage {name}", flush=True)
                        continue
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            print(f"START {name}: {command}; output {paths[0]}", flush=True)
            with (log_dir / f"{name}.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, JAX_PLATFORMS="cpu" if name == "tv_resolution" else "cuda"))
            record.update(exit_status=result.returncode, finished=time.time(), status="failed")
            if result.returncode == 0:
                try:
                    record.update(artifacts=check_artifacts(paths, count), status="complete")
                except ValueError as error:
                    record["integrity_error"] = str(error)
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            if record["status"] != "complete":
                raise SystemExit(f"FAILED {name}; dependent stages blocked; see {receipt}")
            print(f"COMPLETE {name}: {record['finished'] - record['started']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
