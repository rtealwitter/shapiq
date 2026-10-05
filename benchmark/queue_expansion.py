"""Plan shared CPU arrays behind strict stage barriers; submit each job at most once.

An ambiguous scheduler response is deliberately not retried. Inspect Slurm and
reconcile the recorded intent first. Scientific audits are a separately reviewed,
hash-pinned entrypoint; their nonzero exit blocks the next wave.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: str | Path) -> str:
    """Hash a file with bounded memory using the system Python bootstrap."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value: dict) -> str:
    """Match runner identity without importing machine-learning packages."""
    # Same JSON identity as shapiq_benchmark.runner, without importing ML packages.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path: Path, value: dict) -> None:
    """Replace a journal only after its new contents are complete."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_plan(campaign_path: Path, settings: dict) -> dict:
    """Build CPU-bounded stage barriers from authenticated scientific batches."""
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    workers, cap = settings["workers"], settings["cpu_cap"]
    nodes = settings["nodes"]
    if workers != 16 or cap != 128 or not nodes or len(set(nodes)) != len(nodes):
        msg = "This reviewed policy uses 16 workers and 128 total CPU cores"
        raise ValueError(msg)
    if len(nodes) > cap // workers or set(nodes) - {"himem01", "himem02", "gpu15"}:
        msg = "Only the currently qualified CPU node pool is enabled"
        raise ValueError(msg)
    if campaign["source"]["source_dirty"] or not campaign["source"]["git_commit"]:
        msg = "Freeze and commit the scientific source before planning submission"
        raise ValueError(msg)
    slots = {
        node: (cap // workers) // len(nodes) + (i < (cap // workers) % len(nodes))
        for i, node in enumerate(nodes)
    }
    if slots.get("gpu15", 0) > 3:
        msg = "gpu15 supports at most three concurrent 208-GiB allocations"
        raise ValueError(msg)
    pins = dict(settings["pins"])
    pins[str(campaign_path)] = digest(campaign_path)
    related = [
        campaign["reuse"],
        campaign["duplicate_seed"],
        {
            "path": str(Path(settings["source"]) / "benchmark/plan_expansion.py"),
            "sha256": campaign["planner_sha256"],
        },
    ]
    for item in related:
        if digest(item["path"]) != item["sha256"]:
            msg = "Reuse, duplicate seed or planner authentication changed"
            raise ValueError(msg)
        pins[item["path"]] = item["sha256"]
    for name, expected in campaign["scripts"].items():
        path = Path(settings["source"]) / "benchmark" / name
        if digest(path) != expected:
            msg = "Scientific launch scripts changed"
            raise ValueError(msg)
        pins[str(path)] = expected
    jobs = [
        {
            "key": "warm",
            "step": "warm",
            "dependencies": [],
            "cpus": 1,
            "memory_gb": 32,
            "hours": 12,
            "target": "all",
        }
    ]
    previous = ["warm"]
    for phase in sorted({b["phase"] for b in campaign["batches"]}):
        stages = {}
        for step in ("prepare", "evaluate"):
            stages[step] = []
            for node in nodes:
                batches = [
                    b for b in campaign["batches"] if b["phase"] == phase and b["node"] == node
                ]
                if not batches:
                    continue
                if any(b["device"] != "cpu" for b in batches) or len(batches) > 1000:
                    msg = "CPU-only batches must fit the scheduler array limit"
                    raise ValueError(msg)
                manifest = campaign_path.parent / f"phase-{phase}-cpu-{node}.json"
                if json.loads(manifest.read_text()) != batches:
                    msg = "Group manifest differs from campaign"
                    raise ValueError(msg)
                pins[str(manifest)] = digest(manifest)
                for batch in batches:
                    suite = Path(batch["directory"]) / "suite.json"
                    if identity(json.loads(suite.read_text())) != batch["suite_sha256"]:
                        msg = "A recipe suite changed"
                        raise ValueError(msg)
                    pins[str(suite)] = digest(suite)
                key = f"wave-{phase}-{step}-{node}"
                stages[step].append(key)
                jobs.append(
                    {
                        "key": key,
                        "step": step,
                        "target": str(manifest),
                        "dependencies": previous if step == "prepare" else stages["prepare"],
                        "node": node,
                        "cpus": workers,
                        "memory_gb": 208,
                        "hours": 72,
                        "array": f"0-{len(batches) - 1}%{slots[node]}",
                    }
                )
        expected = [b for b in campaign["batches"] if b["phase"] == phase]
        if any(b["node"] not in nodes for b in expected) or not stages["evaluate"]:
            msg = "Every planned batch must belong to the qualified node pool"
            raise ValueError(msg)
        key = f"wave-{phase}-audit"
        jobs.append(
            {
                "key": key,
                "step": "audit",
                "target": str(phase),
                "dependencies": stages["evaluate"],
                "cpus": 1,
                "memory_gb": 32,
                "hours": 24,
            }
        )
        previous = [key]
    return {
        **settings,
        "campaign": str(campaign_path),
        "campaign_identity": identity(campaign),
        "pins": pins,
        "concurrency_slots": slots,
        "jobs": jobs,
    }


def verify(plan: dict) -> None:
    """Reject drafts, changed inputs and unpinned executable entrypoints."""
    if plan.get("status") != "READY":
        msg = "Draft launch settings cannot be submitted or finalized"
        raise ValueError(msg)
    for path, expected in plan["pins"].items():
        if not expected or digest(path) != expected:
            msg = f"Missing or changed launch input: {path}"
            raise ValueError(msg)
    for name in (
        "warm_script",
        "audit_script",
        "run_script",
        "wrapper",
        "stage_script",
        "launcher",
    ):
        if plan[name] not in plan["pins"]:
            msg = f"Unpinned executable: {name}"
            raise ValueError(msg)


def submit(plan_path: Path) -> None:
    """Journal scheduler intent before submission and never retry an ambiguous call."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    verify(plan)
    source = Path(plan["source"])
    commit = subprocess.check_output(  # noqa: S603 -- fixed read-only git command
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    dirty = subprocess.check_output(  # noqa: S603 -- fixed read-only git command
        ["/usr/bin/git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
    ).strip()
    campaign = json.loads(Path(plan["campaign"]).read_text())
    if dirty or commit != campaign["source"]["git_commit"]:
        msg = "Scientific checkout is not the clean frozen campaign revision"
        raise ValueError(msg)
    root = Path(plan["campaign"]).parent
    lock = root / ".submit-lock"
    lock.mkdir()  # Atomic across hosts; never reclaim a potentially live owner.
    try:
        journal_path = root / "jobs.json"
        journal = (
            json.loads(journal_path.read_text())
            if journal_path.exists()
            else {
                "plan_sha256": plan["campaign_identity"],
                "launch_plan_sha256": digest(plan_path),
                "jobs": {},
            }
        )
        if journal["plan_sha256"] != plan["campaign_identity"] or journal[
            "launch_plan_sha256"
        ] != digest(plan_path):
            msg = "Submission journal belongs to a different frozen plan"
            raise ValueError(msg)
        (root / "logs").mkdir(exist_ok=True)
        (root / "audits").mkdir(exist_ok=True)
        for job in plan["jobs"]:
            key = job["key"]
            existing = journal["jobs"].get(key)
            if existing:
                if existing["status"] != "SUBMITTED":
                    msg = f"Reconcile ambiguous submission before continuing: {key}"
                    raise RuntimeError(msg)
                continue
            dependencies = [journal["jobs"][k]["job_id"] for k in job["dependencies"]]
            if job["step"] == "audit":
                prefix = f"wave-{job['target']}-"
                evidence = {
                    "plan_sha256": journal["plan_sha256"],
                    "launch_plan_path": str(plan_path),
                    "launch_plan_sha256": journal["launch_plan_sha256"],
                    "jobs": {
                        k: v
                        for k, v in journal["jobs"].items()
                        if k.startswith(prefix) and ("-prepare-" in k or "-evaluate-" in k)
                    },
                }
                evidence_path = root / "audits" / f"phase-{job['target']}-jobs.json"
                if evidence_path.exists() and json.loads(evidence_path.read_text()) != evidence:
                    msg = "Wave scheduler evidence changed"
                    raise ValueError(msg)
                write(evidence_path, evidence)
            command = [
                "/usr/bin/sbatch",
                "--parsable",
                "--partition=main",
                "--account=standard",
                "--qos=normal",
                "--nodes=1",
                "--ntasks=1",
                f"--job-name=matrix-{plan['campaign_identity'][:10]}-{key}",
                f"--cpus-per-task={job['cpus']}",
                f"--mem={job['memory_gb']}G",
                f"--time={job['hours']}:00:00",
                "--hint=nomultithread",
                f"--output={root}/logs/{key}-%A_%a.out",
                f"--error={root}/logs/{key}-%A_%a.err",
            ]
            if "node" in job:
                command.append(f"--nodelist={job['node']}")
            if "array" in job:
                command.append(f"--array={job['array']}")
            if dependencies:
                command.append("--dependency=afterok:" + ":".join(dependencies))
            command += [
                plan["wrapper"],
                str(plan_path),
                digest(plan_path),
                job["step"],
                job["target"],
            ]
            journal["jobs"][key] = {"status": "INTENT", "command": command}
            write(journal_path, journal)
            result = subprocess.run(  # noqa: S603 -- fixed scheduler plus authenticated plan arguments
                command, cwd=source, text=True, capture_output=True, check=False
            )
            record = journal["jobs"][key]
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            job_id = result.stdout.strip().split(";")[0]
            if result.returncode or not job_id.isdecimal():
                write(journal_path, journal)
                msg = f"Submission needs reconciliation: {key}"
                raise RuntimeError(msg)
            record.update(status="SUBMITTED", job_id=job_id)
            write(journal_path, journal)
    finally:
        lock.rmdir()


def main() -> None:
    """Write a reviewable plan or explicitly submit its dependency graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit(args.plan)
    else:
        plan = make_plan(args.campaign, json.loads(args.settings.read_text()))
        verify(plan)
        with args.plan.open("x") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
