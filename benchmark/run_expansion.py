"""Execute one frozen wave stage using byte-verified node-local dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path: str | Path) -> str:
    """Hash bootstrap inputs without loading scientific packages."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(plan_path: str, plan_sha: str, step: str, target: str) -> None:
    """Authenticate inputs, stage dependencies, execute one stage and clean local copies."""
    if sys.flags.optimize or digest(plan_path) != plan_sha:
        msg = "Assertions or immutable plan authentication disabled"
        raise ValueError(msg)
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("status") != "READY":
        msg = "Draft launch plan cannot execute"
        raise ValueError(msg)
    for path, expected in plan["pins"].items():
        if not expected or digest(path) != expected:
            msg = f"Launch input changed: {path}"
            raise ValueError(msg)
    if not any(j["step"] == step and j["target"] == target for j in plan["jobs"]):
        msg = "Unplanned stage"
        raise ValueError(msg)
    source = Path(plan["source"])
    root = Path(plan["campaign"]).parent
    cache = root.parent / "cache"
    temporary = cache / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": f"{source}/src:{source}/benchmark",
        "CUDA_VISIBLE_DEVICES": "",
        "XDG_CACHE_HOME": str(cache),
        "TMPDIR": str(temporary),
        "JOBLIB_TEMP_FOLDER": str(temporary),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCH_HOME": plan["model_cache"]["TORCH_HOME"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    environment["HF_HOME"] = plan["model_cache"]["HF_HOME"]
    environment["TABPFN_MODEL_CACHE_DIR"] = plan["model_cache"]["TABPFN_MODEL_CACHE_DIR"]
    staged = subprocess.check_output(  # noqa: S603 -- hash-pinned runtime stager and source paths
        [sys.executable, plan["stage_script"], str(source), plan["campaign"]],
        cwd=source,
        env=environment,
        text=True,
    ).strip()
    stage = Path(staged).parents[2]
    if (
        stage.parent != Path("/tmp")  # noqa: S108 -- required private, owned runtime staging root
        or not stage.name.startswith("shapiq-runtime-")
        or stage.is_symlink()
        or stage.stat().st_uid != os.getuid()
    ):
        msg = "Unexpected local runtime path"
        raise ValueError(msg)
    try:
        if step in ("prepare", "evaluate"):
            warm = json.loads((root / "warm-cache.json").read_text())
            if warm["status"] != "PASS" or warm["campaign_sha256"] != digest(plan["campaign"]):
                msg = "Serial cache warming did not complete for this campaign"
                raise ValueError(msg)
            for path, expected in warm["cache_files"].items():
                if digest(path) != expected:
                    msg = "A warmed cache file changed"
                    raise ValueError(msg)
            command = [
                str(source / "benchmark/phase_batch.py"),
                target,
                step,
                "--workers",
                str(plan["workers"]),
                "--index",
                os.environ["SLURM_ARRAY_TASK_ID"],
            ]
        elif step == "warm":
            command = [plan["warm_script"], str(root)]
        elif step == "audit":
            command = [plan["audit_script"], str(root), "--phase", target]
        else:
            msg = "Unknown stage"
            raise ValueError(msg)
        subprocess.run(  # noqa: S603 -- pinned stage scripts and verified runtime executable
            ["/usr/bin/srun", "--cpu-bind=cores", staged, *command],
            cwd=source,
            env=environment,
            check=True,
        )
    finally:
        shutil.rmtree(stage)


if __name__ == "__main__":
    run(*sys.argv[1:])
