"""Serialize shipped-loader cache writes before parallel benchmark preparation.

This does not change datasets, fit models, or populate game payoff caches. Existing
TabPFN checkpoint files must already be present; no model downloads are hidden.
Wine's shipped loader still performs remote reads on each call.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from shapiq_benchmark.datasets import DATASETS, load_raw_dataset
from shapiq_benchmark.models import _tabpfn_checkpoint
from shapiq_benchmark.runner import digest, identity, provenance
from shapiq_games.datasets._all import SHAPIQ_DATASETS_FOLDER


def arrays_identity(loaded: tuple) -> str:
    """Bind loader shapes, dtypes, numeric bytes and ordered feature names."""
    x, y, names = loaded
    value = hashlib.sha256()
    for value_array in (x, y):
        array = np.asarray(value_array)
        value.update(str((array.shape, array.dtype)).encode())
        value.update(
            json.dumps(array.tolist(), allow_nan=True, separators=(",", ":")).encode()
            if array.dtype.kind in "OU"
            else np.ascontiguousarray(array).tobytes()
        )
    value.update(json.dumps(names).encode())
    return value.hexdigest()


def warm(root: Path) -> None:
    """Finish loader cache writes serially and record stable reload identities."""
    campaign_path = root / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    if provenance() != campaign["source"]:
        msg = "Wrong source or environment for cache warming"
        raise ValueError(msg)
    names, tabpfn_tasks = set(), set()
    for batch in campaign["batches"]:
        suite = json.loads((Path(batch["directory"]) / "suite.json").read_text())
        if identity(suite) != batch["suite_sha256"]:
            msg = "Changed recipe suite"
            raise ValueError(msg)
        for spec in [*suite.get("families", []), *suite.get("games", [])]:
            if spec.get("dataset") in DATASETS:
                names.add(spec["dataset"])
                if (
                    spec.get("model_profile") == "tabpfn_prediction"
                    or spec.get("family") == "tabpfn"
                ):
                    tabpfn_tasks.add(DATASETS[spec["dataset"]]["task"])
    warmed = {}
    for name in sorted(names):
        load_raw_dataset(name)  # Finish first-download CSV writes before comparing reloads.
        first = arrays_identity(load_raw_dataset(name))
        second = arrays_identity(load_raw_dataset(name))
        if first != second:
            msg = f"Shipped loader representation changed across serial reloads: {name}"
            raise ValueError(msg)
        warmed[name] = second
    cache_files = {
        str(path.resolve()): digest(path) for path in SHAPIQ_DATASETS_FOLDER.glob("*.csv")
    }
    for task in sorted(tabpfn_tasks):
        path = _tabpfn_checkpoint(task).resolve()
        cache_files[str(path)] = digest(path)
    if provenance() != campaign["source"]:
        msg = "Cache warming changed scientific source provenance"
        raise ValueError(msg)
    receipt = {
        "status": "PASS",
        "campaign_sha256": digest(campaign_path),
        "datasets": warmed,
        "cache_files": cache_files,
        "source": campaign["source"],
        "note": "Serial cache warm only; uncached shipped loaders still access their URLs.",
    }
    destination = root / "warm-cache.json"
    if destination.exists():
        if json.loads(destination.read_text()) != receipt:
            msg = "Existing warm-cache receipt differs"
            raise ValueError(msg)
    else:
        with destination.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    warm(Path(sys.argv[1]))
