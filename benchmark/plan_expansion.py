"""Plan the full compatible matrix, retaining quality gates and authenticated reuse.

This command writes manifests only. Scheduling, qualification, independent audits,
publication and any separately qualified GPU backend remain explicit later steps.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from queue_phases import NODES, immutable_write, scripts

from shapiq_benchmark.duplicates import payoff_fingerprint
from shapiq_benchmark.protocol import build_expansion
from shapiq_benchmark.runner import digest, identity, provenance

CONFIG_FIELDS = (
    "game_seeds",
    "seeds",
    "relative_budgets",
    "methods",
    "method_parameters",
    "targets",
    "min_players",
    "min_signal_ratio",
)


def recipe_key(kind: str, spec: dict, suite: dict) -> str:
    """Names do not change a recipe; all constructor and evaluation settings do."""
    return identity(
        {
            "kind": kind,
            "spec": {key: value for key, value in spec.items() if key != "id"},
            "config": {key: suite[key] for key in CONFIG_FIELDS},
        }
    )


def read_reuse(path: Path, expected_sha256: str) -> dict:
    """Accept only an explicitly pinned, independently reviewed completion ledger."""
    if digest(path) != expected_sha256:
        message = "Reuse manifest differs from the reviewed SHA256"
        raise ValueError(message)
    ledger = json.loads(path.read_text())
    if ledger.get("status") != "PASS" or ledger.get("version") != 1:
        message = "Reuse manifest has not passed independent review"
        raise ValueError(message)
    inputs = ledger.get("authenticated_inputs", {})
    if not inputs or any(digest(Path(name)) != expected for name, expected in inputs.items()):
        message = "Authenticated reuse inputs changed or are missing"
        raise ValueError(message)
    seen = set()
    for entry in ledger["entries"]:
        if (
            entry.get("complete") is not True
            or not entry.get("original_sources")
            or not entry.get("snapshot_id")
            or not entry.get("evidence")
            or any(name not in inputs for name in entry["evidence"])
            or entry["key"] != recipe_key(entry["kind"], entry["spec"], entry["config"])
            or entry["key"] in seen
        ):
            message = "Reuse entry is incomplete, ambiguous, or not authenticated"
            raise ValueError(message)
        seen.add(entry["key"])
    return ledger


def seed_registry(root: Path, ledger: dict, reuse_sha256: str) -> dict:
    """Seed immutable historical payoff identities using their published quality roles.

    No historical snapshot is edited. Native games cannot be identified by a
    full-table hash, and retain their separate recipe-level reuse evidence.
    """
    canonical = {}
    for item in ledger.get("canonical_snapshots", []):
        path = Path(item["path"])
        if ledger["authenticated_inputs"].get(str(path)) != item["sha256"]:
            message = "Historical canonical snapshot is not authenticated"
            raise ValueError(message)
        snapshot = json.loads(path.read_text())
        games = {game["id"]: game for game in snapshot["games"]}
        if snapshot["snapshot_id"] != item["snapshot_id"]:
            message = "Historical canonical snapshot identity changed"
            raise ValueError(message)
        for game_id in item["completed_game_ids"]:
            game = games[game_id]
            artifact = path.parent / game["artifact"]
            if str(artifact) not in ledger["authenticated_inputs"]:
                message = "Historical canonical artifact is not authenticated"
                raise ValueError(message)
            fingerprint = payoff_fingerprint(game, path.parent)
            if fingerprint is not None:
                key = f"{fingerprint}:{item['effective_roles'][game_id]}"
                canonical.setdefault(
                    key, {"game_id": game_id, "snapshot_id": snapshot["snapshot_id"]}
                )
    registry_path = root / "duplicate-games.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        if any(registry.get(key) != value for key, value in canonical.items()):
            message = "Historical duplicate registry entries changed"
            raise ValueError(message)
    else:
        immutable_write(registry_path, canonical)
    receipt = {
        "reuse_sha256": reuse_sha256,
        "canonical_entries": canonical,
        "policy": "Exact saved table plus effective published role; native games are not hashed",
    }
    path = root / "duplicate-seed.json"
    immutable_write(path, receipt)
    return {"path": str(path), "sha256": digest(path), "entries": len(canonical)}


def plan(
    root: Path,
    base: dict,
    *,
    reuse_path: Path,
    reuse_sha256: str,
    batch_size: int = 8,
    nodes: tuple[str, ...] = NODES,
) -> dict:
    """Write immutable ascending-dimension waves; exclude only verified complete recipes."""
    if type(batch_size) is not int or batch_size < 1:
        message = "Batch size must be a positive integer"
        raise ValueError(message)
    if not nodes or len(set(nodes)) != len(nodes) or not set(nodes) <= set(NODES):
        message = "Select unique nodes from the verified CPU pool"
        raise ValueError(message)
    source = provenance()
    if source.get("source_dirty") is not False or not source.get("git_commit"):
        message = "Planning requires a clean committed source"
        raise ValueError(message)
    ledger = read_reuse(reuse_path, reuse_sha256)
    reuse = {entry["key"]: entry for entry in ledger["entries"]}
    inventories = [build_expansion(base, structured=native) for native in (False, True)]
    all_keys = set()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    batches, waves, reused = [], [], []
    phase = 0
    for inventory in inventories:
        kind = "games" if inventory["games"] else "families"
        reference = "structured" if kind == "games" else "enumeration"
        inventory["duplicate_registry"] = str(root / "duplicate-games.json")
        immutable_write(root / f"inventory-{reference}.json", inventory)
        for count in sorted({spec["n_players"] for spec in inventory[kind]}):
            phase += 1
            selected = []
            for spec in inventory[kind]:
                if spec["n_players"] != count:
                    continue
                key = recipe_key(kind, spec, inventory)
                if key in all_keys:
                    message = "Full inventory contains duplicate semantic recipes"
                    raise ValueError(message)
                all_keys.add(key)
                if key in reuse:
                    reused.append({"id": spec["id"], "key": key, "phase": phase})
                else:
                    selected.append(spec)
            wave_id = f"{reference}-d{count}"
            suite = copy.deepcopy(inventory)
            suite[kind] = selected
            suite["name"] = f"matrix-{wave_id}"
            suite["phase_plan"] = {
                "wave_id": wave_id,
                "reference": reference,
                "n_players": count,
                "candidates": [
                    row
                    for row in inventory["phase_plan"]["candidates"]
                    if row["n_players"] == count
                ],
                "reused": [entry for entry in reused if entry["phase"] == phase],
            }
            immutable_write(root / f"phase-{phase}-inventory.json", suite)
            wave = {
                "phase": phase,
                "wave_id": wave_id,
                "reference": reference,
                "n_players": count,
                "recipes": len(selected),
                "reused": len(suite["phase_plan"]["reused"]),
            }
            waves.append(wave)
            for offset in range(0, len(selected), batch_size):
                slot = offset // batch_size
                name = f"phase-{phase}-cpu-{slot:04d}"
                directory = root / name
                directory.mkdir(exist_ok=True)
                batch = copy.deepcopy(suite)
                batch[kind] = selected[offset : offset + batch_size]
                batch["name"] = name
                batch["phase_plan"] = {
                    "inventory_sha256": identity(suite["phase_plan"]),
                    "selected_recipe_ids": [spec["id"] for spec in batch[kind]],
                    "note": "Disjoint full-matrix wave; runtime qualification remains mandatory.",
                }
                immutable_write(directory / "suite.json", batch)
                batches.append(
                    {
                        "id": name,
                        "phase": phase,
                        "device": "cpu",
                        "node": nodes[slot % len(nodes)],
                        "directory": str(directory),
                        "recipes": len(batch[kind]),
                        "suite_sha256": identity(batch),
                        "wave_id": wave_id,
                        "reference": reference,
                        "n_players": count,
                    }
                )
    if set(reuse) - all_keys:
        message = "Reuse manifest contains recipes outside this full inventory"
        raise ValueError(message)
    campaign = {
        "source": source,
        "scripts": scripts(),
        "batches": batches,
        "waves": waves,
        "reuse": {"path": str(reuse_path.resolve()), "sha256": reuse_sha256, "recipes": reused},
        "planner_sha256": digest(Path(__file__)),
        "duplicate_seed": seed_registry(root, ledger, reuse_sha256),
        "policy": "full compatible matrix; four instances; quality-v2; CPU until separately qualified",
    }
    # Recheck evidence before sealing a campaign; a prior successful read is not a lease.
    read_reuse(reuse_path, reuse_sha256)
    if provenance() != source:
        message = "Source changed while planning"
        raise ValueError(message)
    immutable_write(root / "campaign.json", campaign)
    for number, node in sorted({(b["phase"], b["node"]) for b in batches}):
        immutable_write(
            root / f"phase-{number}-cpu-{node}.json",
            [b for b in batches if b["phase"] == number and b["node"] == node],
        )
    return campaign


def main() -> None:
    """Produce a reviewable campaign without contacting Slurm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--base", type=Path, default=Path(__file__).parent / "suites/all-families.json"
    )
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--reuse-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nodes", nargs="+", default=list(NODES))
    args = parser.parse_args()
    campaign = plan(
        args.root,
        json.loads(args.base.read_text()),
        reuse_path=args.reuse,
        reuse_sha256=args.reuse_sha256,
        batch_size=args.batch_size,
        nodes=tuple(args.nodes),
    )
    print(  # noqa: T201 -- CLI summary
        json.dumps(
            {
                "waves": len(campaign["waves"]),
                "batches": len(campaign["batches"]),
                "reused": len(campaign["reuse"]["recipes"]),
            }
        )
    )


if __name__ == "__main__":
    main()
