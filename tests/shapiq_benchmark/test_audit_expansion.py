"""Independent wave checks reject plausible but incomplete or altered evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shapiq import InteractionValues
from shapiq_benchmark.campaign import _prepared_suite
from shapiq_benchmark.results_io import Checkpoint
from shapiq_benchmark.runner import score

spec = importlib.util.spec_from_file_location(
    "audit_expansion", Path(__file__).resolve().parents[2] / "benchmark/audit_expansion.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_independent_sparse_and_zero_signal_scores():
    game = {
        "index": "SII",
        "n_players": 11,
        "order": 2,
        "metadata": {"payoff_std": 2.0},
        "truth": {"coordinates": [[0], [1, 2]], "values": [2.0, 3.0], "energy": 13.0},
    }
    estimate = InteractionValues(
        values=np.array([1.0, 4.0]),
        index="SII",
        max_order=2,
        min_order=1,
        n_players=11,
        interaction_lookup={(0,): 0, (3, 4): 1},
        baseline_value=0,
    )
    row = {
        **score(estimate, game),
        "estimate": {"coordinates": [[0], [3, 4]], "values": [1.0, 4.0]},
    }
    audit.rescore(row, game)
    assert row["nmse"] == 2.0
    for key in ("nmse", "mse", "truth_energy"):
        changed = copy.deepcopy(row)
        changed[key] += 0.01
        with pytest.raises(ValueError):
            audit.rescore(changed, game)
    game["truth"]["values"] = [0.0, 0.0]
    game["truth"]["energy"] = 0.0
    row.update(score(estimate, game))
    audit.rescore(row, game)
    row["nmse"] = 0.0
    with pytest.raises(ValueError):
        audit.rescore(row, game)


@pytest.mark.parametrize(
    "coordinates,values",
    [([[0], [0]], [1.0, 2.0]), ([[11]], [1.0]), ([[1, 0]], [1.0]), ([[0]], [float("inf")])],
)
def test_invalid_sparse_vectors(coordinates, values):
    with pytest.raises(ValueError):
        audit.coefficients({"coordinates": coordinates, "values": values}, 11, 2)


def test_evidence_lists_and_mutation(tmp_path):
    path = tmp_path / "group.json"
    path.write_text('[{"id":"a"}]')
    evidence = audit.Evidence()
    assert evidence.read(path) == [{"id": "a"}]
    evidence.stable()
    path.write_text("[]")
    with pytest.raises(ValueError):
        evidence.stable()


def test_scheduler_requires_exact_completed_task(monkeypatch):
    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="", stderr="Invalid job id specified", returncode=1),
    )
    monkeypatch.setattr(
        audit.subprocess, "check_output", lambda *a, **k: "123_0|124|COMPLETED|0:0|16|himem01|\n"
    )
    assert audit.terminal("123_0", "himem01")["worker_job_id"] == "124"
    with pytest.raises(ValueError):
        audit.terminal("123", "himem01")
    with pytest.raises(ValueError):
        audit.terminal("123_0", "himem02")
    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="123_0|RUNNING", stderr="", returncode=0),
    )
    with pytest.raises(ValueError):
        audit.terminal("123_0", "himem01")


def test_duplicate_requires_completed_same_role_canonical():
    old = {"old": {"fingerprint": "abc", "role": "core"}}
    new = {"new": {"fingerprint": "abc", "role": "core"}}
    audit.check_aliases(old, new, {"new": "old"})
    with pytest.raises(ValueError):
        audit.check_aliases(old, new, {})
    with pytest.raises(ValueError):
        audit.check_aliases(old, new, {"new": "missing"})
    new["new"]["role"] = "control"
    audit.check_aliases(old, new, {})
    with pytest.raises(ValueError):
        audit.check_aliases(old, new, {"new": "old"})


def test_known_failures_never_accept_generic_validation_error():
    execution = {"timeout": 600, "memory_gb": 12}
    row = {
        "status": "failed",
        "game_id": "g",
        "method": "KernelSHAP",
        "budget": 11,
        "error": "TimeoutError: worker wall-time limit exceeded.",
        "seconds": None,
        "queries": None,
        "requested_queries": None,
        "wall_seconds": 601,
    }
    assert audit.known_failure(row, execution) == "timeout"
    row["wall_seconds"] = 5
    with pytest.raises(ValueError):
        audit.known_failure(row, execution)
    row["error"] = "ValueError: arbitrary numerical failure"
    with pytest.raises(ValueError, match="Unknown failure"):
        audit.known_failure(row, execution)


def qualification_fixture(tmp_path, failure=None):
    source = {"git_commit": "fixture", "source_dirty": False}
    spec = {"id": "cluster", "family": "cluster", "n_players": 11}
    original = {"families": [spec], "game_seeds": [0, 1, 2, 3]}
    instances = []
    directory = tmp_path / "qualification"
    directory.mkdir()
    for seed in range(4):
        result = failure or {"status": "measured", "projected_seconds": 2.0, "n_players": 11}
        key = {"spec": spec, "kind": "families", "seed": seed, "source": source}
        (directory / f"{audit.identity(key)}.json").write_text(
            json.dumps({"identity": key, "result": result})
        )
        instances.append({"seed": seed, **result})
    qualified = {**original, "families": [] if failure else [spec], "preparation_exclusions": []}
    if failure:
        qualified["preparation_exclusions"] = [
            {"spec": spec, "kind": "families", "reason": "preflight_failed", "instances": instances}
        ]
    qualified["preparation_preflight"] = {
        "source": source,
        "requested_suite_sha256": audit.identity(original),
        "maximum_seconds_per_instance": 28800,
        "pilot_timeout_seconds": 480,
        "structured_timeout_seconds": 480,
        "families": [
            {
                "id": "cluster",
                "kind": "families",
                "instances": instances,
                "status": "excluded" if failure else "qualified",
            }
        ],
    }
    (tmp_path / "qualified-suite.json").write_text(json.dumps(qualified))
    (tmp_path / "qualification-decision.json").write_text(
        json.dumps(
            {
                "source": source,
                "requested_suite_sha256": audit.identity(original),
                "qualified_suite_sha256": audit.identity(qualified),
            }
        )
    )
    return original, source


def test_all_four_qualification_seeds_required(tmp_path):
    original, source = qualification_fixture(tmp_path)
    audit.qualification(tmp_path, original, source, audit.Evidence())
    next((tmp_path / "qualification").glob("*.json")).unlink()
    with pytest.raises((KeyError, ValueError)):
        audit.qualification(tmp_path, original, source, audit.Evidence())


def test_clustering_screen_has_actual_feature_evidence(tmp_path):
    failure = {
        "status": "failed",
        "reason": "insufficient_eligible_clustering_features",
        "error_type": "QualityExclusion",
        "details": {
            "requested_players": 11,
            "available_features": 2,
            "eligible_feature_indices": [0, 2],
            "training_rows": [0, 1, 2],
        },
    }
    original, source = qualification_fixture(tmp_path, failure)
    audit.qualification(tmp_path, original, source, audit.Evidence())
    failure["details"]["available_features"] = 12
    # Consistently alter the pilot and decision: evidence itself must still be valid.
    shutil.rmtree(tmp_path / "qualification")
    original, source = qualification_fixture(tmp_path, failure)
    with pytest.raises(ValueError, match="clustering feature"):
        audit.qualification(tmp_path, original, source, audit.Evidence())


def test_table_quality_spectrum_and_eligibility(tmp_path):
    n = 11
    values = np.array([float(mask & 1) for mask in range(2**n)])
    artifact = tmp_path / "g.npz"
    np.savez(artifact, values=values, evaluation_seconds=np.full(len(values), 0.001))
    quality = audit.payoff_diagnostics(values, n)
    game = {
        "id": "g",
        "artifact": "g.npz",
        "n_players": n,
        "order": 1,
        "index": "SV",
        "truth": {"coordinates": [[0]], "values": [1.0], "energy": 1.0, "baseline": 0.0},
        "metadata": {
            "payoff_std": 0.5,
            "game_quality": quality,
            "fourier_spectrum": audit.fourier_spectrum(values, n),
            "signal_ratio": 2 / np.sqrt(n),
            "score_eligible": True,
        },
    }
    snapshot = {
        "snapshot_id": "fixture",
        "suite": {"min_signal_ratio": 1e-6},
        "games": [game],
        "artifacts": {"g.npz": audit.Evidence().track(artifact)},
    }
    audit.inspect_games(snapshot, tmp_path, audit.Evidence())
    game["metadata"]["game_quality"]["role"] = "core"
    with pytest.raises(ValueError, match="quality/Fourier"):
        audit.inspect_games(snapshot, tmp_path, audit.Evidence())


def test_fully_excluded_batch_still_needs_complete_authenticated_qualification(
    tmp_path, monkeypatch
):
    batch_root = tmp_path / "batch"
    batch_root.mkdir()
    failure = {"status": "failed", "reason": "pilot_timeout", "limit_seconds": 480}
    original, source = qualification_fixture(batch_root, failure)
    (batch_root / "suite.json").write_text(json.dumps(original))
    qualified = json.loads((batch_root / "qualified-suite.json").read_text())
    (batch_root / "excluded.json").write_text(
        json.dumps(
            {
                "suite_sha256": audit.identity(qualified),
                "reason": "Every recipe failed the declared preparation gate",
            }
        )
    )
    for step in ("prepare", "evaluate"):
        (batch_root / f"{step}-resource-policy.json").write_text(
            json.dumps({"workers": 16, "timing_profile": "diagnostic", "allocation": "shared-cpu"})
        )
        (batch_root / f"{step}-allocation.json").write_text(
            json.dumps(
                {"hostname": "himem01", "cpu_model": "AMD EPYC 9754", "affinity": list(range(16))}
            )
        )
    ledger = tmp_path / "reuse.json"
    ledger.write_text(
        json.dumps({"status": "PASS", "authenticated_inputs": {}, "canonical_snapshots": []})
    )
    ledger_hash = audit.Evidence().track(ledger)
    seed = tmp_path / "duplicate-seed.json"
    seed.write_text(json.dumps({"reuse_sha256": ledger_hash, "canonical_entries": {}}))
    campaign = {
        "source": source,
        "reuse": {"path": str(ledger), "sha256": ledger_hash},
        "duplicate_seed": {"path": str(seed), "sha256": audit.Evidence().track(seed)},
        "batches": [
            {
                "id": "batch",
                "phase": 1,
                "directory": str(batch_root),
                "node": "himem01",
                "suite_sha256": audit.identity(original),
            }
        ],
    }
    (tmp_path / "campaign.json").write_text(json.dumps(campaign))
    monkeypatch.setattr(audit, "provenance", lambda: source)
    monkeypatch.setattr(audit, "scheduler", lambda *args: {})
    result = audit.audit(tmp_path, 1)
    assert result["status"] == "PASS" and result["batches"][0]["cells"] == 0
    assert len(result["batches"][0]["exclusions"][0]["instances"]) == 4
    (batch_root / "excluded.json").unlink()
    with pytest.raises(FileNotFoundError):
        audit.audit(tmp_path, 1)


@pytest.mark.parametrize("native", [False, True])
def test_complete_batch_full_audit_rejects_consistent_score_and_grid_tampering(
    tmp_path, monkeypatch, native
):
    """Use real snapshot/grid readers and compact journals, without a scheduler or ML fitting."""

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    batch_root = tmp_path / "batch"
    batch_root.mkdir()
    original, source = qualification_fixture(batch_root)
    source["source_sha256"] = "fixture-source"
    n = 32 if native else 11
    kind = "games" if native else "families"
    recipe = {"id": "cluster", "family": "knn" if native else "cluster", "n_players": n}
    if native:
        recipe.update(index="SV", order=1)
    original.update(
        families=[] if native else [recipe],
        games=[recipe] if native else [],
        methods=["KernelSHAP"],
        seeds=[0],
        relative_budgets=[0.5, 1, 2, 4, 8, 16, 32, 64, 128],
        targets=[{"index": "SV", "order": 1}],
        min_players=11,
        min_signal_ratio=1e-6,
    )
    instances = []
    for path in (batch_root / "qualification").glob("*.json"):
        path.unlink()
    for seed in range(4):
        key = {"spec": recipe, "kind": kind, "seed": seed, "source": source}
        result = {"status": "measured", "projected_seconds": 2.0, "n_players": n}
        if native:
            result.update(
                index="SV",
                order=1,
                actual_preparation_seconds=2.0,
                artifact_bytes=100,
                metadata_bytes=100,
            )
        write(
            batch_root / "qualification" / f"{audit.identity(key)}.json",
            {"identity": key, "result": result},
        )
        instances.append({"seed": seed, **result})
    qualified = {
        **original,
        "preparation_exclusions": [],
        "preparation_preflight": {
            "source": source,
            "requested_suite_sha256": audit.identity(original),
            "maximum_seconds_per_instance": 28800,
            "families": [
                {"id": "cluster", "kind": kind, "instances": instances, "status": "qualified"}
            ],
        },
    }
    write(batch_root / "suite.json", original)
    write(batch_root / "qualified-suite.json", qualified)
    write(
        batch_root / "qualification-decision.json",
        {
            "source": source,
            "requested_suite_sha256": audit.identity(original),
            "qualified_suite_sha256": audit.identity(qualified),
        },
    )
    games = []
    prepared = batch_root / "prepared"
    prepared.mkdir()
    for seed in range(4):
        amplitude = seed + 1.0
        artifact = f"g-{seed}.npz"
        metadata = {"instance_seed": seed, "score_eligible": True}
        if native:
            np.savez(prepared / artifact, x=np.ones((n, 3)))
            metadata.update(
                payoff_range_upper_bound=amplitude,
                signal_ratio=2 / np.sqrt(n),
                small_validation_players=8,
                small_validation_max_error=0.0,
            )
        else:
            values = np.array([amplitude * (mask & 1) for mask in range(2**n)])
            np.savez(
                prepared / artifact, values=values, evaluation_seconds=np.full(len(values), 0.001)
            )
            metadata.update(
                payoff_std=amplitude / 2,
                signal_ratio=2 / np.sqrt(n),
                game_quality=audit.payoff_diagnostics(values, n),
                fourier_spectrum=audit.fourier_spectrum(values, n),
            )
        games.append(
            {
                "id": f"cluster-i{seed}" + ("" if native else "-sv-1"),
                "family": "fixture",
                "stratum": "fixture",
                "oracle": "knn" if native else "table",
                "n_players": n,
                "index": "SV",
                "order": 1,
                "artifact": artifact,
                "metadata": metadata,
                "truth": {
                    "coordinates": [[0]],
                    "values": [amplitude],
                    "energy": amplitude**2,
                    "baseline": 0.0,
                },
            }
        )
    snapshot = {
        "schema_version": 1,
        "provenance": source,
        "suite": _prepared_suite(qualified, games),
        "games": games,
        "artifacts": {
            g["artifact"]: audit.Evidence().track(prepared / g["artifact"]) for g in games
        },
    }
    snapshot["snapshot_id"] = audit.identity(snapshot)
    snapshot_path = prepared / "snapshot.json"
    write(snapshot_path, snapshot)
    cpus = list(range(16))
    write(
        batch_root / "sweep/allocation.json",
        {
            "snapshot_id": snapshot["snapshot_id"],
            "game_ids": [g["id"] for g in games],
            "cpus": cpus,
        },
    )
    task = {"task": "123_0", "worker_job_id": "124", "node": "himem01", "cpus": 16}
    for step in ("prepare", "evaluate"):
        write(
            batch_root / f"{step}-resource-policy.json",
            {"workers": 16, "timing_profile": "diagnostic", "allocation": "shared-cpu"},
        )
        write(
            batch_root / f"{step}-allocation.json",
            {"hostname": "himem01", "cpu_model": "AMD EPYC 9754", "affinity": cpus},
        )
    rows = []
    for slot, game in enumerate(games):
        result = {
            "schema_version": 1,
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_provenance": source,
            "suite": snapshot["suite"],
            "games": games,
            "coverage": [],
            "methods": {
                "KernelSHAP": {
                    "source_sha256": source["source_sha256"],
                    "software_sha256": audit.identity(source),
                    "private": False,
                }
            },
            "run_provenance": {
                **source,
                "execution": {
                    "threads": 1,
                    "timeout": 600,
                    "memory_gb": 12,
                    "timing_profile": "diagnostic",
                    "game_ids": [game["id"]],
                    "hardware": {
                        "hostname": "himem01",
                        "cpu_model": "AMD EPYC 9754",
                        "affinity": [slot],
                    },
                },
            },
            "records": [],
        }
        estimate = InteractionValues(
            values=np.array(game["truth"]["values"]),
            index="SV",
            min_order=1,
            max_order=1,
            n_players=n,
            interaction_lookup={(0,): 0},
            baseline_value=0,
        )
        for budget in snapshot["suite"]["budgets_by_game"][game["id"]]:
            row = {
                **score(estimate, game),
                "estimate": game["truth"],
                "game_id": game["id"],
                "method": "KernelSHAP",
                "budget": budget,
                "seed": 0,
                "status": "ok",
                "error": None,
                "queries": 2,
                "requested_queries": 2,
                "seconds": 0.2,
                "wall_seconds": 0.3,
                "timing_profile": "diagnostic",
                "official_timing": False,
                "timing_scope": f"estimator_with_{game['oracle']}_oracle",
                "worker": {
                    "hostname": "himem01",
                    "cpu_model": "AMD EPYC 9754",
                    "affinity": [slot],
                    "slurm_job_id": "124",
                    "thread_pools": [{"num_threads": 1}],
                    "thread_environment": dict.fromkeys(audit.THREAD_VARIABLES, "1"),
                },
            }
            if not native:
                row.update(
                    cache_lookup_seconds=0.01,
                    estimated_oracle_seconds=0.002,
                    estimated_uncached_seconds=0.192,
                )
            result["records"].append(row)
        result["resume_key"] = audit.identity({k: v for k, v in result.items() if k != "records"})
        result["campaign"] = {"planned": 9, "completed": 9, "complete": True}
        path = batch_root / f"sweep/shard-{slot:03}/results.json"
        path.parent.mkdir(parents=True)
        (path.parent / ".campaign.lock").touch()
        Checkpoint(path.parent, snapshot_path, result)
        rows.append((path, result))
    ledger = tmp_path / "reuse.json"
    write(ledger, {"status": "PASS", "authenticated_inputs": {}, "canonical_snapshots": []})
    ledger_hash = audit.Evidence().track(ledger)
    seed = tmp_path / "seed.json"
    write(seed, {"reuse_sha256": ledger_hash, "canonical_entries": {}})
    campaign = {
        "source": source,
        "reuse": {"path": str(ledger), "sha256": ledger_hash},
        "duplicate_seed": {"path": str(seed), "sha256": audit.Evidence().track(seed)},
        "batches": [
            {
                "id": "batch",
                "phase": 1,
                "directory": str(batch_root),
                "node": "himem01",
                "suite_sha256": audit.identity(original),
            }
        ],
    }
    write(tmp_path / "campaign.json", campaign)
    monkeypatch.setattr(audit, "provenance", lambda: source)
    monkeypatch.setattr(
        audit, "scheduler", lambda *args: {("batch", s): task for s in ("prepare", "evaluate")}
    )
    monkeypatch.setattr(audit, "terminal", lambda *args: task)
    receipt = audit.audit(tmp_path, 1)
    assert receipt["batches"][0]["cells"] == 36 and len(receipt["canonical_games"]) == 4
    path, result = rows[0]
    # Switch to a valid legacy checkpoint so corruption passes all JSON/storage checks.
    result["records"][0]["nmse"] = 0.5
    write(path, result)
    with pytest.raises(ValueError, match="nMSE differs"):
        audit.audit(tmp_path, 1)
    result["records"][0]["nmse"] = 0.0
    result["records"].pop()
    result["campaign"] = {"planned": 8, "completed": 8, "complete": True}
    write(path, result)
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        audit.audit(tmp_path, 1)


def test_native_leaf_threshold_nan_is_unused_but_active_nan_is_rejected(tmp_path):
    artifact = tmp_path / "tree.npz"
    game = {
        "id": "tree",
        "n_players": 32,
        "index": "SV",
        "order": 1,
        "oracle": "pathdependent_tree",
        "artifact": "tree.npz",
        "truth": {"coordinates": [[0]], "values": [1.0], "energy": 1.0, "baseline": 0.0},
        "metadata": {
            "small_validation_players": 8,
            "small_validation_max_error": 0.0,
            "payoff_range_upper_bound": 1.0,
            "signal_ratio": 2 / np.sqrt(32),
            "score_eligible": True,
        },
    }
    for children, accepted in (([-1], True), ([1], False)):
        np.savez(
            artifact, tree_0_thresholds=np.array([np.nan]), tree_0_children_left=np.array(children)
        )
        snapshot = {
            "snapshot_id": "fixture",
            "suite": {"min_signal_ratio": 1e-6},
            "games": [game],
            "artifacts": {"tree.npz": audit.Evidence().track(artifact)},
        }
        if accepted:
            audit.inspect_games(snapshot, tmp_path, audit.Evidence())
        else:
            with pytest.raises(ValueError, match="active reconstruction"):
                audit.inspect_games(snapshot, tmp_path, audit.Evidence())
