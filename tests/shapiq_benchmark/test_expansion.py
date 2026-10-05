"""Full expansion preserves science while omitting only authenticated completed recipes."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from shapiq_benchmark.protocol import build_expansion, build_phase
from shapiq_benchmark.runner import digest, identity

DIRECTORY = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("plan_expansion", DIRECTORY / "plan_expansion.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def test_full_inventory_keeps_quality_without_bounded_selection() -> None:
    base = json.loads((DIRECTORY / "suites/all-families.json").read_text())
    before = copy.deepcopy(base)
    suite = build_expansion(base)
    assert base == before
    assert len(suite["families"]) == len(build_phase(6, base)["families"]) == 10419
    assert len(suite["protocol"]["datasets"]) == 63
    assert suite["method_parameters"] == {"OddSHAP": {"ridge": 0.001}}
    assert suite["game_seeds"] == [0, 1, 2, 3]
    assert suite["relative_budgets"] == [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    assert [s["n_players"] for s in suite["families"]] == sorted(
        s["n_players"] for s in suite["families"]
    )
    assert all(
        s["quality_protocol"] == "quality-v2" and s.get("device", "cpu") == "cpu"
        for s in suite["families"]
    )
    assert "cluster" not in {s["family"] for s in suite["families"]}
    assert any(s["family"] == "cluster_continuous_v1" for s in suite["families"])
    native = build_expansion(base, structured=True)
    assert len(native["games"]) == 1246
    assert all(s["quality_protocol"] == "quality-v2" for s in native["games"])
    assert any(r.get("target_exclusions") for r in native["phase_plan"]["candidates"])


@pytest.fixture
def miniature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    base = json.loads((DIRECTORY / "suites/all-families.json").read_text())
    table = build_expansion(base)
    table["families"] = table["families"][:3]
    native = build_expansion(base, structured=True)
    native["games"] = native["games"][:1]
    monkeypatch.setattr(
        planner,
        "build_expansion",
        lambda _, structured=False: copy.deepcopy(native if structured else table),
    )
    monkeypatch.setattr(
        planner, "provenance", lambda: {"source_dirty": False, "git_commit": "frozen"}
    )
    monkeypatch.setattr(planner, "scripts", lambda: {"phase_batch.py": "reviewed"})
    evidence = tmp_path / "completed-audit.json"
    evidence.write_text('{"status":"PASS","complete":true}')
    entry = {
        "kind": "families",
        "spec": table["families"][0],
        "config": {key: table[key] for key in planner.CONFIG_FIELDS},
        "complete": True,
        "original_sources": [{"software_sha256": "historical"}],
        "snapshot_id": "old",
        "evidence": [str(evidence)],
    }
    entry["key"] = planner.recipe_key(entry["kind"], entry["spec"], entry["config"])
    ledger = {
        "version": 1,
        "status": "PASS",
        "entries": [entry],
        "authenticated_inputs": {str(evidence): digest(evidence)},
    }
    reuse = tmp_path / "reuse.json"
    reuse.write_text(json.dumps(ledger))
    return base, table, native, reuse, evidence


def test_complete_partition_and_immutable_resume(miniature: tuple, tmp_path: Path) -> None:
    base, table, native, reuse, _ = miniature
    root = tmp_path / "campaign"
    result = planner.plan(root, base, reuse_path=reuse, reuse_sha256=digest(reuse), batch_size=1)
    assert len(result["batches"]) == 3
    assert len(result["reuse"]["recipes"]) == 1
    assert [w["reference"] for w in result["waves"]] == ["enumeration", "structured"]
    planned = []
    for batch in result["batches"]:
        suite = json.loads((Path(batch["directory"]) / "suite.json").read_text())
        assert identity(suite) == batch["suite_sha256"]
        assert suite["method_parameters"] == {"OddSHAP": {"ridge": 0.001}}
        planned.extend(s["id"] for s in suite["families"] + suite["games"])
    assert set(planned) == {s["id"] for s in table["families"][1:] + native["games"]}
    assert (
        planner.plan(root, base, reuse_path=reuse, reuse_sha256=digest(reuse), batch_size=1)
        == result
    )
    with pytest.raises(ValueError, match="changed"):
        planner.plan(root, base, reuse_path=reuse, reuse_sha256=digest(reuse), batch_size=2)


def test_reuse_requires_pinned_complete_evidence(miniature: tuple) -> None:
    _, _, _, reuse, evidence = miniature
    with pytest.raises(ValueError, match="SHA256"):
        planner.read_reuse(reuse, "wrong")
    original = evidence.read_text()
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="inputs changed"):
        planner.read_reuse(reuse, digest(reuse))
    evidence.write_text(original)
    ledger = json.loads(reuse.read_text())
    ledger["entries"][0]["complete"] = False
    reuse.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="incomplete"):
        planner.read_reuse(reuse, digest(reuse))


def test_recipe_identity_preserves_every_semantic_setting(miniature: tuple) -> None:
    _, table, _, _, _ = miniature
    spec = table["families"][0]
    key = planner.recipe_key("families", spec, table)
    assert key == planner.recipe_key("families", {**spec, "id": "different-name"}, table)
    assert key != planner.recipe_key("families", {**spec, "n_players": 12}, table)
    assert key != planner.recipe_key("families", spec, {**table, "game_seeds": [0, 1]})
    assert key != planner.recipe_key("families", spec, {**table, "method_parameters": {}})


def test_registry_uses_published_role_and_authenticates_artifacts(tmp_path: Path) -> None:
    """An old table reclassified as a control must not suppress a new core game."""
    import numpy as np

    artifact = tmp_path / "payoffs.npz"
    np.savez(artifact, values=np.arange(4, dtype=float))
    game = {
        "id": "old",
        "oracle": "table",
        "n_players": 2,
        "index": "SV",
        "order": 1,
        "artifact": artifact.name,
        "metadata": {"game_quality": {"role": "core"}},
    }
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"snapshot_id": "original", "games": [game]}))
    ledger = {
        "canonical_snapshots": [
            {
                "path": str(snapshot),
                "sha256": digest(snapshot),
                "snapshot_id": "original",
                "completed_game_ids": ["old"],
                "effective_roles": {"old": "control"},
            }
        ],
        "authenticated_inputs": {str(snapshot): digest(snapshot), str(artifact): digest(artifact)},
    }
    root = tmp_path / "new"
    root.mkdir()
    receipt = planner.seed_registry(root, ledger, "reviewed-ledger")
    registry = json.loads((root / "duplicate-games.json").read_text())
    assert receipt["entries"] == 1
    assert next(iter(registry)).endswith(":control")
    assert (
        json.loads(snapshot.read_text())["games"][0]["metadata"]["game_quality"]["role"] == "core"
    )
    assert planner.seed_registry(root, ledger, "reviewed-ledger") == receipt
    del ledger["authenticated_inputs"][str(artifact)]
    with pytest.raises(ValueError, match="artifact is not authenticated"):
        planner.seed_registry(root, ledger, "reviewed-ledger")
