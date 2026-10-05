"""Audit one completed expansion wave before allowing the next wave to start.

This is a numerical/provenance gate, not publication approval. Unknown failures
stop the pipeline; independent review is still required before publishing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from shapiq_benchmark.campaign import _batch_results, _game_ids, _prepared_suite, _qualification
from shapiq_benchmark.duplicates import payoff_fingerprint
from shapiq_benchmark.execution import THREAD_VARIABLES
from shapiq_benchmark.quality import clustering_diagnostics, payoff_diagnostics
from shapiq_benchmark.report import budget_failure
from shapiq_benchmark.results_io import read_results, result_inputs
from shapiq_benchmark.runner import identity, load_snapshot, method_catalog, provenance
from shapiq_benchmark.spectrum import fourier_spectrum


def require(condition: bool, message: str) -> None:  # noqa: FBT001 -- assertion helper
    """Fail closed without depending on Python assertion settings."""
    if not condition:
        raise ValueError(message)


class Evidence:
    """Hash each input once and recheck every byte before sealing the receipt."""

    def __init__(self) -> None:
        """Start an empty input closure."""
        self.files: dict[str, str] = {}

    def track(self, path: Path, expected: str | None = None) -> str:
        """Register exact file bytes once."""
        path = path.resolve()
        if str(path) not in self.files:
            with path.open("rb") as stream:
                self.files[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        actual = self.files[str(path)]
        require(expected is None or actual == expected, f"Changed input: {path}")
        return actual

    def read(self, path: Path) -> dict:
        """Read lists, JSON documents, or authenticated compact checkpoints."""
        self.track(path)
        raw = json.loads(path.read_text())
        if isinstance(raw, dict) and raw.get("storage_format") == "snapshot-journal-v1":
            for companion in result_inputs(path):
                self.track(companion)
            return read_results(path)
        return raw

    def stable(self) -> None:
        """Reject any input modified while this audit was running."""
        for name, expected in self.files.items():
            with Path(name).open("rb") as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == expected,
                    f"Input changed during audit: {name}",
                )


def terminal(job: str, node: str, cpus: int = 16) -> dict:
    """Use exact array task IDs, not array parents or absence from the queue."""
    require(re.fullmatch(r"\d+_\d+", job) is not None, "An exact scheduler array task is required")
    queue = subprocess.run(  # noqa: S603 -- validated scheduler identifier
        ["squeue", "-j", job, "-h", "-o", "%i|%T"],  # noqa: S607 -- scheduler on PATH
        capture_output=True,
        text=True,
        check=False,
    )
    require(
        not queue.stdout.strip()
        and (queue.returncode == 0 or "Invalid job id specified" in queue.stderr),
        f"Task is active or queue query failed: {job}",
    )
    text = subprocess.check_output(  # noqa: S603 -- fixed scheduler command
        [  # noqa: S607 -- scheduler on PATH
            "sacct",
            "-j",
            job,
            "--format=JobID,JobIDRaw,State,ExitCode,AllocCPUS,NodeList",
            "-Pn",
        ],
        text=True,
    )
    rows = [line.split("|") for line in text.splitlines() if line.split("|")[0] == job]
    require(
        len(rows) == 1 and rows[0][2:6] == ["COMPLETED", "0:0", str(cpus), node],
        f"Task is not successfully complete on its declared allocation: {job}",
    )
    return {"task": job, "worker_job_id": rows[0][1], "node": node, "cpus": cpus}


def close(actual: float | None, expected: float | None, label: str) -> None:
    """Scores retain null for undefined references, never replace them with zero."""
    if expected is None:
        require(actual is None, label)
    else:
        require(
            isinstance(actual, int | float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=1e-28),
            label,
        )


def coefficients(encoded: dict, n: int, order: int) -> dict:
    """Validate sparse coordinates independently of the production scorer."""
    keys, values = encoded["coordinates"], encoded["values"]
    require(len(keys) == len(values), "Coordinate/value length mismatch")
    result = {}
    for coordinate, value in zip(keys, values, strict=True):
        key = tuple(coordinate)
        require(
            key not in result
            and len(key) <= order
            and key == tuple(sorted(set(key)))
            and all(type(i) is int and 0 <= i < n for i in key)
            and isinstance(value, int | float)
            and math.isfinite(value),
            "Invalid or repeated coefficient",
        )
        result[key] = float(value)
    return result


def rescore(row: dict, game: dict) -> None:
    """Check every full-target and per-order score using stored coefficient vectors."""
    n, order = game["n_players"], game["order"]
    truth = coefficients(game["truth"], n, order)
    prediction = coefficients(row["estimate"], n, order)
    coordinates = (truth.keys() | prediction.keys()) - {()}
    energy = math.fsum(v * v for k, v in truth.items() if k)
    error = math.fsum((prediction.get(k, 0.0) - truth.get(k, 0.0)) ** 2 for k in coordinates)
    require(math.isfinite(error) and math.isfinite(energy), "Nonfinite score arithmetic")
    count = sum(math.comb(n, k) for k in range(1, order + 1))
    close(row["mse"], error / count, "Full-target MSE differs")
    close(row["nmse"], error / energy if energy else None, "Full-target nMSE differs")
    close(row["truth_energy"], energy, "Truth energy differs")
    require(
        row["normalization"] == "nonempty_l2_energy" and row["zero_truth_energy"] == (energy == 0),
        "Normalization metadata differs",
    )
    std = game["metadata"].get("payoff_std")
    scale = std if std is not None else math.sqrt(energy / count)
    require(math.isfinite(scale) and scale >= 0, "Invalid order reference scale")
    require(
        set(row["order_scores"]) == {str(k) for k in range(1, order + 1)}, "Missing degree scores"
    )
    for degree in range(1, order + 1):
        score = row["order_scores"][str(degree)]
        degree_energy = math.fsum(v * v for k, v in truth.items() if len(k) == degree)
        squared = math.fsum(
            (prediction.get(k, 0.0) - truth.get(k, 0.0)) ** 2
            for k in coordinates
            if len(k) == degree
        )
        dimension = math.comb(n, degree)
        ratio = math.sqrt(degree_energy / dimension) / scale if scale else 0.0
        eligible = degree_energy > 0 and ratio >= 1e-6
        require(
            score["score_eligible"] == eligible
            and score["signal_reference"]
            == ("payoff_std" if std is not None else "full_target_rms"),
            "Order eligibility differs",
        )
        for key, value in {
            "mse": squared / dimension,
            "nmse": squared / degree_energy if eligible else None,
            "truth_energy": degree_energy,
            "energy_share": degree_energy / energy if energy else None,
            "signal_ratio": ratio,
        }.items():
            close(score[key], value, f"Order {degree} {key} differs")


def known_failure(row: dict, execution: dict) -> str:
    """Accept recorded resource/budget limits; unexpected errors require investigation."""
    error = row.get("error") or ""
    if error == "TimeoutError: worker wall-time limit exceeded.":
        require(
            row.get("seconds") is None
            and row.get("queries") is None
            and row.get("requested_queries") is None
            and row["wall_seconds"] >= execution["timeout"],
            "Timeout lacks its recorded resource limit",
        )
        return "timeout"
    if error.startswith(("MemoryError:", "_ArrayMemoryError:")):
        require(execution["memory_gb"] > 0, "Memory failure lacks a declared cap")
        return "memory_limit"
    if budget_failure(row):
        return "under_budget"
    if error.startswith("BudgetExceededError:"):
        require(
            row["requested_queries"] > row["budget"] and row["queries"] <= row["budget"],
            "Budget violation lacks counted requests",
        )
        return "query_limit"
    message = (
        f"Unknown failure requires review: {row['game_id']} / {row['method']} / {row['budget']}"
    )
    raise ValueError(message)


def check_worker(worker: dict, node: str, cpu: int, job_id: str) -> None:
    """Require the actual standardized worker and single-thread runtime, not its label."""
    require(
        worker["hostname"].split(".")[0] == node
        and "EPYC 9754" in worker["cpu_model"]
        and worker["affinity"] == [cpu]
        and str(worker["slurm_job_id"]) == job_id,
        "Worker placement differs from the completed task",
    )
    require(
        worker["thread_pools"]
        and all(p["num_threads"] == 1 for p in worker["thread_pools"])
        and worker["thread_environment"] == dict.fromkeys(THREAD_VARIABLES, "1"),
        "Worker thread limits differ",
    )


@lru_cache(maxsize=1)
def capabilities() -> dict:
    """Reuse the frozen capability catalog for all cells."""
    return method_catalog()


def check_row(row: dict, game: dict, execution: dict, task: dict, cpu: int) -> str:
    """Audit a terminal cell without silently discarding failures or zero-energy games."""
    require(
        row["timing_profile"] == "diagnostic"
        and row["official_timing"] is False
        and row["timing_scope"] == f"estimator_with_{game.get('oracle', 'table')}_oracle",
        "Timing protocol differs",
    )
    status = row["status"]
    supported = game["index"] in capabilities()[row["method"]]["indices"]
    if status in {"unsupported", "duplicate"}:
        require(status == "duplicate" or not supported, "Supported estimator marked unsupported")
        require(
            row.get("queries") == row.get("requested_queries") == 0
            and row.get("seconds") is None
            and row.get("wall_seconds") == 0
            and not row.get("worker")
            and not row.get("estimate")
            and row.get("nmse") is None
            and row.get("mse") is None
            and not row.get("order_scores"),
            "Skipped cell pretends to have measurements",
        )
        return status
    require(supported and status in {"ok", "failed"}, "Unknown or incompatible cell status")
    if row.get("worker"):
        check_worker(row["worker"], task["node"], cpu, task["worker_job_id"])
    for field in ("queries", "requested_queries"):
        value = row.get(field)
        require(value is None or (type(value) is int and value >= 0), "Invalid query count")
    if row.get("queries") is not None:
        require(
            row["queries"] <= row["budget"] and row["requested_queries"] >= row["queries"],
            "Actual query count exceeds cap or requests",
        )
    for field in ("seconds", "wall_seconds"):
        value = row.get(field)
        require(value is None or (math.isfinite(value) and value >= 0), "Invalid measured time")
    if status == "failed":
        require(
            row.get("nmse") is None
            and row.get("mse") is None
            and not row.get("estimate")
            and not row.get("order_scores"),
            "Failed cell retains a successful estimate",
        )
        return known_failure(row, execution)
    require(
        row.get("worker")
        and type(row.get("queries")) is int
        and row["requested_queries"] == row["queries"]
        and row.get("seconds") is not None
        and row.get("wall_seconds") is not None
        and not row.get("error"),
        "Incomplete successful measurement",
    )
    rescore(row, game)
    fields = {"cache_lookup_seconds", "estimated_oracle_seconds", "estimated_uncached_seconds"}
    if game.get("oracle", "table") == "table":
        require(fields <= row.keys(), "Cached table lacks oracle-cost accounting")
        for field in fields:
            require(math.isfinite(row[field]) and row[field] >= 0, "Invalid cached time")
        close(
            row["estimated_uncached_seconds"],
            max(0.0, row["seconds"] - row["cache_lookup_seconds"])
            + row["estimated_oracle_seconds"],
            "Cached time arithmetic differs",
        )
    else:
        require(not fields.intersection(row), "Native oracle incorrectly claims cached timing")
    return "ok"


def qualification(directory: Path, original: dict, source: dict, evidence: Evidence) -> dict:
    """Bind all four measured construction seeds to the unchanged recipe partition."""
    qualified = evidence.read(directory / "qualified-suite.json")
    _qualification(original, qualified)
    require(
        evidence.read(directory / "qualification-decision.json")
        == {
            "requested_suite_sha256": identity(original),
            "qualified_suite_sha256": identity(qualified),
            "source": source,
        },
        "Qualification decision changed",
    )
    preflight = qualified["preparation_preflight"]
    require(
        preflight["source"] == source
        and preflight["requested_suite_sha256"] == identity(original)
        and original["game_seeds"] == [0, 1, 2, 3],
        "Qualification source or seed grid differs",
    )
    pilots = {}
    for path in (directory / "qualification").glob("*.json"):
        if path.name == "qualified-suite.json":
            continue
        cached = evidence.read(path)
        if "identity" in cached and "source" in cached["identity"]:
            key = cached["identity"]
            require(
                key["source"] == source and path.stem == identity(key), "Pilot identity changed"
            )
            pair = (key["spec"]["id"], key["seed"])
            require(pair not in pilots, "Repeated construction pilot")
            pilots[pair] = cached
    requested = {
        s["id"]: (kind, s) for kind in ("families", "games") for s in original.get(kind, [])
    }
    summaries = preflight["families"]
    require(
        len(summaries) == len(requested) and {s["id"] for s in summaries} == requested.keys(),
        "Preflight omits requested recipes",
    )
    exclusions = {s["spec"]["id"]: s for s in qualified["preparation_exclusions"]}
    for summary in summaries:
        kind, spec = requested[summary["id"]]
        instances = summary["instances"]
        require(
            summary["kind"] == kind and [v["seed"] for v in instances] == original["game_seeds"],
            "Preflight omits or repeats construction seeds",
        )
        failed, costly = False, False
        for instance in instances:
            cached = pilots[(spec["id"], instance["seed"])]
            require(
                cached["identity"]["spec"] == spec
                and cached["identity"]["kind"] == kind
                and cached["result"] == {k: v for k, v in instance.items() if k != "seed"},
                "Seed qualification does not match its frozen pilot",
            )
            if instance["status"] == "measured":
                seconds = instance["projected_seconds"]
                require(
                    math.isfinite(seconds)
                    and seconds >= 0
                    and instance["n_players"] == spec["n_players"],
                    "Preparation cost or requested dimensionality differs",
                )
                costly |= seconds > preflight["maximum_seconds_per_instance"]
                if kind == "games":
                    require(
                        instance["index"] == spec["index"]
                        and instance["order"] == spec["order"]
                        and instance["actual_preparation_seconds"] == seconds
                        and instance["artifact_bytes"] > 0
                        and instance["metadata_bytes"] > 0,
                        "Native solver lacks requested-size qualification",
                    )
            else:
                failed = True
                reason = instance["reason"]
                details = instance.get("details", {})
                if reason == "pilot_timeout":
                    require(
                        instance["limit_seconds"]
                        in {
                            preflight["pilot_timeout_seconds"],
                            preflight["structured_timeout_seconds"],
                        },
                        "Unbound preparation timeout",
                    )
                elif reason == "model_not_better_than_validation_dummy":
                    require(
                        details["passed"] is False
                        and (
                            not math.isfinite(details["model_loss"])
                            or details["model_loss"] >= details["dummy_loss"]
                        ),
                        "Invalid predictor exclusion",
                    )
                elif reason == "unstable_imputation":
                    limit = details["maximum_noise_ratio"]
                    ratios = [v["noise_ratio"] for v in details["levels"]] + [
                        details["sample_size_drift_ratio"]
                    ]
                    require(
                        limit == 0.1
                        and details["status"] != "stable"
                        and any(v is None or v > limit for v in ratios),
                        "Invalid stochastic exclusion",
                    )
                elif reason == "insufficient_eligible_clustering_features":
                    eligible = details["eligible_feature_indices"]
                    require(
                        details["requested_players"] == spec["n_players"]
                        and details["available_features"]
                        == len(set(eligible))
                        == len(eligible)
                        < spec["n_players"]
                        and all(type(i) is int and i >= 0 for i in eligible)
                        and len(set(details["training_rows"])) == len(details["training_rows"]) > 0,
                        "Invalid clustering feature exclusion",
                    )
                elif instance.get("error_type") in {"MemoryError", "_ArrayMemoryError"}:
                    require(preflight["structured_memory_gb"] > 0, "Unbound memory exclusion")
                elif (
                    reason == "construction_or_payoff_validation_failed"
                    and instance.get("error_type") == "ValueError"
                ):
                    log = (
                        directory / "qualification/private" / f"{identity(cached['identity'])}.log"
                    )
                    evidence.track(log)
                    last = log.read_text().strip().splitlines()[-1]
                    require(
                        kind == "games"
                        and last
                        in {
                            "ValueError: KNN n_players must be between 8 and the training split size.",
                            "ValueError: Product-kernel classification requires exactly two classes.",
                        },
                        f"Unknown native constructor exclusion requires review: {spec['id']}",
                    )
                else:
                    message = (
                        f"Unknown preparation exclusion requires review: {spec['id']} / {reason}"
                    )
                    raise ValueError(message)
        excluded = failed or costly
        require(
            summary["status"] == ("excluded" if excluded else "qualified")
            and (spec["id"] in exclusions) == excluded,
            "Qualification decision differs from measured limits",
        )
        if excluded:
            require(
                exclusions[spec["id"]]["instances"] == instances
                and exclusions[spec["id"]]["reason"]
                == ("preflight_failed" if failed else "projected_cost_limit"),
                "Exclusion evidence differs",
            )
    return qualified


def inspect_games(snapshot: dict, root: Path, evidence: Evidence) -> dict:
    """Authenticate truth and native arrays; recompute table-only spectrum and quality."""
    facts, checked = {}, {}
    for artifact, expected in snapshot["artifacts"].items():
        evidence.track(root / artifact, expected)
    for game in snapshot["games"]:
        n, metadata = game["n_players"], game["metadata"]
        require(n >= 11, "Players below declared benchmark minimum")
        require(math.isfinite(game["truth"]["baseline"]), "Nonfinite oracle baseline")
        truth = coefficients(game["truth"], n, game["order"])
        energy = math.fsum(v * v for key, v in truth.items() if key)
        close(game["truth"]["energy"], energy, "Frozen truth energy differs")
        table = game.get("oracle", "table") == "table"
        if game["artifact"] not in checked:
            with np.load(root / game["artifact"], allow_pickle=False) as arrays:
                for name in arrays.files:
                    values = arrays[name]
                    if not np.issubdtype(values.dtype, np.number):
                        continue
                    finite = np.isfinite(values)
                    if not table and re.fullmatch(r"tree_\d+_thresholds", name):
                        leaves = arrays[name.removesuffix("thresholds") + "children_left"] == -1
                        finite |= np.isnan(values) & leaves
                    require(finite.all(), "Nonfinite active reconstruction array")
                if table:
                    values, costs = arrays["values"], arrays["evaluation_seconds"]
                    require(
                        n <= 20 and values.shape == costs.shape == (2**n,) and np.all(costs >= 0),
                        "Incomplete table or invalid oracle charges",
                    )
                    quality = payoff_diagnostics(values, n)
                    clustering_diagnostics(values, metadata, quality)
                    if metadata.get("synthetic"):
                        quality["control_reasons"].append("synthetic_control")
                    if (
                        metadata.get("stochastic_frozen")
                        and metadata.get("imputation_stability", {}).get("status") != "stable"
                    ):
                        quality["control_reasons"].append("unqualified_stochastic_payoffs")
                    if quality["control_reasons"]:
                        quality["role"] = "control"
                    checked[game["artifact"]] = (
                        float(np.std(values, dtype=np.longdouble)),
                        quality,
                        fourier_spectrum(values, n),
                    )
                else:
                    require(
                        "evaluation_seconds" not in arrays.files and arrays.files,
                        "Native artifact must retain reconstruction arrays",
                    )
                    checked[game["artifact"]] = None
        if table:
            std, quality, spectrum = checked[game["artifact"]]
            close(metadata["payoff_std"], std, "Payoff scale differs")
            require(
                metadata["game_quality"] == quality and metadata["fourier_spectrum"] == spectrum,
                "Frozen quality/Fourier diagnostics differ",
            )
            scale = std
        else:
            require(
                metadata["small_validation_players"] == 8
                and math.isfinite(metadata["small_validation_max_error"]),
                "Native reference qualification missing",
            )
            scale = metadata["payoff_range_upper_bound"] / 2
            require(math.isfinite(scale) and scale >= 0, "Invalid native payoff range")
        dimension = sum(math.comb(n, d) for d in range(1, game["order"] + 1))
        ratio = math.sqrt(energy / dimension) / scale if scale else 0.0
        close(metadata["signal_ratio"], ratio, "Whole-target signal ratio differs")
        eligible = ratio >= snapshot["suite"]["min_signal_ratio"] and energy > 0
        require(metadata["score_eligible"] == eligible, "Zero/near-zero truth eligibility differs")
        facts[game["id"]] = {
            "fingerprint": payoff_fingerprint(game, root),
            "role": metadata.get("game_quality", {}).get("role", "unqualified"),
            "snapshot_id": snapshot["snapshot_id"],
            "score_eligible": eligible,
        }
    return facts


def scheduler(root: Path, phase: int, campaign: dict, evidence: Evidence) -> dict:
    """Bind immutable submitted commands and exact terminal task allocations."""
    journal = evidence.read(root / "audits" / f"phase-{phase}-jobs.json")
    require(journal["plan_sha256"] == identity(campaign), "Scheduler evidence has another campaign")
    plan_path = Path(journal["launch_plan_path"])
    evidence.track(plan_path, journal["launch_plan_sha256"])
    plan = evidence.read(plan_path)
    require(
        plan["status"] == "READY" and plan["campaign_identity"] == identity(campaign),
        "Launch plan differs",
    )
    for path, digest in plan["pins"].items():
        evidence.track(Path(path), digest)
    batches = [b for b in campaign["batches"] if b["phase"] == phase]
    expected = {
        f"wave-{phase}-{step}-{b['node']}" for b in batches for step in ("prepare", "evaluate")
    }
    require(set(journal["jobs"]) == expected, "Scheduler evidence has missing or extra wave jobs")
    tasks = {}
    for batch in batches:
        node = batch["node"]
        group = evidence.read(root / f"phase-{phase}-cpu-{node}.json")
        require(
            group == [b for b in batches if b["node"] == node], "Array group differs from campaign"
        )
        for step in ("prepare", "evaluate"):
            key = f"wave-{phase}-{step}-{node}"
            job = journal["jobs"][key]
            require(
                job["status"] == "SUBMITTED" and job["returncode"] == 0,
                "Submission was not acknowledged",
            )
            declared = next(j for j in plan["jobs"] if j["key"] == key)
            command = job["command"]
            require(
                declared["cpus"] == 16
                and declared["node"] == node
                and command[-5:]
                == [
                    plan["wrapper"],
                    str(plan_path),
                    journal["launch_plan_sha256"],
                    step,
                    declared["target"],
                ]
                and {
                    "--cpus-per-task=16",
                    "--mem=208G",
                    f"--nodelist={node}",
                    f"--array={declared['array']}",
                }
                <= set(command)
                and job["stdout"].strip().split(";")[0] == job["job_id"],
                "Submitted command differs from the pinned wave plan",
            )
            tasks[batch["id"], step] = terminal(f"{job['job_id']}_{group.index(batch)}", node)
    return tasks


def check_aliases(history: dict, games: dict, aliases: dict) -> None:
    """No repeated table/role may silently become another evaluated game."""
    known = {(g["fingerprint"], g["role"]) for g in history.values() if g["fingerprint"]}
    canonical = dict(history)
    for name, game in games.items():
        if name in aliases:
            continue
        key = (game["fingerprint"], game["role"])
        require(not game["fingerprint"] or key not in known, "Duplicate game was evaluated again")
        known.add(key)
        canonical[name] = game
    for alias, target in aliases.items():
        require(
            target in canonical
            and games[alias]["fingerprint"] is not None
            and games[alias]["fingerprint"] == canonical[target]["fingerprint"]
            and games[alias]["role"] == canonical[target]["role"],
            "Duplicate has no measured same-role canonical game",
        )


def audit(root: Path, phase: int) -> dict:
    """Audit one bounded wave, reusing strict campaign validation for each batch."""
    evidence = Evidence()
    evidence.track(Path(__file__))
    campaign = evidence.read(root / "campaign.json")
    require(
        campaign["source"] == provenance()
        and campaign["source"]["source_dirty"] is False
        and campaign["source"]["git_commit"],
        "Auditor is not using the clean frozen execution source",
    )
    require(any(b["phase"] == phase for b in campaign["batches"]), "Unknown or empty wave")
    tasks = scheduler(root, phase, campaign, evidence)
    prior = sorted({b["phase"] for b in campaign["batches"] if b["phase"] < phase})
    canonical, summaries = {}, []
    reuse = campaign["reuse"]
    evidence.track(Path(reuse["path"]), reuse["sha256"])
    ledger = evidence.read(Path(reuse["path"]))
    require(ledger["status"] == "PASS", "Historical reuse is not qualified")
    for path, sha in ledger["authenticated_inputs"].items():
        evidence.track(Path(path), sha)
    for entry in ledger["canonical_snapshots"]:
        path = Path(entry["path"])
        evidence.track(path, entry["sha256"])
        snapshot, artifacts = load_snapshot(path, historical=True)
        require(
            snapshot["snapshot_id"] == entry["snapshot_id"], "Historical canonical snapshot differs"
        )
        for game in snapshot["games"]:
            if game["id"] in entry["completed_game_ids"]:
                canonical[game["id"]] = {
                    "fingerprint": payoff_fingerprint(game, artifacts),
                    "role": entry["effective_roles"][game["id"]],
                    "snapshot_id": snapshot["snapshot_id"],
                }
    for previous in prior:
        receipt = evidence.read(root / "audits" / f"phase-{previous}.json")
        require(
            receipt["status"] == "PASS" and receipt["plan_sha256"] == identity(campaign),
            "Earlier wave is not verified",
        )
        for path, sha in receipt["authenticated_inputs"].items():
            evidence.track(Path(path), sha)
        canonical.update(receipt["canonical_games"])
    seed_spec = campaign["duplicate_seed"]
    evidence.track(Path(seed_spec["path"]), seed_spec["sha256"])
    seed = evidence.read(Path(seed_spec["path"]))
    require(seed["reuse_sha256"] == reuse["sha256"], "Historical registry seed differs")
    for key, claim in seed["canonical_entries"].items():
        game = canonical[claim["game_id"]]
        require(
            key == f"{game['fingerprint']}:{game['role']}"
            and claim["snapshot_id"] == game["snapshot_id"],
            "Historical registry claim differs",
        )
    aliases, all_facts = {}, {}
    for batch in (b for b in campaign["batches"] if b["phase"] == phase):
        directory = Path(batch["directory"])
        original = evidence.read(directory / "suite.json")
        require(identity(original) == batch["suite_sha256"], "Requested suite changed")
        qualified = qualification(directory, original, campaign["source"], evidence)
        for step in ("prepare", "evaluate"):
            require(
                evidence.read(directory / f"{step}-resource-policy.json")
                == {"workers": 16, "timing_profile": "diagnostic", "allocation": "shared-cpu"},
                "Resource override changed",
            )
            hardware = evidence.read(directory / f"{step}-allocation.json")
            require(
                hardware["hostname"].split(".")[0] == batch["node"]
                and "EPYC 9754" in hardware["cpu_model"]
                and len(set(hardware["affinity"])) == 16,
                "Controller allocation differs",
            )
        summary = {"batch_id": batch["id"], "exclusions": qualified["preparation_exclusions"]}
        if not qualified.get("families") and not qualified.get("games"):
            require(
                evidence.read(directory / "excluded.json")
                == {
                    "suite_sha256": identity(qualified),
                    "reason": "Every recipe failed the declared preparation gate",
                },
                "Fully excluded batch lacks evidence",
            )
            summaries.append({**summary, "games": 0, "cells": 0})
            continue
        snapshot_path = directory / "prepared/snapshot.json"
        evidence.track(snapshot_path)
        snapshot, artifacts = load_snapshot(snapshot_path)
        require(
            snapshot["provenance"] == campaign["source"]
            and snapshot["suite"] == _prepared_suite(qualified, snapshot["games"])
            and {g["id"] for g in snapshot["games"]} == _game_ids(qualified),
            "Prepared grid/source differs",
        )
        expected_games = {}
        for kind in ("families", "games"):
            for recipe in qualified.get(kind, []):
                targets = qualified["targets"] if kind == "families" else [recipe]
                for seed in qualified["game_seeds"]:
                    for target in targets:
                        (identifier,) = _game_ids(
                            {kind: [recipe], "game_seeds": [seed], "targets": [target]}
                        )
                        expected_games[identifier] = (
                            recipe["n_players"],
                            target["index"],
                            target["order"],
                            seed,
                        )
        require(
            all(
                (g["n_players"], g["index"], g["order"], g["metadata"]["instance_seed"])
                == expected_games[g["id"]]
                for g in snapshot["games"]
            ),
            "Prepared players, target, or construction seed differs from its recipe",
        )
        facts = inspect_games(snapshot, artifacts, evidence)
        require(not all_facts.keys() & facts.keys(), "Repeated game IDs across batches")
        all_facts.update(facts)
        by_id = {g["id"]: g for g in snapshot["games"]}
        allocation = evidence.read(directory / "sweep/allocation.json")
        require(
            len(allocation["cpus"]) == 16 and set(allocation["cpus"]) <= set(hardware["affinity"]),
            "Sweep CPU allocation differs",
        )
        counts, kinds, game_status = Counter(), Counter(), {}

        def read(
            path: Path,
            *,
            batch: dict = batch,
            allocation: dict = allocation,
            by_id: dict = by_id,
            kinds: Counter = kinds,
            counts: Counter = counts,
            game_status: dict = game_status,
        ) -> dict:
            result = evidence.read(path)
            if "records" not in result:
                return result
            slot = int(path.parent.name.split("-")[-1])
            execution = result["run_provenance"]["execution"]
            require(
                execution["threads"] == 1
                and execution["timing_profile"] == "diagnostic"
                and execution["timeout"] == 600
                and execution["memory_gb"] == 12,
                "Execution limits changed",
            )
            placed = execution["hardware"]
            require(
                placed["hostname"].split(".")[0] == batch["node"]
                and "EPYC 9754" in placed["cpu_model"]
                and placed["affinity"] == [allocation["cpus"][slot]],
                "Shard execution allocation differs",
            )
            for row in result["records"]:
                kind = check_row(
                    row,
                    by_id[row["game_id"]],
                    execution,
                    tasks[batch["id"], "evaluate"],
                    allocation["cpus"][slot],
                )
                kinds[kind] += 1
                counts[row["status"]] += 1
                skipped = row["status"] == "duplicate"
                require(
                    game_status.setdefault(row["game_id"], skipped) == skipped,
                    "Game mixes duplicate and measured cells",
                )
                if skipped:
                    require(
                        aliases.setdefault(row["game_id"], row["duplicate_of"])
                        == row["duplicate_of"],
                        "Conflicting duplicate claims",
                    )
            return result

        _batch_results(directory, snapshot, campaign["source"], read)
        summaries.append(
            {
                **summary,
                "snapshot_id": snapshot["snapshot_id"],
                "games": len(facts),
                "cells": sum(counts.values()),
                "statuses": dict(counts),
                "limitations": dict(kinds),
            }
        )
    check_aliases(canonical, all_facts, aliases)
    for task in tasks.values():
        require(
            terminal(task["task"], task["node"]) == task, "Scheduler evidence changed during audit"
        )
    require(provenance() == campaign["source"], "Execution source changed during audit")
    evidence.stable()
    return {
        "status": "PASS",
        "complete": True,
        "phase": phase,
        "plan_sha256": identity(campaign),
        "source": campaign["source"],
        "batches": summaries,
        "tasks": list(tasks.values()),
        "canonical_games": {k: v for k, v in all_facts.items() if k not in aliases},
        "aliases": aliases,
        "authenticated_inputs": evidence.files,
        "scope": "Frozen truth authenticated; scores independently recomputed. Table quality/Fourier recomputed. Native truth uses bounded preparation qualification, not independent full rederivation. Cached timing arithmetic is checked, not replayed per coalition. Human review required before publication.",
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    """Write one immutable receipt only after all gates pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--phase", type=int, required=True)
    args = parser.parse_args()
    output = args.campaign / "audits" / f"phase-{args.phase}.json"
    require(not output.exists(), "Audit receipt already exists; preserve its immutable evidence")
    receipt = audit(args.campaign.resolve(), args.phase)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
