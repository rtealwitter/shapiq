"""Scheduler-free checks of global resource bounds and durable submission intent."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "queue_expansion", Path(__file__).resolve().parents[2] / "benchmark/queue_expansion.py"
)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "benchmark").mkdir(parents=True)
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        script = self.source / "benchmark" / "phase_batch.py"
        script.write_text("# fixture\n")
        settings = {
            "status": "READY",
            "source": str(self.source),
            "workers": 16,
            "cpu_cap": 128,
            "nodes": ["himem01", "himem02"],
            "pins": {},
        }
        for name in (
            "warm_script",
            "audit_script",
            "run_script",
            "wrapper",
            "stage_script",
            "launcher",
        ):
            settings[name] = str(script)
        settings["pins"][str(script)] = launch.digest(script)
        self.settings = settings
        batches = []
        for phase in (1, 2):
            for node in settings["nodes"]:
                group = []
                for i in range(2):
                    directory = self.campaign / f"{phase}-{node}-{i}"
                    directory.mkdir()
                    suite = {"families": [{"id": directory.name}]}
                    (directory / "suite.json").write_text(json.dumps(suite))
                    group.append(
                        {
                            "id": directory.name,
                            "phase": phase,
                            "node": node,
                            "device": "cpu",
                            "directory": str(directory),
                            "suite_sha256": launch.identity(suite),
                        }
                    )
                batches += group
                (self.campaign / f"phase-{phase}-cpu-{node}.json").write_text(json.dumps(group))
        campaign = {
            "source": {"git_commit": "abc", "source_dirty": False},
            "scripts": {"phase_batch.py": launch.digest(script)},
            "batches": batches,
        }
        for field in ("reuse", "duplicate_seed"):
            path = self.campaign / (field + ".json")
            path.write_text("{}")
            campaign[field] = {"path": str(path), "sha256": launch.digest(path)}
        planner = self.source / "benchmark/plan_expansion.py"
        planner.write_text("# fixture\n")
        campaign["planner_sha256"] = launch.digest(planner)
        self.path = self.campaign / "campaign.json"
        self.path.write_text(json.dumps(campaign))

    def plan(self):
        return launch.make_plan(self.path, self.settings)

    def test_global_cap_and_stage_barriers(self):
        plan = self.plan()
        self.assertEqual(sum(plan["concurrency_slots"].values()) * 16, 128)
        jobs = {j["key"]: j for j in plan["jobs"]}
        for phase in (1, 2):
            prepares = [f"wave-{phase}-prepare-{n}" for n in self.settings["nodes"]]
            evaluates = [f"wave-{phase}-evaluate-{n}" for n in self.settings["nodes"]]
            for key in prepares:
                self.assertEqual(
                    jobs[key]["dependencies"], ["warm" if phase == 1 else "wave-1-audit"]
                )
                self.assertEqual(jobs[key]["array"], "0-1%4")
            for key in evaluates:
                self.assertEqual(jobs[key]["dependencies"], prepares)
            self.assertEqual(jobs[f"wave-{phase}-audit"]["dependencies"], evaluates)

    def test_invalid_workers_or_pool_rejected(self):
        for key, value in [("workers", 32), ("cpu_cap", 256), ("nodes", ["gpu01"])]:
            with self.subTest(key=key):
                settings = {**self.settings, key: value}
                with self.assertRaises(ValueError):
                    launch.make_plan(self.path, settings)

    def test_suite_tampering_rejected(self):
        path = self.campaign / "1-himem01-0/suite.json"
        path.write_text("{}")
        with self.assertRaises(ValueError):
            self.plan()

    def test_reuse_seed_and_planner_pins_cannot_be_omitted(self):
        plan = self.plan()
        for path in (
            self.campaign / "reuse.json",
            self.campaign / "duplicate_seed.json",
            self.source / "benchmark/plan_expansion.py",
        ):
            self.assertEqual(plan["pins"][str(path)], launch.digest(path))
        (self.campaign / "duplicate_seed.json").write_text("changed")
        with self.assertRaises(ValueError):
            self.plan()

    def test_script_tampering_rejected(self):
        plan = self.plan()
        Path(plan["wrapper"]).write_text("changed")
        with self.assertRaises(ValueError):
            launch.verify(plan)

    def test_missing_executable_pin_rejected(self):
        plan = self.plan()
        plan["warm_script"] = "/not-pinned.py"
        with self.assertRaises(ValueError):
            launch.verify(plan)

    def test_draft_plan_cannot_submit(self):
        plan = self.plan()
        plan["status"] = "DRAFT"
        with self.assertRaises(ValueError):
            launch.verify(plan)

    def save_plan(self):
        path = self.root / "plan.json"
        path.write_text(json.dumps(self.plan()))
        return path

    def git(self, args, **kwargs):
        return "abc\n" if args[1] == "rev-parse" else ""

    def test_success_is_idempotent_and_wave_receipt_immutable(self):
        plan = self.save_plan()
        count = 0

        def scheduler(*args, **kwargs):
            nonlocal count
            count += 1
            return type("Result", (), {"returncode": 0, "stdout": f"{count}\n", "stderr": ""})()

        with (
            patch.object(launch.subprocess, "check_output", side_effect=self.git),
            patch.object(launch.subprocess, "run", side_effect=scheduler),
        ):
            launch.submit(plan)
            first = count
            launch.submit(plan)
            self.assertEqual(count, first)
        receipt = json.loads((self.campaign / "audits/phase-1-jobs.json").read_text())
        self.assertEqual(len(receipt["jobs"]), 4)
        self.assertTrue(all(v["status"] == "SUBMITTED" for v in receipt["jobs"].values()))
        commands = [
            v["command"]
            for v in json.loads((self.campaign / "jobs.json").read_text())["jobs"].values()
        ]
        self.assertTrue(all("--exclusive" not in command for command in commands))

    def test_ambiguous_submission_never_retried(self):
        plan = self.save_plan()
        failure = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "lost response"})()
        with (
            patch.object(launch.subprocess, "check_output", side_effect=self.git),
            patch.object(launch.subprocess, "run", return_value=failure) as run,
        ):
            with self.assertRaises(RuntimeError):
                launch.submit(plan)
            with self.assertRaises(RuntimeError):
                launch.submit(plan)
            self.assertEqual(run.call_count, 1)

    def test_cross_host_atomic_submit_lock(self):
        plan = self.save_plan()
        (self.campaign / ".submit-lock").mkdir()
        with (
            patch.object(launch.subprocess, "check_output", side_effect=self.git),
            patch.object(launch.subprocess, "run") as run,
        ):
            with self.assertRaises(FileExistsError):
                launch.submit(plan)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
