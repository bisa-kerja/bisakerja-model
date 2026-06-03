from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.verify_phase_27_1_27_2_release_gate import build_report


ROOT = Path(__file__).resolve().parents[1]
PHASE25_MANIFEST = ROOT / "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json"
RELEASE_MANIFEST = ROOT / "artifacts/phase_25_tensorflow_training_delivery/tensorboard_release/manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase27ReleaseGateTest(unittest.TestCase):
    def test_tensorboard_release_manifest_records_non_ignored_release_event(self) -> None:
        release_manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(release_manifest["schema_version"], "phase-27-tensorboard-release-manifest-v1")
        self.assertEqual(release_manifest["requirement_reference"], "REQUIREMENT.md#1.4")
        self.assertFalse(release_manifest["release_git_ignore_check"]["release_event_file_ignored"])
        self.assertFalse(release_manifest["release_git_ignore_check"]["release_manifest_ignored"])
        self.assertEqual(release_manifest["event_file_count"], 1)

        event = release_manifest["event_files"][0]
        self.assertTrue(event["path"].startswith("artifacts/phase_25_tensorflow_training_delivery/tensorboard_release/"))
        event_path = ROOT / event["path"]
        self.assertTrue(event_path.exists())
        self.assertEqual(event_path.stat().st_size, event["size_bytes"])
        self.assertEqual(sha256_file(event_path), event["sha256"])

    def test_artifact_manifest_records_tensorboard_release_evidence(self) -> None:
        artifact_manifest = json.loads(PHASE25_MANIFEST.read_text(encoding="utf-8"))
        entries = {entry["artifact_id"]: entry for entry in artifact_manifest["artifacts"]}
        for artifact_id in ("phase25_tensorboard_release_manifest", "phase25_tensorboard_event_file"):
            self.assertIn(artifact_id, entries)
            entry = entries[artifact_id]
            path = ROOT / entry["path"]
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_size, entry["size_bytes"])
            self.assertEqual(sha256_file(path), entry["sha256"])
            self.assertFalse(entry["required_for_inference"])

    def test_release_gate_enforces_clean_baseline_before_production_ready(self) -> None:
        report = build_report()
        self.assertTrue(report["steps"]["27.1"]["implemented_gate"])
        self.assertIn(report["steps"]["27.1"]["status"], {"PASS", "FAIL"})
        self.assertEqual(report["steps"]["27.2"]["status"], "PASS")
        if report["git_state"]["dirty"]:
            self.assertEqual(report["final_decision"], "blocked")
        else:
            self.assertEqual(report["final_decision"], "production-ready")


if __name__ == "__main__":
    unittest.main()
