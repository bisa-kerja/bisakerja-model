#!/usr/bin/env python3
"""Verify TODO stabilization Step 2 TensorFlow runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / "training/.tf-venv-3.13"
VENV_PYTHON = VENV_DIR / "Scripts/python.exe"
REQUIREMENTS_PATH = ROOT / "training/requirements.txt"
REPORT_JSON_PATH = ROOT / "reports/training_step_2_runtime.json"
REPORT_MD_PATH = ROOT / "reports/training_step_2_runtime.md"

EXPECTED_PYTHON_MAJOR_MINOR = (3, 13)
EXPECTED_TENSORFLOW = "2.21.0"
EXPECTED_KERAS = "3.14.1"
EXPECTED_KERNEL_NAME = "bisakerja-model-tf-3.13"
EXPECTED_KERNEL_DISPLAY_NAME = "Bisakerja Model TF 3.13"
REQUIRED_IMPORTS = ("tensorflow", "keras", "pandas", "sentence_transformers", "pytest")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    return {
        "path": rel(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def status_from(blockers: list[str], warnings: list[str] | None = None) -> str:
    if blockers:
        return "BLOCKED"
    if warnings:
        return "WARN"
    return "PASS"


def run_command(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except OSError as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": type(exc).__name__,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": "TimeoutExpired",
        }
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def parse_json_from_stdout(result: dict[str, Any]) -> dict[str, Any]:
    stdout = str(result.get("stdout", "")).strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                return parsed
    for line in reversed(str(result.get("stdout", "")).splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_pyvenv_cfg(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def requirements_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    if not REQUIREMENTS_PATH.exists():
        return versions
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        name, version = stripped.split("==", 1)
        versions[name.strip().lower().replace("_", "-")] = version.strip()
    return versions


def audit_venv() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    pyvenv = parse_pyvenv_cfg(VENV_DIR / "pyvenv.cfg")

    if not VENV_DIR.exists():
        blockers.append("training/.tf-venv-3.13 does not exist.")
    if not VENV_PYTHON.exists():
        blockers.append("training/.tf-venv-3.13/Scripts/python.exe does not exist.")
    if pyvenv.get("version") and not pyvenv["version"].startswith("3.13."):
        blockers.append(f"pyvenv.cfg records Python {pyvenv['version']}, expected 3.13.x.")
    if pyvenv.get("executable") and "Python313" not in pyvenv["executable"]:
        warnings.append(f"pyvenv.cfg base executable is not clearly Python313: {pyvenv['executable']}.")

    version_probe = run_command(
        [
            str(VENV_PYTHON),
            "-c",
            (
                "import json, sys; "
                "print(json.dumps({'executable': sys.executable, "
                "'version': sys.version.split()[0], "
                "'major': sys.version_info.major, "
                "'minor': sys.version_info.minor, "
                "'micro': sys.version_info.micro}))"
            ),
        ],
        timeout=30,
    )
    version_payload = parse_json_from_stdout(version_probe)
    if version_probe.get("returncode") != 0:
        blockers.append(
            "training/.tf-venv-3.13/Scripts/python.exe could not execute; "
            f"{version_probe.get('error') or version_probe.get('stderr', '').strip() or 'unknown error'}."
        )
    elif (version_payload.get("major"), version_payload.get("minor")) != EXPECTED_PYTHON_MAJOR_MINOR:
        blockers.append(f"Venv Python is {version_payload.get('version')}, expected 3.13.x.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "venv_dir": file_record(VENV_DIR / "pyvenv.cfg"),
        "venv_python": file_record(VENV_PYTHON),
        "pyvenv_cfg": pyvenv,
        "version_probe": {
            "returncode": version_probe.get("returncode"),
            "stderr": str(version_probe.get("stderr", "")).strip(),
            "payload": version_payload,
        },
    }


def audit_requirements_and_imports() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    required_versions = requirements_versions()

    for package in ("tensorflow", "pandas", "sentence-transformers", "pytest"):
        if package not in required_versions:
            blockers.append(f"training/requirements.txt does not pin {package}.")
    if required_versions.get("tensorflow") != EXPECTED_TENSORFLOW:
        blockers.append(
            f"training/requirements.txt pins tensorflow=={required_versions.get('tensorflow')}, expected {EXPECTED_TENSORFLOW}."
        )

    import_probe = run_command(
        [
            str(VENV_PYTHON),
            "-c",
            (
                "import importlib.metadata as md, json; "
                "import tensorflow as tf; import keras; import pandas; import sentence_transformers; import pytest; "
                "print(json.dumps({"
                "'tensorflow': tf.__version__, "
                "'keras': keras.__version__, "
                "'pandas': md.version('pandas'), "
                "'sentence_transformers': md.version('sentence-transformers'), "
                "'pytest': md.version('pytest')"
                "}))"
            ),
        ],
        timeout=180,
    )
    import_payload = parse_json_from_stdout(import_probe)
    if import_probe.get("returncode") != 0:
        blockers.append(
            "Required imports could not be verified from training/.tf-venv-3.13; "
            f"{import_probe.get('error') or import_probe.get('stderr', '').strip() or 'unknown error'}."
        )
    else:
        missing = [name for name in REQUIRED_IMPORTS if name not in import_payload]
        if missing:
            blockers.append(f"Import probe did not report required modules: {missing}.")
        if import_payload.get("tensorflow") != EXPECTED_TENSORFLOW:
            blockers.append(f"TensorFlow is {import_payload.get('tensorflow')}, expected {EXPECTED_TENSORFLOW}.")
        if import_payload.get("keras") != EXPECTED_KERAS:
            blockers.append(f"Keras is {import_payload.get('keras')}, expected {EXPECTED_KERAS}.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "requirements": file_record(REQUIREMENTS_PATH),
        "required_versions": required_versions,
        "import_probe": {
            "returncode": import_probe.get("returncode"),
            "stderr": str(import_probe.get("stderr", "")).strip(),
            "payload": import_payload,
        },
    }


def audit_kernel() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    kernel_probe = run_command([str(VENV_PYTHON), "-m", "jupyter", "kernelspec", "list", "--json"], timeout=60)
    kernels_payload = parse_json_from_stdout(kernel_probe)
    kernels = kernels_payload.get("kernelspecs", {}) if isinstance(kernels_payload.get("kernelspecs"), dict) else {}
    expected_kernel = kernels.get(EXPECTED_KERNEL_NAME, {})
    expected_spec = expected_kernel.get("spec", {}) if isinstance(expected_kernel, dict) else {}
    argv = expected_spec.get("argv", []) if isinstance(expected_spec, dict) else []

    if kernel_probe.get("returncode") != 0:
        blockers.append(
            "Jupyter kernelspec list could not run from training/.tf-venv-3.13; "
            f"{kernel_probe.get('error') or kernel_probe.get('stderr', '').strip() or 'unknown error'}."
        )
    elif EXPECTED_KERNEL_NAME not in kernels:
        blockers.append(f"Kernel {EXPECTED_KERNEL_NAME} is not registered.")
    else:
        display_name = expected_spec.get("display_name")
        if display_name != EXPECTED_KERNEL_DISPLAY_NAME:
            blockers.append(f"Kernel display_name is {display_name!r}, expected {EXPECTED_KERNEL_DISPLAY_NAME!r}.")
        argv_text = " ".join(str(item) for item in argv)
        if ".tf-venv-3.13" not in argv_text.replace("\\", "/"):
            blockers.append("Kernel argv does not point to training/.tf-venv-3.13.")
        if "python" not in argv_text.lower():
            warnings.append("Kernel argv does not clearly include a Python executable.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "expected_kernel": EXPECTED_KERNEL_NAME,
        "kernel_probe": {
            "returncode": kernel_probe.get("returncode"),
            "stderr": str(kernel_probe.get("stderr", "")).strip(),
            "registered_kernel_names": sorted(kernels.keys()),
            "expected_kernel_spec": expected_spec,
        },
    }


def build_report() -> dict[str, Any]:
    venv = audit_venv()
    imports = audit_requirements_and_imports()
    kernel = audit_kernel()
    blockers: list[str] = []
    warnings: list[str] = []
    for section in (venv, imports, kernel):
        blockers.extend(section.get("blockers", []))
        warnings.extend(section.get("warnings", []))

    acceptance = {
        "venv_uses_python_3_13": venv["status"] != "BLOCKED",
        "pytest_available_in_venv": "pytest" in imports.get("import_probe", {}).get("payload", {}),
        "tensorflow_imports_in_venv": "tensorflow" in imports.get("import_probe", {}).get("payload", {}),
        "tensorflow_version_2_21_0": imports.get("import_probe", {}).get("payload", {}).get("tensorflow") == EXPECTED_TENSORFLOW,
        "keras_version_3_14_1": imports.get("import_probe", {}).get("payload", {}).get("keras") == EXPECTED_KERAS,
        "kernel_points_to_training_venv": kernel["status"] != "BLOCKED",
        "python_3_14_not_used_for_phase_25": (
            venv.get("version_probe", {}).get("payload", {}).get("major"),
            venv.get("version_probe", {}).get("payload", {}).get("minor"),
        )
        == EXPECTED_PYTHON_MAJOR_MINOR,
    }
    if blockers:
        final_decision = "blocked"
    elif warnings:
        final_decision = "implemented-with-warnings"
    else:
        final_decision = "pass"

    return {
        "schema_version": "training-step-2-runtime-v1",
        "phase_id": "training_step_2_runtime",
        "generated_at": now_iso(),
        "todo_source": "training/TODOS.md#step-2-stabilkan-runtime",
        "expected": {
            "python": "3.13.x",
            "tensorflow": EXPECTED_TENSORFLOW,
            "keras": EXPECTED_KERAS,
            "kernel_name": EXPECTED_KERNEL_NAME,
            "kernel_display_name": EXPECTED_KERNEL_DISPLAY_NAME,
        },
        "runtime": venv,
        "dependencies": imports,
        "kernel": kernel,
        "acceptance": acceptance,
        "blockers": blockers,
        "warnings": warnings,
        "final_decision": final_decision,
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Training Step 2 Runtime",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
        "## Runtime",
        "",
        f"Status: **{report['runtime']['status']}**",
        f"Python payload: `{report['runtime']['version_probe']['payload']}`",
        "",
        "## Dependencies",
        "",
        f"Status: **{report['dependencies']['status']}**",
        f"Import payload: `{report['dependencies']['import_probe']['payload']}`",
        "",
        "## Kernel",
        "",
        f"Status: **{report['kernel']['status']}**",
        f"Registered kernels: `{report['kernel']['kernel_probe']['registered_kernel_names']}`",
        "",
    ]
    if report["blockers"]:
        lines.append("## Blockers")
        lines.append("")
        lines.extend(f"- {item}" for item in report["blockers"])
        lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify training TODO stabilization Step 2 runtime.")
    parser.add_argument("--write", action="store_true", help="Write JSON and Markdown audit reports.")
    args = parser.parse_args()

    report = build_report()
    if args.write:
        write_json(REPORT_JSON_PATH, report)
        write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] in {"pass", "implemented-with-warnings"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
