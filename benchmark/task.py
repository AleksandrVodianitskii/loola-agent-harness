"""Inventory migration task and independent acceptance check."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


BATCH_COUNT = 32
MIGRATION_ID = "warehouse-cutover-r17"
SCHEMA_VERSION = 3
LOW_STOCK_BELOW = 7
PROMPT = f"""Implement a small inventory migration in this workspace.

Read migration_profile.json first and take the migration ID and schema version
from it. The inputs are raw/00.json through raw/{BATCH_COUNT - 1:02d}.json. Process
them in order, explicitly reading each file individually and writing its
normalized JSON before reading the next batch. Ignore the diagnostics field.

For each file, process only records with enabled set to true. Normalize each SKU
by stripping the outer whitespace, uppercasing it, and replacing each run of
whitespace with a hyphen. Group rows by normalized SKU and add their units and
reserved fields. Write normalized/NN.json as an object with batch (two-digit
string) and items (list sorted by SKU). Each item must have sku, units, reserved,
and available (units minus reserved).

Implement audit.py with a command-line entry point. With no arguments it should
read the sibling normalized directory. It must also accept an optional directory
as its first positional argument. It should fail with a nonzero exit status if
any of the {BATCH_COUNT} numbered files is missing. Print one JSON object with
migration_id, schema_version, batch_count, total_available, and low_stock (a
sorted list of unique SKUs with available strictly below {LOW_STOCK_BELOW}).
Compute the summary from the output files when the script runs.

Do not edit the raw files or use a bulk script to migrate the batches. When the
workspace artifacts are ready, report completion.
"""


def _sku_components(index: int) -> tuple[int, int, int, int]:
    return (index % 9 + 1, index % 3, index % 5 + 2, index % 4)


def prepare(workspace: Path) -> dict[str, bytes]:
    """Create a fresh, deterministic workspace without exposing the oracle."""
    (workspace / "raw").mkdir()
    (workspace / "normalized").mkdir()
    (workspace / "migration_profile.json").write_text(
        json.dumps({"migration_id": MIGRATION_ID, "schema_version": SCHEMA_VERSION}) + "\n",
        encoding="utf-8",
    )
    rng = random.Random(8407)
    words = "building channel crate dock label location operator pallet rack route shift sweep zone".split()
    for index in range(BATCH_COUNT):
        first_units, first_reserved, second_units, second_reserved = _sku_components(index)
        document = {
            "batch": f"{index:02d}",
            "records": [
                {"sku": f"  bin {index:02d} a", "units": first_units, "reserved": first_reserved, "enabled": True},
                {"sku": f"BIN  {index:02d} A ", "units": 4, "reserved": 2, "enabled": True},
                {"sku": f"Bin\t{index:02d}\tb", "units": second_units, "reserved": second_reserved, "enabled": True},
                {"sku": f"bin {index:02d} c", "units": 400, "reserved": 0, "enabled": False},
            ],
            "diagnostics": " ".join(rng.choice(words) for _ in range(285)),
        }
        (workspace / "raw" / f"{index:02d}.json").write_text(
            json.dumps(document) + "\n", encoding="utf-8"
        )
    return _raw_hashes(workspace)


def _raw_hashes(workspace: Path) -> dict[str, bytes]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).digest()
        for path in (workspace / "raw").glob("*.json")
        if path.is_file()
    }


def _expected_batch(index: int) -> dict:
    batch = f"{index:02d}"
    first_units, first_reserved, second_units, second_reserved = _sku_components(index)
    return {
        "batch": batch,
        "items": [
            {
                "sku": f"BIN-{batch}-A",
                "units": first_units + 4,
                "reserved": first_reserved + 2,
                "available": first_units - first_reserved + 2,
            },
            {
                "sku": f"BIN-{batch}-B",
                "units": second_units,
                "reserved": second_reserved,
                "available": second_units - second_reserved,
            },
        ],
    }


def _expected_audit(batches: list[dict]) -> dict:
    items = [item for batch in batches for item in batch["items"]]
    return {
        "migration_id": MIGRATION_ID,
        "schema_version": SCHEMA_VERSION,
        "batch_count": len(batches),
        "total_available": sum(item["available"] for item in items),
        "low_stock": sorted({item["sku"] for item in items if item["available"] < LOW_STOCK_BELOW}),
    }


def _run_audit(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(workspace / "audit.py"), *args],
        cwd=workspace,
        env={"PATH": os.defpath, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _audit_matches(result: subprocess.CompletedProcess[str], expected: dict) -> bool:
    try:
        return result.returncode == 0 and json.loads(result.stdout) == expected
    except ValueError:
        return False


def verify(workspace: Path, raw_hashes: dict[str, bytes]) -> dict:
    """Check actual files, default/alternate CLI input, and a missing batch."""
    expected_batches = [_expected_batch(index) for index in range(BATCH_COUNT)]
    path_by_name = {path.name: path for path in (workspace / "normalized").glob("*.json")}
    failures: list[str] = []
    correct_batches = 0
    for batch in expected_batches:
        filename = batch["batch"] + ".json"
        path = path_by_name.pop(filename, None)
        if path is None:
            failures.append(f"missing normalized/{filename}")
            continue
        try:
            if json.loads(path.read_text(encoding="utf-8")) == batch:
                correct_batches += 1
                continue
        except (OSError, ValueError):
            pass
        failures.append(f"invalid normalized/{filename}")
    failures.extend(f"unexpected normalized/{name}" for name in sorted(path_by_name))
    raw_unchanged = raw_hashes == _raw_hashes(workspace)
    if not raw_unchanged:
        failures.append("raw inputs changed")

    audit_passed = False
    alternate_audit_passed = False
    missing_batch_rejected = False
    if (workspace / "audit.py").is_file():
        default_expected = _expected_audit(expected_batches)
        try:
            audit_passed = _audit_matches(_run_audit(workspace), default_expected)
            with tempfile.TemporaryDirectory(dir=workspace, prefix="audit-test-") as directory:
                alt_dir = Path(directory)
                expected_batches[0]["items"][0]["units"] += 20
                expected_batches[0]["items"][0]["available"] += 20
                for batch in expected_batches:
                    (alt_dir / (batch["batch"] + ".json")).write_text(
                        json.dumps(batch), encoding="utf-8"
                    )
                alternate_audit_passed = _audit_matches(
                    _run_audit(workspace, str(alt_dir)), _expected_audit(expected_batches)
                )
                (alt_dir / "00.json").unlink()
                missing_batch_rejected = _run_audit(workspace, str(alt_dir)).returncode != 0
        except (subprocess.TimeoutExpired, OSError):
            pass
    if not audit_passed:
        failures.append("audit default input failed")
    if not alternate_audit_passed:
        failures.append("audit alternate input failed")
    if not missing_batch_rejected:
        failures.append("audit did not reject a missing batch")
    return {
        "passed": not failures,
        "correct_batches": correct_batches,
        "expected_batches": BATCH_COUNT,
        "audit_default": audit_passed,
        "audit_alternate": alternate_audit_passed,
        "audit_missing_batch": missing_batch_rejected,
        "raw_unchanged": raw_unchanged,
        "failures": failures,
    }
