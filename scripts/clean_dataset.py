#!/usr/bin/env python3
"""Derive a local training manifest and audit reports without editing raw data.

Python 3.10+; standard library only. Run from any directory:
    python3 /path/to/darps/scripts/clean_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


LIMBS = ("arm_left", "arm_right", "leg_left", "leg_right")
CLASSES = ("no_injury", "wound", "amputation", "not_testable")
FIELDS = ("casualty_id", "filepath", "media_format", *LIMBS)
MEDIA_DIRS = {"image": "images", "video": "videos"}
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".mp4", ".mov"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest(source: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    data = source.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not set(FIELDS).issubset(fields):
        raise ValueError(f"Expected unique CSV columns including {', '.join(FIELDS)}")
    rows = list(reader)
    for record, row in enumerate(rows, 1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Record {record}: wrong number of CSV fields")
        if any(not row[field].strip() for field in FIELDS):
            raise ValueError(f"Record {record}: blank required field")
        if row["media_format"] not in MEDIA_DIRS:
            raise ValueError(f"Record {record}: unknown media_format {row['media_format']!r}")
        for limb in LIMBS:
            if row[limb] not in CLASSES:
                raise ValueError(f"Record {record}: unknown {limb} label {row[limb]!r}")
    return fields, rows, data


def local_path(row: dict[str, str], root: Path) -> Path:
    """Rebase the media-relative suffix, never use another machine's absolute path."""
    root = root.resolve()
    folder = MEDIA_DIRS[row["media_format"]]
    parts = PurePosixPath(row["filepath"].replace("\\", "/")).parts
    if folder in parts:
        position = len(parts) - 1 - parts[::-1].index(folder)
        suffix = parts[position + 1 :]
    elif len(parts) == 1:
        suffix = parts
    else:
        raise ValueError(f"Cannot locate {folder}/ in source path {row['filepath']!r}")
    if not suffix or ".." in suffix:
        raise ValueError(f"Invalid media-relative path: {row['filepath']!r}")
    media_root = (root / folder).resolve()
    candidate = media_root.joinpath(*suffix).resolve()
    if not candidate.is_relative_to(media_root):
        raise ValueError(f"Path escapes {folder}/: {row['filepath']!r}")
    if not candidate.is_relative_to(root):
        raise ValueError(f"Media directory escapes the dataset root: {media_root}")
    return candidate


def snapshot(root: Path, output: Path) -> dict[str, tuple[int, int]]:
    """Inventory every source file, excluding only the derived output subtree."""
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_relative_to(output)
    }


def detected_format(path: Path) -> str:
    """Identify common container signatures; this does not decode media."""
    with path.open("rb") as stream:
        header = stream.read(64)
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[4:8] == b"ftyp":
        brands = header[8:]
        if any(brand in brands for brand in (b"heic", b"heix", b"hevc", b"hevx")):
            return "heic"
        if b"mif1" in brands or b"msf1" in brands:
            return "heif"
        if header[8:12] == b"qt  ":
            return "mov"
        return "mp4"
    return "unknown"


def expected_formats(path: Path) -> set[str]:
    return {
        ".jpg": {"jpeg"}, ".jpeg": {"jpeg"}, ".png": {"png"},
        ".heic": {"heic"}, ".heif": {"heic", "heif"},
        ".mp4": {"mp4"}, ".mov": {"mov", "mp4"},
    }.get(path.suffix.lower(), set())


def file_kind(path: Path) -> str:
    if path.name in {".DS_Store", "Thumbs.db", "desktop.ini"} or path.name.startswith("._"):
        return "system_metadata"
    return "media" if path.suffix.lower() in MEDIA_EXTENSIONS else "support_file"


def distribution(rows: list[dict[str, str]]) -> dict:
    counts = Counter(row[limb] for row in rows for limb in LIMBS)
    return {
        "rows": len(rows),
        "casualties": len({row["casualty_id"] for row in rows}),
        "label_counts": {label: counts[label] for label in CLASSES},
        "label_percentages": {
            label: round(100 * counts[label] / (4 * len(rows)), 4) if rows else 0
            for label in CLASSES
        },
        "labels_by_limb": {
            limb: {label: sum(row[limb] == label for row in rows) for label in CLASSES}
            for limb in LIMBS
        },
    }


def csv_bytes(fields: list[str] | tuple[str, ...], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_atomic(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def quality_text(summary: dict, missing: list[dict], unused: list[dict]) -> str:
    checks = summary["checks"]
    lines = [
        "Dataset cleanup quality summary", "",
        f"Canonical source: {summary['source_csv']}",
        f"Source SHA-256: {summary['source_csv_sha256']}",
        "Original CSV bytes and source file names/sizes/mtimes verified unchanged.",
        "Only filepath values and file-presence filtering were changed in the derived CSV.",
        "Original schema, row order, casualty IDs and labels are preserved.", "",
        f"Source records: {summary['source_rows']}",
        f"Cleaned records: {summary['cleaned']['rows']}",
        f"Missing files excluded: {len(missing)}",
        f"Casualties retained: {summary['cleaned']['casualties']}",
        f"Casualties lost: {len(summary['casualties_lost'])}",
        f"Unused files: {len(unused)} ({summary['unused_file_kinds']})", "",
        "Cleaned label distribution (four labels per media file):",
        "Media       Rows  Casualties  no_injury  wound  amputation  not_testable",
    ]
    for media, stats in [("all", summary["cleaned"]), *summary["by_media"].items()]:
        counts = stats["label_counts"]
        lines.append(
            f"{media:<10} {stats['rows']:>5} {stats['casualties']:>11}  "
            + "  ".join(f"{counts[label]:>9}" for label in CLASSES)
        )
    lines += ["", "Image label percentages:"]
    for label, percent in summary["by_media"]["image"]["label_percentages"].items():
        lines.append(f"  {label}: {percent:.2f}%")
    lines += [
        "", "Checks on the cleaned manifest:",
        "  Required fields populated; all media types and limb labels valid.",
        f"  Missing local paths: {checks['missing_paths']}",
        f"  Duplicate rows beyond the first occurrence: {checks['duplicate_rows']}",
        f"  Repeated resolved paths beyond the first occurrence: {checks['duplicate_paths']}",
        f"  Empty referenced files: {checks['empty_files']}",
        f"  Unknown referenced container signatures: {checks['unknown_signatures']}",
        f"  Referenced extension/signature mismatches: {checks['extension_mismatches']}",
        f"  Conflicting known injury labels within a casualty/limb: {len(summary['known_label_conflicts'])}",
        "", f"Media per casualty: {summary['media_per_casualty']}",
        f"Images with any not_testable limb: {summary['images_any_not_testable']}",
        f"Images with all four limbs not_testable: {summary['images_all_not_testable']}",
        f"Video not_testable labels: {summary['by_media']['video']['label_counts']['not_testable']}",
        "", "Missing files (source_record counts parsed data records from 1):",
    ]
    lines += [f"  record {row['source_record']}: {row['expected_filepath']}" for row in missing] or ["  None"]
    lines += ["", "All files present but unreferenced by the media manifest:"]
    lines += [f"  {row['relative_path']} [{row['kind']}, {row['detected_format']}]" for row in unused] or ["  None"]
    lines += [
        "", "Interpretation and scope:",
        "  Casualty IDs repeat across views. Keep each casualty in a single evaluation split.",
        "  not_testable is a visibility class, not a missing value; it is retained.",
        "  Video labels are not frame-level visibility labels. Qualification inputs are images.",
        "  Left/right refers to casualty anatomy; preprocessing must preserve that meaning.",
        "  Unreferenced files are reported, never deleted or added to training automatically.",
        "  File presence, size, container signatures and manifest consistency were checked.",
        "  Media were not fully decoded; visual label correctness and content duplicates were not checked.",
        "  Source media preservation uses names/sizes/mtimes, not full media content hashes.",
    ]
    return "\n".join(lines) + "\n"


def clean_dataset(root: Path, output: Path | None = None, source: Path | None = None) -> dict:
    root = root.resolve()
    source = (source or root / "train_dataset.csv").resolve()
    output = (output or root / "cleaned").resolve()
    if not root.is_dir() or not source.is_file():
        raise ValueError("Dataset root and source CSV must exist")
    if source.is_relative_to(output):
        raise ValueError("Output directory must not contain or replace the canonical source CSV")
    for folder in MEDIA_DIRS.values():
        media_root = (root / folder).resolve()
        if output.is_relative_to(media_root) or media_root.is_relative_to(output):
            raise ValueError("Output directory must be separate from the source media directories")

    fields, rows, original = read_manifest(source)
    before = snapshot(root, output)
    cleaned, audit, missing = [], [], []
    for record, row in enumerate(rows, 1):
        local = local_path(row, root)
        exists = local.is_file()
        entry = {
            "source_record": record,
            "casualty_id": row["casualty_id"],
            "media_format": row["media_format"],
            "original_filepath": row["filepath"],
            "expected_filepath": str(local),
            "status": "included" if exists else "missing",
        }
        audit.append(entry)
        if exists:
            cleaned.append({**row, "filepath": str(local)})
        else:
            missing.append(entry)

    referenced = {Path(row["filepath"]) for row in cleaned}
    stems = {path.with_suffix("") for path in referenced}
    unused = []
    for relative, (size, _) in before.items():
        path = root / relative
        if path.resolve() == source or path.resolve() in referenced:
            continue
        kind = file_kind(path)
        actual = detected_format(path) if kind == "media" else "not_media"
        unused.append({
            "relative_path": relative, "filepath": str(path), "size_bytes": size,
            "kind": kind, "extension": path.suffix.lower(), "detected_format": actual,
            "extension_mismatch": kind == "media" and actual not in expected_formats(path),
            "same_stem_as_referenced_file": path.resolve().with_suffix("") in stems,
        })

    file_issues = []
    for path in sorted(referenced):
        actual = detected_format(path)
        issues = []
        if path.stat().st_size == 0:
            issues.append("empty_file")
        if actual == "unknown":
            issues.append("unknown_signature")
        elif actual not in expected_formats(path):
            issues.append("extension_mismatch")
        if issues:
            file_issues.append({"filepath": str(path), "detected_format": actual, "issues": issues})

    groups = defaultdict(list)
    for row in cleaned:
        groups[row["casualty_id"]].append(row)
    conflicts = []
    for casualty, records in sorted(groups.items()):
        for limb in LIMBS:
            labels = sorted({row[limb] for row in records} - {"not_testable"})
            if len(labels) > 1:
                conflicts.append({"casualty_id": casualty, "limb": limb, "labels": labels})
    group_sizes = [len(records) for records in groups.values()]
    images = [row for row in cleaned if row["media_format"] == "image"]
    cleaned_data = csv_bytes(fields, cleaned)
    summary = {
        "source_csv": str(source), "source_csv_sha256": sha256(original),
        "cleaned_csv": str(output / "train_dataset_cleaned.csv"),
        "cleaned_csv_sha256": sha256(cleaned_data),
        "source_rows": len(rows), "missing_rows": len(missing),
        "cleaned": distribution(cleaned),
        "by_media": {media: distribution([r for r in cleaned if r["media_format"] == media]) for media in MEDIA_DIRS},
        "casualties_lost": sorted({r['casualty_id'] for r in rows} - set(groups)),
        "media_per_casualty": {
            "min": min(group_sizes, default=0),
            "median": statistics.median(group_sizes) if group_sizes else 0,
            "max": max(group_sizes, default=0),
        },
        "images_any_not_testable": sum(any(r[s] == "not_testable" for s in LIMBS) for r in images),
        "images_all_not_testable": sum(all(r[s] == "not_testable" for s in LIMBS) for r in images),
        "unused_file_count": len(unused),
        "unused_file_kinds": dict(sorted(Counter(r["kind"] for r in unused).items())),
        "unused_media_extension_mismatches": sum(r["extension_mismatch"] for r in unused),
        "known_label_conflicts": conflicts, "referenced_file_issues": file_issues,
        "checks": {
            "missing_paths": sum(not Path(r["filepath"]).is_file() for r in cleaned),
            "duplicate_rows": len(cleaned) - len({tuple(r[f] for f in fields) for r in cleaned}),
            "duplicate_paths": len(cleaned) - len(referenced),
            "empty_files": sum("empty_file" in r["issues"] for r in file_issues),
            "unknown_signatures": sum("unknown_signature" in r["issues"] for r in file_issues),
            "extension_mismatches": sum("extension_mismatch" in r["issues"] for r in file_issues),
        },
        "scope": "Path/presence cleanup only. Labels unchanged. No full media decoding or content deduplication.",
    }
    if len(rows) != len(cleaned) + len(missing) or len(audit) != len(rows):
        raise RuntimeError("Source records did not reconcile with cleanup decisions")

    output.mkdir(parents=True, exist_ok=True)
    audit_fields = list(audit[0]) if audit else [
        "source_record", "casualty_id", "media_format", "original_filepath", "expected_filepath", "status",
    ]
    unused_fields = [
        "relative_path", "filepath", "size_bytes", "kind", "extension", "detected_format",
        "extension_mismatch", "same_stem_as_referenced_file",
    ]
    for name, data in {
        "train_dataset_cleaned.csv": cleaned_data,
        "path_audit.csv": csv_bytes(audit_fields, audit),
        "missing_files.csv": csv_bytes(audit_fields, missing),
        "unused_files.csv": csv_bytes(unused_fields, unused),
    }.items():
        write_atomic(output / name, data)

    if source.read_bytes() != original or snapshot(root, output) != before:
        raise RuntimeError("Canonical source changed during cleanup; derived results need review")
    summary["source_csv_unchanged"] = True
    summary["source_file_inventory_unchanged"] = True
    summary["source_inventory_check"] = "Relative filenames, byte sizes and modification times; excludes derived output directory"
    write_atomic(output / "quality_summary.json", (json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    write_atomic(output / "quality_summary.txt", quality_text(summary, missing, unused).encode("utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "train")
    parser.add_argument("--output-dir", type=Path, help="Derived artifacts directory; defaults to DATASET_ROOT/cleaned")
    parser.add_argument("--csv", type=Path, help="Canonical source CSV; defaults to DATASET_ROOT/train_dataset.csv")
    args = parser.parse_args()
    try:
        result = clean_dataset(args.dataset_root, args.output_dir, args.csv)
    except (ValueError, OSError, csv.Error, UnicodeError, RuntimeError) as error:
        parser.exit(1, f"Cleanup failed: {error}\n")
    print(f"Cleaned {result['source_rows']} -> {result['cleaned']['rows']} records; "
          f"{result['missing_rows']} missing; {result['unused_file_count']} unused files.")
    print(f"Canonical source unchanged. Derived artifacts: {Path(result['cleaned_csv']).parent}")


if __name__ == "__main__":
    main()
