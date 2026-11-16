#!/usr/bin/env python3
"""Audit manifest images and make review previews without modifying source data.

Python 3.10+ and Pillow 10.1+. Run after clean_dataset.py:
    python3 scripts/audit_images.py
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import warnings
from collections import Counter, defaultdict, deque
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

from clean_dataset import (
    CLASSES, LIMBS, csv_bytes, distribution, local_path, read_manifest, sha256,
    write_atomic,
)


BACKGROUND = (114, 114, 114)
PREVIEW_SIZE = 448
RARE_CASUALTY_LIMIT = 3
CLASS_SAMPLE_SIZE = 24
PAGE_SIZE = 12
VALIDATION_FIELDS = [
    "source_record", "manifest_record", "casualty_id", "filepath", "original_filepath", *LIMBS,
    "status", "error_type", "error_message", "size_bytes", "image_sha256", "detected_format",
    "frame_count", "primary_frame", "additional_frames_decoded", "frame_dimensions",
    "original_mode", "original_width", "original_height", "exif_orientation", "orientation_applied",
    "oriented_width", "oriented_height", "output_mode", "transparency_composited",
    "preview_filepath", "preview_size", "content_width", "content_height", "scale",
    "upscaled", "padding_left", "padding_top", "padding_right", "padding_bottom",
    "decoder_warnings", "review_flags",
]


def load_image(path: Path) -> tuple[Image.Image, dict]:
    """Return full-resolution display-oriented RGB pixels; caller closes the image.

    No resize, crop, augmentation, colour enhancement, label change or file write.
    MPO containers use primary picture 0; all stored pictures are decoded and recorded.
    Other multi-frame formats and truncated images are rejected.
    """
    data = path.read_bytes()
    old_truncated_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(io.BytesIO(data)) as original:
            frame_count = getattr(original, "n_frames", 1)
            if frame_count != 1 and original.format != "MPO":
                raise ValueError("Unsupported image sequence; expected a still image or MPO primary picture")
            frame_dimensions = []
            for frame in range(frame_count):
                original.seek(frame)
                original.load()
                frame_dimensions.append({"frame": frame, "width": original.width,
                                         "height": original.height, "mode": original.mode})
            original.seek(0)  # The manifest's still image is the MPO primary picture.
            original.load()  # Force actual pixel decoding, not just header identification.
            orientation = original.getexif().get(274)
            metadata = {
                "size_bytes": len(data), "image_sha256": sha256(data),
                "detected_format": original.format, "original_mode": original.mode,
                "frame_count": frame_count, "primary_frame": 0,
                "additional_frames_decoded": frame_count - 1,
                "frame_dimensions": json.dumps(frame_dimensions, separators=(",", ":")),
                "original_width": original.width, "original_height": original.height,
                "exif_orientation": orientation if orientation is not None else "absent",
                "orientation_applied": orientation in range(2, 9) if isinstance(orientation, int) else False,
            }
            with ImageOps.exif_transpose(original) as oriented:
                transparent = "A" in oriented.getbands() or "transparency" in oriented.info
                if transparent:
                    with oriented.convert("RGBA") as rgba:
                        with Image.new("RGBA", rgba.size, (*BACKGROUND, 255)) as background:
                            with Image.alpha_composite(background, rgba) as composited:
                                rgb = composited.convert("RGB")
                else:
                    rgb = oriented.convert("RGB")
                metadata.update({
                    "oriented_width": rgb.width, "oriented_height": rgb.height,
                    "output_mode": rgb.mode, "transparency_composited": transparent,
                })
        return rgb, metadata
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old_truncated_setting


def make_preview(image: Image.Image, size: int) -> tuple[Image.Image, dict]:
    """Fit the entire RGB image into a square, preserving aspect ratio, with padding."""
    if size < 1:
        raise ValueError("Preview size must be positive")
    with ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS) as resized:
        left, top = (size - resized.width) // 2, (size - resized.height) // 2
        preview = Image.new("RGB", (size, size), BACKGROUND)
        preview.paste(resized, (left, top))
        metadata = {
            "preview_size": size, "content_width": resized.width, "content_height": resized.height,
            "scale": round(size / max(image.size), 8), "upscaled": max(image.size) < size,
            "padding_left": left, "padding_top": top,
            "padding_right": size - resized.width - left, "padding_bottom": size - resized.height - top,
        }
    return preview, metadata


def image_bytes(image: Image.Image, file_format: str) -> bytes:
    stream = io.BytesIO()
    options = {"quality": 92, "subsampling": 0} if file_format == "JPEG" else {}
    image.save(stream, format=file_format, **options)
    return stream.getvalue()


def source_inventory(root: Path, output: Path) -> dict:
    exclusions = [root / "cleaned", output]
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not any(path.is_relative_to(folder) for folder in exclusions)
    }


def manifest_records(root: Path, manifest: Path, source: Path) -> tuple[list, list, bytes, bytes]:
    """Match every cleaned record to an unchanged source record, including duplicates."""
    fields, rows, manifest_data = read_manifest(manifest)
    source_fields, source_rows, source_data = read_manifest(source)
    if fields != source_fields:
        raise ValueError("Cleaned and canonical CSV schemas differ")
    candidates = defaultdict(deque)
    for index, row in enumerate(source_rows, 1):
        rebased = {**row, "filepath": str(local_path(row, root))}
        candidates[tuple(rebased[field] for field in fields)].append((index, row["filepath"]))
    records = []
    for index, row in enumerate(rows, 1):
        key = tuple(row[field] for field in fields)
        if not candidates[key]:
            raise ValueError(f"Cleaned record {index} cannot be traced to the canonical CSV")
        source_record, original_path = candidates[key].popleft()
        records.append({
            **row, "source_record": source_record, "manifest_record": index,
            "original_filepath": original_path,
        })
    return fields, records, manifest_data, source_data


def representative_sample(records: list[dict], limit: int, group_key: str = "casualty_id") -> list[dict]:
    """Deterministic round-robin: prioritize different casualties before repeated views."""
    groups = defaultdict(deque)
    for record in records:
        groups[str(record[group_key])].append(record)
    selected = []
    while groups and len(selected) < limit:
        for key in list(groups):
            selected.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
            if len(selected) == limit:
                break
    return selected


def short_text(draw: ImageDraw.ImageDraw, value: str, font, width: int) -> str:
    if draw.textbbox((0, 0), value, font=font)[2] <= width:
        return value
    while value and draw.textbbox((0, 0), value + "...", font=font)[2] > width:
        value = value[:-1]
    return value + "..."


def contact_sheet(records: list[dict], title: str, page: int, pages: int, output: Path) -> Image.Image:
    tile, gap, columns, label_height, header = 320, 16, 4, 128, 76
    cell_width, cell_height = tile + gap, tile + label_height + gap
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width + gap, header + rows * cell_height + gap), "#f1f3f5")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = (ImageFont.load_default(size=size) for size in (22, 15, 13))
    draw.text((gap, 10), short_text(draw, title, title_font, canvas.width - 2 * gap), font=title_font, fill="#17212b")
    draw.text((gap, 42), f"Page {page}/{pages} | Anatomical left/right | REVIEW ONLY - labels are printed below each image",
              font=small_font, fill="#435363")
    for index, record in enumerate(records):
        x = gap + (index % columns) * cell_width
        y = header + (index // columns) * cell_height
        draw.rectangle((x - 4, y - 4, x + tile + 4, y + tile + label_height - 2), fill="white")
        with Image.open(output / record["preview_filepath"]) as preview:
            with preview.resize((tile, tile), Image.Resampling.LANCZOS) as thumbnail:
                canvas.paste(thumbnail, (x, y))
        labels = [
            f"{Path(record['filepath']).name} | source #{record['source_record']}", record["casualty_id"],
            f"Left arm: {record['arm_left']}", f"Right arm: {record['arm_right']}",
            f"Left leg: {record['leg_left']}", f"Right leg: {record['leg_right']}",
        ]
        for line, text in enumerate(labels):
            draw.text((x + 3, y + tile + 5 + line * 19), short_text(draw, text, body_font, tile - 6),
                      font=body_font, fill="#17212b")
    return canvas


def galleries(records: list[dict]) -> list[dict]:
    good = [record for record in records if record["status"] == "ok"]
    groups = []
    by_casualty = defaultdict(list)
    for record in good:
        by_casualty[record["casualty_id"]].append(record)
    for casualty, members in sorted(by_casualty.items()):
        groups.append({"kind": "by_casualty", "title": casualty, "eligible": len(members), "records": members})
    for limb in LIMBS:
        for label in CLASSES:
            members = [record for record in good if record[limb] == label]
            casualties = len({record["casualty_id"] for record in members})
            selected = members if casualties <= RARE_CASUALTY_LIMIT else representative_sample(members, CLASS_SAMPLE_SIZE)
            if members:
                groups.append({
                    "kind": "by_class", "title": f"{limb}: {label}", "eligible": len(members), "records": selected,
                })
    review = [record for record in good if record["review_flags"]]
    oriented = [record for record in good if record["orientation_applied"]]
    for title, members, selected in [
        ("Review candidates (flags are not confirmed label errors)", review, review),
        ("EXIF orientation examples", oriented, representative_sample(oriented, CLASS_SAMPLE_SIZE, "exif_orientation")),
    ]:
        if members:
            groups.append({"kind": "review", "title": title, "eligible": len(members), "records": selected})
    return groups


def render_galleries(records: list[dict], output: Path) -> list[dict]:
    index = []
    for group in galleries(records):
        title = group["title"]
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:80] + "_" + sha256(title.encode())[:8]
        members = group["records"]
        pages = (len(members) + PAGE_SIZE - 1) // PAGE_SIZE
        for page in range(1, pages + 1):
            selected = members[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
            relative = Path("contact_sheets") / group["kind"] / f"{slug}_{page:03}.jpg"
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
            with contact_sheet(selected, title, page, pages, output) as sheet:
                write_atomic(output / relative, image_bytes(sheet, "JPEG"))
            index.append({
                "gallery": group["kind"], "group": title, "page": page, "pages": pages,
                "filepath": relative.as_posix(), "images_on_page": len(selected),
                "eligible_images": group["eligible"], "selected_images": len(members),
                "source_records": ";".join(str(record["source_record"]) for record in selected),
            })
    return index


def html_report(summary: dict, sheets: list[dict]) -> str:
    escape = html.escape
    groups = defaultdict(list)
    for sheet in sheets:
        groups[(sheet["gallery"], sheet["group"])].append(sheet)
    sections = []
    for kind, heading in [("review", "Review candidates and orientation examples"),
                          ("by_class", "By limb and class"), ("by_casualty", "All images by casualty")]:
        sections.append(f"<h2>{heading}</h2>")
        for (gallery, title), pages in groups.items():
            if gallery != kind:
                continue
            first = pages[0]
            images = "".join(
                f'<a href="{escape(page["filepath"], quote=True)}"><img loading="lazy" '
                f'src="{escape(page["filepath"], quote=True)}" alt="{escape(title, quote=True)} '
                f'page {page["page"]}"></a>' for page in pages
            )
            sections.append(
                f'<details><summary>{escape(title)} — {first["selected_images"]} of '
                f'{first["eligible_images"]} images, {len(pages)} page(s)</summary>{images}</details>'
            )
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image audit and contact sheets</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#17212b;background:#f8fafb}}
h1{{font-size:30px}}h2{{margin-top:32px}}a{{color:#155ca3}}details{{background:white;border:1px solid #d8e0e7;border-radius:6px;margin:10px 0;padding:12px}}
summary{{cursor:pointer;font-weight:600}}img{{display:block;width:100%;max-width:1360px;height:auto;margin:16px auto}}code{{background:#e9eef2;padding:2px 5px}}
</style>
<h1>Image audit and contact sheets</h1>
<p>{summary['images_checked']} images checked; {summary['images_passed']} passed; {summary['images_failed']} failed.
{summary['orientation_applied']} required EXIF orientation handling. {summary['review_candidate_images']} have review flags.</p>
<p>Original files and labels remain unchanged. Contact sheets have printed labels and are for review only.
Unannotated previews use {summary['preview_policy']['size']} × {summary['preview_policy']['size']} RGB canvases with padding and no crop.</p>
<p>Flags identify candidates for inspection, not confirmed annotation errors. No visual label corrections have been applied.
Open a contact sheet at full size to read labels, then inspect the original for uncertain details.</p>
<p><a href="image_audit_summary.txt">Summary</a> · <a href="image_validation.csv">Every image</a> ·
<a href="decode_failures.csv">Technical failures</a> · <a href="review_flags.csv">Review queue</a> ·
<a href="class_coverage.csv">Class coverage</a> · <a href="image_dataset_validated.csv">Validated image manifest</a></p>
<p>By-casualty galleries cover every successfully loaded image. Class galleries show up to {CLASS_SAMPLE_SIZE}
casualty-diverse examples, or every image when a class has at most {RARE_CASUALTY_LIMIT} casualties.</p>
{''.join(sections)}
</html>'''


def summary_text(summary: dict) -> str:
    lines = [
        "Image audit quality summary", "",
        f"Canonical source: {summary['source_csv']}", f"Source SHA-256: {summary['source_csv_sha256']}",
        f"Cleaned manifest SHA-256: {summary['manifest_sha256']}",
        f"Pillow version: {summary['pillow_version']}", "",
        f"Images fully decoded: {summary['images_passed']} / {summary['images_checked']}",
        f"Technical failures: {summary['images_failed']}", f"Videos not inspected: {summary['videos_skipped']}",
        f"Casualties with passing images: {summary['validated']['casualties']}",
        f"EXIF orientation applied: {summary['orientation_applied']}",
        f"Original colour modes: {summary['original_modes']}",
        f"Original formats: {summary['formats']}",
        f"Images with additional MPO pictures: {summary['mpo_images']}",
        f"Total stored pictures fully decoded: {summary['stored_pictures_decoded']}",
        f"Review candidates: {summary['review_candidate_images']}",
        f"Review flag counts (may overlap): {summary['review_flag_counts']}",
        f"Contact sheet pages: {summary['contact_sheet_pages']}", "",
        "Rare class coverage (at most three independent casualties):",
    ]
    lines += [f"  {row['limb']}/{row['label']}: {row['validated_images']} images, {row['validated_casualties']} casualties"
              for row in summary["rare_class_coverage"]] or ["  None"]
    lines += [
        "", "Preservation checks:",
        "  Original CSV and input cleaned manifest bytes verified unchanged.",
        "  Canonical source file names, sizes and modification times verified unchanged.",
        "  Every successfully read input image rehashed after processing; bytes verified unchanged.",
        "  Labels, casualty IDs and order preserved in the validated image-only manifest.",
        "", "Preview policy:",
        f"  {summary['preview_policy']['size']}px square RGB, EXIF orientation applied once, LANCZOS resize.",
        "  Entire frame retained with centered grey padding (RGB 114,114,114).",
        "  No crop, random augmentation, colour/contrast enhancement or source overwrite.",
        "  Resizing is for audit previews, not a final model input-size decision.",
        "  Annotated contact sheets are review artifacts, never training inputs.",
        "", "Review flag definitions:",
        "  upscaled_for_preview: original longest side is shorter than the preview canvas.",
        "  unusual_aspect_ratio: longest side / shortest side >= 3.",
        "  all_limbs_not_testable: all four original labels are not_testable; valid but worth reviewing.",
        "  rare_casualty_label_support: a limb/class occurs in <= 3 passing casualties.",
        "  decoder_warning, unusual_exif_orientation, transparency_composited: inspect handling.",
        "  mpo_additional_pictures: all pictures decoded; primary picture 0 used for the still-image input.",
        "  technical_failure: pixels or preview could not be loaded/processed; see error details.",
        "", "Scope:",
        "  Flags are review candidates, not proof of poor quality or incorrect labels.",
        "  No automated claim is made about limb visibility, wound visibility or annotation correctness.",
        "  MPO files are still-image containers, not videos. Secondary pictures are never separate labeled examples.",
        "  All passing images remain eligible, including low resolution and not_testable examples.",
        "  Colour correctness under embedded ICC profiles and content duplicates were not assessed.",
        "  No training/validation split has been made. Videos were not decoded.",
        "  Browse index.html; all per-casualty galleries cover every passing image.",
    ]
    return "\n".join(lines) + "\n"


def audit_images(root: Path, output: Path | None = None, preview_size: int = PREVIEW_SIZE) -> dict:
    root = root.resolve()
    manifest, source = root / "cleaned" / "train_dataset_cleaned.csv", root / "train_dataset.csv"
    output = (output or root / "cleaned" / "image_audit").resolve()
    if preview_size < 32 or preview_size > 2048:
        raise ValueError("Preview size must be between 32 and 2048")
    if any(path.resolve().is_relative_to(output) for path in [source, manifest]):
        raise ValueError("Output directory must not contain the input manifests")
    for folder in [root / "images", root / "videos"]:
        folder = folder.resolve()
        if output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError("Output directory must be separate from canonical media directories")
    # Avoid following pre-existing output symlinks into source folders.
    if output.exists() and any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Output directory must not contain symlinks")
    fields, records, manifest_data, source_data = manifest_records(root, manifest, source)
    before = source_inventory(root, output)
    images = [record for record in records if record["media_format"] == "image"]
    (output / "previews").mkdir(parents=True, exist_ok=True)
    audited, hashes = [], {}
    for index, record in enumerate(images, 1):
        row = {field: "" for field in VALIDATION_FIELDS}
        row.update({field: record[field] for field in ["source_record", "manifest_record", "casualty_id", "filepath", "original_filepath", *LIMBS]})
        path = Path(record["filepath"])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                # Record the bytes even when full decoding subsequently fails.
                data = path.read_bytes()
                row.update({"size_bytes": len(data), "image_sha256": sha256(data)})
                hashes[path] = row["image_sha256"]
                rgb, metadata = load_image(path)
                row.update(metadata)
                if metadata["image_sha256"] != hashes[path]:
                    raise RuntimeError("Source image changed while being read")
                with rgb:
                    preview, geometry = make_preview(rgb, preview_size)
                    row.update(geometry)
                    relative = Path("previews") / f"source_{record['source_record']:06}.png"
                    with preview:
                        write_atomic(output / relative, image_bytes(preview, "PNG"))
                    row["preview_filepath"] = relative.as_posix()
                row["status"] = "ok"
            except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as error:
                message = re.sub(r"<_io\.BytesIO object at 0x[0-9a-fA-F]+>", str(path), str(error))
                row.update({"status": "error", "error_type": type(error).__name__, "error_message": message})
            row["decoder_warnings"] = " | ".join(str(item.message) for item in captured)
        audited.append(row)
        if index % 100 == 0 or index == len(images):
            print(f"Decoded {index}/{len(images)} images", flush=True)

    good = [row for row in audited if row["status"] == "ok"]
    coverage = []
    for limb in LIMBS:
        for label in CLASSES:
            members = [row for row in good if row[limb] == label]
            casualties = {row["casualty_id"] for row in members}
            coverage.append({
                "limb": limb, "label": label,
                "input_images": sum(row[limb] == label for row in images),
                "validated_images": len(members), "validated_casualties": len(casualties),
                "rare_casualty_support": bool(members) and len(casualties) <= RARE_CASUALTY_LIMIT,
            })
    rare = {(row["limb"], row["label"]) for row in coverage if row["rare_casualty_support"]}
    for row in audited:
        flags = []
        if row["status"] != "ok":
            flags.append("technical_failure")
        else:
            if row["upscaled"]:
                flags.append("upscaled_for_preview")
            if max(row["oriented_width"], row["oriented_height"]) / min(row["oriented_width"], row["oriented_height"]) >= 3:
                flags.append("unusual_aspect_ratio")
            if all(row[limb] == "not_testable" for limb in LIMBS):
                flags.append("all_limbs_not_testable")
            if any((limb, row[limb]) in rare for limb in LIMBS):
                flags.append("rare_casualty_label_support")
            if row["exif_orientation"] not in ["absent", *range(1, 9)]:
                flags.append("unusual_exif_orientation")
            if row["transparency_composited"]:
                flags.append("transparency_composited")
            if row["additional_frames_decoded"]:
                flags.append("mpo_additional_pictures")
        if row["decoder_warnings"]:
            flags.append("decoder_warning")
        row["review_flags"] = ";".join(flags)

    print("Rendering contact sheets", flush=True)
    sheets = render_galleries(audited, output)
    passed_ids = {row["source_record"] for row in good}
    validated = [{field: record[field] for field in fields} for record in images if record["source_record"] in passed_ids]
    review = [{**row, "review_status": "pending", "reviewer_note": ""} for row in audited if row["review_flags"]]
    failure = [row for row in audited if row["status"] != "ok"]
    for name, columns, rows in [
        ("image_validation.csv", VALIDATION_FIELDS, audited),
        ("decode_failures.csv", VALIDATION_FIELDS, failure),
        ("review_flags.csv", [*VALIDATION_FIELDS, "review_status", "reviewer_note"], review),
        ("image_dataset_validated.csv", fields, validated),
        ("class_coverage.csv", ["limb", "label", "input_images", "validated_images", "validated_casualties", "rare_casualty_support"], coverage),
        ("contact_sheet_index.csv", ["gallery", "group", "page", "pages", "filepath", "images_on_page", "eligible_images", "selected_images", "source_records"], sheets),
    ]:
        write_atomic(output / name, csv_bytes(columns, rows))

    print("Verifying original image hashes and source preservation", flush=True)
    changed = [str(path) for path, digest in hashes.items() if not path.is_file() or sha256(path.read_bytes()) != digest]
    if changed or source.read_bytes() != source_data or manifest.read_bytes() != manifest_data or source_inventory(root, output) != before:
        raise RuntimeError(f"Inputs changed during audit; results need review. Changed images: {changed}")
    covered = {int(record) for sheet in sheets if sheet["gallery"] == "by_casualty" for record in sheet["source_records"].split(";")}
    if covered != passed_ids or len(audited) != len(good) + len(failure):
        raise RuntimeError("Image audit/contact sheet coverage did not reconcile")
    summary = {
        "source_csv": str(source), "source_csv_sha256": sha256(source_data),
        "manifest": str(manifest), "manifest_sha256": sha256(manifest_data),
        "pillow_version": PIL.__version__, "images_checked": len(images), "images_passed": len(good),
        "images_failed": len(failure), "videos_skipped": sum(row["media_format"] == "video" for row in records),
        "validated": distribution(validated), "orientation_applied": sum(row["orientation_applied"] for row in good),
        "exif_orientations": dict(sorted(Counter(str(row["exif_orientation"]) for row in good).items())),
        "original_modes": dict(Counter(row["original_mode"] for row in good)),
        "formats": dict(Counter(row["detected_format"] for row in good)),
        "mpo_images": sum(row["detected_format"] == "MPO" for row in good),
        "stored_pictures_decoded": sum(row["frame_count"] for row in good),
        "original_dimensions": [{"width": size[0], "height": size[1], "count": count} for size, count in
                                Counter((row["original_width"], row["original_height"]) for row in good).most_common()],
        "review_candidate_images": len(review),
        "review_flag_counts": dict(sorted(Counter(flag for row in review for flag in row["review_flags"].split(";")).items())),
        "rare_class_coverage": [row for row in coverage if row["rare_casualty_support"]],
        "contact_sheet_pages": len(sheets), "contact_sheet_pages_by_gallery": dict(Counter(row["gallery"] for row in sheets)),
        "preview_policy": {"size": preview_size, "mode": "RGB", "resampling": "LANCZOS", "padding_rgb": BACKGROUND,
                           "crop": False, "apply_exif_orientation": True, "final_training_resolution": False},
        "input_manifests_unchanged": True, "canonical_file_inventory_unchanged": True,
        "input_image_hashes_verified_unchanged": len(hashes), "labels_changed": 0,
        "visual_label_review_complete": False,
        "scope": "Full still-image pixel decoding and deterministic preview audit. No video decoding, automatic label correction, content deduplication or split creation.",
    }
    write_atomic(output / "image_audit_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    write_atomic(output / "image_audit_summary.txt", summary_text(summary).encode())
    write_atomic(output / "index.html", html_report(summary, sheets).encode())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "train")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preview-size", type=int, default=PREVIEW_SIZE)
    args = parser.parse_args()
    try:
        summary = audit_images(args.dataset_root, args.output_dir, args.preview_size)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Image audit failed: {error}\n")
    print(f"Audit complete: {summary['images_passed']}/{summary['images_checked']} images passed; "
          f"{summary['images_failed']} failed; {summary['review_candidate_images']} review candidates.")
    print(f"Report: {(args.output_dir or args.dataset_root / 'cleaned' / 'image_audit').resolve() / 'index.html'}")


if __name__ == "__main__":
    main()
