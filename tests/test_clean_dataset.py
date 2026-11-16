import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clean_dataset.py"
SPEC = importlib.util.spec_from_file_location("clean_dataset", SCRIPT)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (Path(self.temporary.name) / "train").resolve()
        (self.root / "images" / "nested").mkdir(parents=True)
        (self.root / "videos").mkdir()
        self.source = self.root / "train_dataset.csv"
        self.image = self.root / "images" / "nested" / "0001.jpg"
        self.image.write_bytes(b"\xff\xd8\xfffixture")
        (self.root / "README.md").write_text("Canonical documentation")

    def row(self, filepath, **overrides):
        return {
            "casualty_id": "casualty_001", "filepath": filepath,
            "media_format": "image", "arm_left": "not_testable",
            "arm_right": "wound", "leg_left": "no_injury",
            "leg_right": "amputation", **overrides,
        }

    def write_source(self, rows):
        # Exercise CSV quoting and CRLF input as well as source-byte preservation.
        with self.source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=cleanup.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def load_csv(self, name):
        with (self.root / "cleaned" / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_reconciles_all_records_and_files_without_changing_source(self):
        rows = [
            self.row("/old/machine/train/images/nested/0001.jpg", casualty_id="casualty,001"),
            self.row("/old/machine/train/images/missing.jpg"),
        ]
        self.write_source(rows)
        (self.root / "images" / "Thumbs.db").write_bytes(b"metadata")
        # Same stem does not authorize substitution or automatic inclusion.
        extra = self.root / "images" / "nested" / "0001.png"
        extra.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic")
        before = {str(p.relative_to(self.root)): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}

        summary = cleanup.clean_dataset(self.root)
        cleaned = self.load_csv("train_dataset_cleaned.csv")
        self.assertEqual(cleaned, [{**rows[0], "filepath": str(self.image)}])
        self.assertEqual(summary["source_rows"], summary["cleaned"]["rows"] + summary["missing_rows"])
        self.assertEqual([r["source_record"] for r in self.load_csv("path_audit.csv")], ["1", "2"])
        self.assertEqual(self.load_csv("missing_files.csv")[0]["source_record"], "2")
        unused = {r["relative_path"]: r for r in self.load_csv("unused_files.csv")}
        self.assertEqual(set(unused), {"README.md", "images/Thumbs.db", "images/nested/0001.png"})
        self.assertEqual(unused["images/nested/0001.png"]["detected_format"], "heic")
        self.assertEqual(unused["images/nested/0001.png"]["same_stem_as_referenced_file"], "True")
        self.assertTrue(summary["source_csv_unchanged"])
        self.assertTrue(summary["source_file_inventory_unchanged"])
        for name, data in before.items():
            self.assertEqual((self.root / name).read_bytes(), data)

        # A rerun must not count its own artifacts or change its results.
        first_outputs = {p.name: p.read_bytes() for p in (self.root / "cleaned").iterdir()}
        self.assertEqual(cleanup.clean_dataset(self.root), summary)
        self.assertEqual({p.name: p.read_bytes() for p in (self.root / "cleaned").iterdir()}, first_outputs)
        saved = json.loads((self.root / "cleaned" / "quality_summary.json").read_text())
        self.assertEqual(saved["cleaned_csv_sha256"], cleanup.sha256(first_outputs["train_dataset_cleaned.csv"]))

    def test_rejects_output_locations_that_overlap_canonical_data(self):
        self.write_source([self.row("images/nested/0001.jpg")])
        original = self.source.read_bytes()
        for output in [self.root, self.root.parent, self.root / "images" / "derived", self.root / "videos"]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                cleanup.clean_dataset(self.root, output)
        self.assertEqual(self.source.read_bytes(), original)

    def test_normalizes_windows_paths_and_rejects_traversal(self):
        row = self.row(r"C:\old\train\images\nested\0001.jpg")
        self.assertEqual(cleanup.local_path(row, self.root), self.image)
        for unsafe in ["images/../train_dataset.csv", "images", "/unrelated/0001.jpg"]:
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                cleanup.local_path(self.row(unsafe), self.root)

    def test_does_not_silently_change_or_deduplicate_labels(self):
        row = self.row("images/nested/0001.jpg")
        self.write_source([row, row])
        summary = cleanup.clean_dataset(self.root)
        self.assertEqual(summary["cleaned"]["rows"], 2)
        self.assertEqual(summary["checks"]["duplicate_rows"], 1)
        self.assertEqual(summary["checks"]["duplicate_paths"], 1)
        self.assertEqual(summary["cleaned"]["label_counts"]["not_testable"], 2)

    def test_invalid_source_stops_before_producing_a_cleaned_manifest(self):
        self.write_source([self.row("images/nested/0001.jpg", arm_left="unknown")])
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "unknown arm_left label"):
            cleanup.clean_dataset(self.root)
        self.assertFalse((self.root / "cleaned").exists())
        self.assertEqual(self.source.read_bytes(), original)

    def test_empty_source_and_all_missing_rows_are_reported(self):
        for rows in [[], [self.row("images/missing.jpg")]]:
            with self.subTest(rows=rows):
                self.write_source(rows)
                summary = cleanup.clean_dataset(self.root)
                self.assertEqual(summary["cleaned"]["rows"], 0)
                self.assertEqual(summary["missing_rows"], len(rows))
                self.assertEqual(self.load_csv("train_dataset_cleaned.csv"), [])


if __name__ == "__main__":
    unittest.main()
