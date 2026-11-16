import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageFile


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_images as audit
import clean_dataset as cleanup


class ImageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (Path(self.temporary.name) / "train").resolve()
        (self.root / "images").mkdir(parents=True)
        (self.root / "videos").mkdir()
        (self.root / "cleaned").mkdir()

    def row(self, name, **overrides):
        return {
            "casualty_id": 'case_"<001>', "filepath": f"/old/train/images/{name}",
            "media_format": "image", "arm_left": "not_testable", "arm_right": "wound",
            "leg_left": "no_injury", "leg_right": "amputation", **overrides,
        }

    def manifests(self, rows):
        source = self.root / "train_dataset.csv"
        cleaned = self.root / "cleaned" / "train_dataset_cleaned.csv"
        source.write_bytes(cleanup.csv_bytes(cleanup.FIELDS, rows))
        cleaned.write_bytes(cleanup.csv_bytes(cleanup.FIELDS, [
            {**row, "filepath": str(cleanup.local_path(row, self.root))} for row in rows
        ]))

    def load_csv(self, name):
        with (self.root / "cleaned" / "image_audit" / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_exif_orientation_is_applied_once_and_source_is_unchanged(self):
        path = self.root / "images" / "oriented.png"
        with Image.new("RGB", (3, 2)) as original:
            original.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                              (255, 255, 0), (255, 0, 255), (0, 255, 255)])
            exif = Image.Exif()
            exif[274] = 6
            original.save(path, exif=exif)
            expected = original.transpose(Image.Transpose.ROTATE_270)
        before = path.read_bytes()
        pixels, metadata = audit.load_image(path)
        with pixels, expected:
            self.assertEqual(pixels.size, (2, 3))
            self.assertEqual(pixels.tobytes(), expected.tobytes())
            self.assertIsNone(pixels.getexif().get(274))
            with audit.ImageOps.exif_transpose(pixels) as twice:
                self.assertEqual(twice.tobytes(), pixels.tobytes())
        self.assertTrue(metadata["orientation_applied"])
        self.assertEqual(metadata["exif_orientation"], 6)
        self.assertEqual(path.read_bytes(), before)

    def test_preview_preserves_frame_and_adds_padding_without_annotations(self):
        with Image.new("RGB", (400, 300), (20, 40, 60)) as original:
            preview, geometry = audit.make_preview(original, 448)
            with preview:
                self.assertEqual(preview.size, (448, 448))
                self.assertEqual((geometry["content_width"], geometry["content_height"]), (448, 336))
                self.assertEqual((geometry["padding_top"], geometry["padding_bottom"]), (56, 56))
                self.assertEqual(preview.getpixel((0, 0)), audit.BACKGROUND)
                self.assertEqual(preview.getpixel((0, 56)), (20, 40, 60))
                self.assertEqual(preview.getpixel((447, 391)), (20, 40, 60))

    def test_grayscale_and_transparency_have_explicit_rgb_handling(self):
        for mode, colour in [("L", 80), ("RGBA", (255, 0, 0, 0))]:
            with self.subTest(mode=mode):
                path = self.root / "images" / f"{mode}.png"
                with Image.new(mode, (12, 8), colour) as image:
                    image.save(path)
                pixels, metadata = audit.load_image(path)
                with pixels:
                    self.assertEqual(pixels.mode, "RGB")
                    self.assertEqual(pixels.getpixel((0, 0)), (80, 80, 80) if mode == "L" else audit.BACKGROUND)
                self.assertEqual(metadata["transparency_composited"], mode == "RGBA")

    def test_header_readable_truncated_jpeg_fails_full_decode(self):
        path = self.root / "images" / "truncated.jpg"
        with Image.effect_noise((128, 128), 50).convert("RGB") as image:
            image.save(path, quality=90)
        path.write_bytes(path.read_bytes()[:-100])
        with Image.open(path) as image:
            self.assertEqual(image.format, "JPEG")  # Header still identifies correctly.
        permissive = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with self.assertRaises(OSError):
                audit.load_image(path)
            self.assertTrue(ImageFile.LOAD_TRUNCATED_IMAGES)
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = permissive

    def test_mpo_decodes_all_pictures_but_returns_only_the_oriented_primary(self):
        path = self.root / "images" / "primary.jpg"
        exif = Image.Exif()
        exif[274] = 6
        with Image.new("RGB", (80, 60), "red") as primary, Image.new("RGB", (16, 12), "blue") as secondary:
            primary.save(path, format="MPO", save_all=True, append_images=[secondary], exif=exif)
        pixels, metadata = audit.load_image(path)
        with pixels:
            self.assertEqual(pixels.size, (60, 80))
            red, _, blue = pixels.getpixel((0, 0))
            self.assertGreater(red, 200)
            self.assertLess(blue, 20)
        self.assertEqual(metadata["detected_format"], "MPO")
        self.assertEqual(metadata["frame_count"], 2)
        self.assertEqual(metadata["additional_frames_decoded"], 1)
        self.assertEqual(json.loads(metadata["frame_dimensions"])[1]["width"], 16)

    def test_corrupt_secondary_mpo_picture_is_not_silently_ignored(self):
        path = self.root / "images" / "broken_secondary.jpg"
        with Image.new("RGB", (80, 60), "red") as primary, Image.effect_noise((128, 128), 50) as secondary:
            primary.save(path, format="MPO", save_all=True, append_images=[secondary])
        path.write_bytes(path.read_bytes()[:-100])
        with Image.open(path) as image:
            image.load()  # The primary is intact.
        with self.assertRaises(OSError):
            audit.load_image(path)

    def test_animation_is_not_silently_treated_as_one_labeled_still(self):
        path = self.root / "images" / "animation.gif"
        with Image.new("RGB", (8, 8), "red") as first, Image.new("RGB", (8, 8), "blue") as second:
            first.save(path, save_all=True, append_images=[second], duration=100)
        with self.assertRaisesRegex(ValueError, "Unsupported image sequence"):
            audit.load_image(path)

    def test_end_to_end_accounts_for_failures_labels_sources_and_galleries(self):
        with Image.new("RGB", (80, 60), "red") as image:
            image.save(self.root / "images" / "good.jpg")
        with Image.new("L", (16, 16), 90) as image:
            image.save(self.root / "images" / "grey.png")
        (self.root / "images" / "broken.jpg").write_bytes(b"\xff\xd8\xffbroken")
        (self.root / "images" / "unreferenced.png").write_bytes(b"not an input")
        (self.root / "videos" / "clip.mp4").write_bytes(b"video is deliberately not decoded")
        rows = [self.row("good.jpg"), self.row("broken.jpg"), self.row("grey.png"),
                self.row("unused", filepath="/old/train/videos/clip.mp4", media_format="video")]
        self.manifests(rows)
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        result = audit.audit_images(self.root, preview_size=64)
        output = self.root / "cleaned" / "image_audit"
        self.assertEqual((result["images_checked"], result["images_passed"], result["images_failed"], result["videos_skipped"]), (3, 2, 1, 1))
        self.assertEqual(len(self.load_csv("image_validation.csv")), 3)
        self.assertEqual(self.load_csv("decode_failures.csv")[0]["source_record"], "2")
        self.assertEqual(self.load_csv("image_dataset_validated.csv"), [
            {**rows[i], "filepath": str(self.root / "images" / filename)}
            for i, filename in [(0, "good.jpg"), (2, "grey.png")]
        ])
        for name, data in before.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual(len(list((output / "previews").glob("*.png"))), 2)
        sheets = self.load_csv("contact_sheet_index.csv")
        covered = {int(record) for sheet in sheets if sheet["gallery"] == "by_casualty"
                   for record in sheet["source_records"].split(";")}
        self.assertEqual(covered, {1, 3})
        for sheet in sheets:
            with Image.open(output / sheet["filepath"]) as image:
                image.load()
        report = (output / "index.html").read_text()
        self.assertIn("case_&quot;&lt;001&gt;", report)
        self.assertNotIn('case_"<001>', report)
        self.assertTrue(result["input_manifests_unchanged"])
        self.assertEqual(result["input_image_hashes_verified_unchanged"], 3)

        # Repeat runs produce the same current artifacts, without scanning their own output as input.
        first = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in output.rglob("*") if path.is_file()}
        self.assertEqual(audit.audit_images(self.root, preview_size=64), result)
        self.assertEqual({str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in output.rglob("*") if path.is_file()}, first)

    def test_changed_cleaned_labels_are_rejected_before_writing_outputs(self):
        self.manifests([self.row("one.jpg")])
        manifest = self.root / "cleaned" / "train_dataset_cleaned.csv"
        manifest.write_text(manifest.read_text().replace(",wound,", ",no_injury,"))
        with self.assertRaisesRegex(ValueError, "cannot be traced"):
            audit.audit_images(self.root)
        self.assertFalse((self.root / "cleaned" / "image_audit").exists())

    def test_output_overlap_and_output_symlinks_are_rejected(self):
        self.manifests([self.row("one.jpg")])
        for output in [self.root, self.root / "cleaned", self.root / "images" / "audit", self.root / "videos"]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                audit.audit_images(self.root, output)
        output = self.root / "cleaned" / "image_audit"
        output.mkdir()
        (output / "previews").symlink_to(self.root / "images", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            audit.audit_images(self.root, output)

    def test_representatives_prioritize_independent_casualties(self):
        records = [{"casualty_id": "a", "id": i} for i in range(5)] + [{"casualty_id": "b", "id": 5}]
        self.assertEqual([row["id"] for row in audit.representative_sample(records, 3)], [0, 5, 1])


if __name__ == "__main__":
    unittest.main()
