"""Release 配布物の検証・辞書マージ・再現性を実際のアーカイブで検査する。"""

import csv
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_models import BUNDLE_NAME, file_digest, merge_dictionaries, package_models


class PackageModelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.models.mkdir()
        (self.models / "test.onnx").write_bytes(b"test model contents")
        (self.models / "accent_dict_jsut.csv").write_text(
            "lemma,reading,accent_type\n猫,ネコ,2\n猫,ビョウ,1\n", encoding="utf-8"
        )
        self.base = self.root / "base.csv"
        self.base.write_text(
            "lemma,reading,accent_type\n猫,ネコ,0\n犬,イヌ,1\n犬,イヌ,2\n",
            encoding="utf-8",
        )
        self.lock = self.root / "lock.json"
        self.lock.write_text(
            json.dumps(
                {
                    "repository": "test/models",
                    "revision": "a" * 40,
                    "files": [
                        {
                            "file": path.name,
                            "size": path.stat().st_size,
                            "sha256": file_digest(path),
                        }
                        for path in sorted(self.models.iterdir())
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.output = self.root / "output"

    def package(self):
        return package_models(self.models, self.output, self.lock, self.base, "b" * 40)

    def test_archive_loadable_dictionary_provenance_and_checksums(self):
        archive_path = self.package()
        expected_hash = (self.output / f"{BUNDLE_NAME}.sha256").read_text().split()[0]
        self.assertEqual(file_digest(archive_path), expected_hash)
        with tarfile.open(archive_path) as archive:
            contents = {
                member.name.removeprefix("kotonoha-models/"): archive.extractfile(
                    member
                ).read()
                for member in archive.getmembers()
            }
        manifest = json.loads(contents.pop("models.json"))
        self.assertEqual(manifest["kotonoha_revision"], "b" * 40)
        self.assertEqual(manifest["source"]["revision"], "a" * 40)
        self.assertEqual(
            {entry["file"] for entry in manifest["files"]}, contents.keys()
        )
        for entry in manifest["files"]:
            data = contents[entry["file"]]
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            self.assertEqual(len(data), entry["size"])
        rows = csv.DictReader(io.StringIO(contents["accent_dict.csv"].decode()))
        accents = {
            (row["lemma"], row["reading"]): int(row["accent_type"]) for row in rows
        }
        self.assertEqual(
            accents, {("猫", "ネコ"): 2, ("猫", "ビョウ"): 1, ("犬", "イヌ"): 1}
        )

    def test_repackaging_is_reproducible(self):
        first = self.package().read_bytes()
        (self.models / "test.onnx").touch()
        self.assertEqual(self.package().read_bytes(), first)

    def test_missing_corrupt_or_unresolved_lfs_model_is_rejected(self):
        path = self.models / "test.onnx"
        original = path.read_bytes()
        for data, message in [
            (None, "Missing model bundle input"),
            (b"x" * len(original), "differs from"),
            (b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
        ]:
            with self.subTest(message=message):
                if data is None:
                    path.unlink()
                else:
                    path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, message):
                    self.package()
                self.assertFalse(self.output.exists())
                path.write_bytes(original)

    def test_invalid_dictionary_accent_is_not_replaced_with_zero(self):
        for accent in ("-1", "256", "broken", "1.5", ""):
            with self.subTest(accent=accent):
                self.base.write_text(
                    f"lemma,reading,accent_type\n猫,ネコ,{accent}\n", encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    merge_dictionaries([self.base])


if __name__ == "__main__":
    unittest.main()
