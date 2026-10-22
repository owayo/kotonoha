"""Release 配布物の検証・辞書マージ・再現性を実際のアーカイブで検査する。"""

import csv
import hashlib
import io
import json
import struct
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
        (self.models / "accent_model_v66_split1.onnx").write_bytes(
            b"test model contents"
        )
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
                            **(
                                {"native_file": "accent.knm"}
                                if path.suffix == ".onnx"
                                else {}
                            ),
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
        return package_models(
            self.models, self.output, self.lock, self.base, "b" * 40, native=False
        )

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
        (self.models / "accent_model_v66_split1.onnx").touch()
        self.assertEqual(self.package().read_bytes(), first)

    def test_hsd_provenance_is_attached_and_included_in_the_archive(self):
        """検証した HSD の記録を単独添付とアーカイブ内で同じバイト列にする。"""
        record = (
            Path(__file__).resolve().parents[2] / ".github/hasami-dictionaries.json"
        )
        archive_path = package_models(
            self.models,
            self.output,
            self.lock,
            self.base,
            "b" * 40,
            native=False,
            hasami_release=record,
        )
        attached = (self.output / "hasami-dictionaries.json").read_bytes()
        with tarfile.open(archive_path) as archive:
            packed = archive.extractfile(
                "kotonoha-models/hasami-dictionaries.json"
            ).read()
            self.assertEqual(attached, packed)
            manifest = json.load(archive.extractfile("kotonoha-models/models.json"))
            entry = next(
                e for e in manifest["files"] if e["file"] == "hasami-dictionaries.json"
            )
            self.assertEqual(entry["sha256"], hashlib.sha256(attached).hexdigest())

    def test_native_archive_replaces_onnx_and_checks_conversion_manifest(self):
        source = json.loads(self.lock.read_text(encoding="utf-8"))
        original = next(
            entry
            for entry in source["files"]
            if entry["file"] == "accent_model_v66_split1.onnx"
        )
        metadata = json.dumps({"source_sha256": original["sha256"]}).encode()
        native = self.models / "accent.knm"
        native.write_bytes(
            struct.pack("<8sIIQQ", b"KOTONNM\0", 1, len(metadata), 1, 0)
            + metadata
            + b"x"
        )
        (self.models / "knm-manifest.json").write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "file": native.name,
                            "size": native.stat().st_size,
                            "sha256": file_digest(native),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        archive = package_models(
            self.models, self.output, self.lock, self.base, "b" * 40, native=True
        )
        self.assertEqual(archive.name, "kotonoha-models.tar.gz")
        with tarfile.open(archive) as bundle:
            names = bundle.getnames()
            self.assertIn("kotonoha-models/accent.knm", names)
            self.assertNotIn("kotonoha-models/accent_model_v66_split1.onnx", names)
        original_archive = archive.read_bytes()
        bundle_dir = self.root / "bundles/base"
        bundle_dir.mkdir(parents=True)
        for name in ("accent.knm", "accent_dict_jsut.csv", "knm-manifest.json"):
            (self.models / name).rename(bundle_dir / name)
        archive = package_models(
            self.models,
            self.output,
            self.lock,
            self.base,
            "b" * 40,
            native=True,
            bundle_dir=bundle_dir,
        )
        self.assertEqual(archive.read_bytes(), original_archive)
        native = bundle_dir / "accent.knm"
        native.write_bytes(native.read_bytes()[:-1] + b"y")
        with self.assertRaisesRegex(ValueError, "conversion manifest"):
            package_models(
                self.models,
                self.output,
                self.lock,
                self.base,
                "b" * 40,
                native=True,
                bundle_dir=bundle_dir,
            )

    def test_missing_corrupt_or_unresolved_lfs_model_is_rejected(self):
        path = self.models / "accent_model_v66_split1.onnx"
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
