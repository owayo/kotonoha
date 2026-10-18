"""HSD 更新の検出と、破損・添付途中・通信障害での停止を検査する。"""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hasami_release as hsd


def snapshot():
    """独立に作成した小さい HSD 目録を作る。

    Returns:
        検証に使う公開記録の形式を持つ辞書。

    """
    return {
        "schema_version": 1,
        "repository": hsd.UPSTREAM,
        "tag": "v26.10.100",
        "catalog_sha256": "a" * 64,
        "catalog": {
            "hasami_version": "26.10.100",
            "format_version": 5,
            "recommended": "ipadic-neologd-sudachi",
            "dictionaries": [
                {
                    "name": name,
                    "file": name + ".hsd",
                    "size": 3,
                    "sha256": hashlib.sha256(b"abc").hexdigest(),
                }
                for name in sorted(hsd.DICTIONARIES)
            ],
        },
    }


class HasamiReleaseTests(unittest.TestCase):
    """自動 Release の判定と、固定した公開物の検証。"""

    def test_changes_use_hsd_content_and_not_just_release_tags(self):
        """同じ辞書ならタグだけの更新では再公開せず、どの辞書の変更も検出する。"""
        old = snapshot()
        latest = copy.deepcopy(old)
        latest["tag"] = "v26.10.101"
        latest["catalog"]["hasami_version"] = "26.10.101"
        self.assertFalse(hsd.rebuild_required(latest, old))
        self.assertTrue(hsd.rebuild_required(latest, None))
        for index in range(3):
            changed = copy.deepcopy(latest)
            changed["catalog"]["dictionaries"][index]["sha256"] = "b" * 64
            self.assertTrue(hsd.rebuild_required(changed, old))

    def test_invalid_release_identity_and_paths_are_rejected(self):
        """任意のリポジトリ・版不一致・パス脱出・不正ハッシュを受け付けない。"""
        for key, value in [("repository", "other/repository"), ("tag", "../main")]:
            record = snapshot()
            record[key] = value
            with self.assertRaises(ValueError):
                hsd.validate_snapshot(record)
        record = snapshot()
        record["catalog"]["hasami_version"] = "26.10.101"
        with self.assertRaises(ValueError):
            hsd.validate_snapshot(record)
        for field, value in [
            ("file", "../outside.hsd"),
            ("sha256", "wrong"),
            ("size", -1),
        ]:
            record = snapshot()
            record["catalog"]["dictionaries"][0][field] = value
            with self.assertRaises(ValueError):
                hsd.validate_snapshot(record)

    def test_metadata_corruption_is_rejected(self):
        """API が返した添付 SHA-256 と一致しない記録では Release を判断しない。"""
        asset = dict(id=1, size=3, state="uploaded", digest="sha256:" + "a" * 64)
        result = subprocess.CompletedProcess([], 0, stdout=b"abc")
        with patch.object(hsd.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                hsd.metadata_asset("example/models", asset)

    def test_upstream_catalog_must_match_complete_stable_release_assets(self):
        """未公開・プレリリース・不一致の HSD を自動 Release の入力にしない。"""
        record = snapshot()
        data = json.dumps(record["catalog"]).encode()
        release = {
            "tag_name": record["tag"],
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "dictionaries.json"}]
            + [
                dict(
                    name=e["file"],
                    size=e["size"],
                    state="uploaded",
                    digest="sha256:" + e["sha256"],
                )
                for e in record["catalog"]["dictionaries"]
            ],
        }
        with (
            patch.object(hsd, "gh_json", return_value=release),
            patch.object(hsd, "metadata_asset", return_value=data),
        ):
            self.assertEqual(hsd.published_snapshot()["catalog"], record["catalog"])
            release["assets"][1]["state"] = "uploading"
            with self.assertRaisesRegex(ValueError, "differs"):
                hsd.published_snapshot()
            release["assets"][1]["state"] = "uploaded"
            release["assets"][1]["size"] += 1
            with self.assertRaisesRegex(ValueError, "differs"):
                hsd.published_snapshot()
            release["draft"] = True
            with self.assertRaisesRegex(ValueError, "stable release"):
                hsd.published_snapshot()
            release["draft"] = False
            release["prerelease"] = True
            with self.assertRaisesRegex(ValueError, "stable release"):
                hsd.published_snapshot()

    def test_missing_release_is_distinct_from_api_failure(self):
        """初回の 404 は許容し、権限・通信エラーは更新とみなさず停止する。"""
        for stderr in (b"HTTP 404", b"HTTP 403", b"connection failed"):
            result = subprocess.CompletedProcess([], 1, stderr=stderr)
            with patch.object(hsd.subprocess, "run", return_value=result):
                if stderr == b"HTTP 404":
                    self.assertIsNone(
                        hsd.gh_json(
                            "repos/example/models/releases/latest", missing_ok=True
                        )
                    )
                else:
                    with self.assertRaises(ValueError):
                        hsd.gh_json(
                            "repos/example/models/releases/latest", missing_ok=True
                        )

    def test_only_fully_uploaded_model_release_counts_as_processed(self):
        """コミット済みの状態ではなく、全添付が揃った公開 Release だけを基準にする。"""
        release = {
            "assets": [
                dict(name=name, state="uploaded") for name in hsd.MODEL_RELEASE_ASSETS
            ]
        }
        with (
            patch.object(hsd, "gh_json", return_value=release),
            patch.object(
                hsd, "metadata_asset", return_value=json.dumps(snapshot()).encode()
            ),
        ):
            self.assertEqual(hsd.last_model_snapshot("example/models"), snapshot())
            release["assets"].pop()
            self.assertIsNone(hsd.last_model_snapshot("example/models"))

    def test_hsd_download_verification_reuse_and_existing_file_protection(self):
        """検証できた HSD だけを配置し、再利用と破損時の保護を確認する。"""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def download(command, **_kwargs):
                Path(command[-1]).write_bytes(b"abc")

            with patch.object(hsd.subprocess, "run", side_effect=download) as fetch:
                target = hsd.download_dictionary(snapshot(), output)
                self.assertEqual(target.read_bytes(), b"abc")
                self.assertEqual(hsd.download_dictionary(snapshot(), output), target)
                self.assertEqual(fetch.call_count, 1)
                target.write_bytes(b"old")
                with self.assertRaisesRegex(ValueError, "Existing HSD"):
                    hsd.download_dictionary(snapshot(), output)
                self.assertEqual(target.read_bytes(), b"old")
            target.unlink()

            def corrupt(command, **_kwargs):
                Path(command[-1]).write_bytes(b"bad")

            with patch.object(hsd.subprocess, "run", side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    hsd.download_dictionary(snapshot(), output)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
