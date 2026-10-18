"""採番の月境界と、Rust/Python の版更新で依存を変えないことを検査する。"""

import sys
import tempfile
import tomllib
import unittest
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_release import JST, next_version, prepare_release


class PrepareReleaseTests(unittest.TestCase):
    """公開版の採番とメタデータ更新の回帰テスト。"""

    def test_initial_release_month_rollover_and_legacy_year(self):
        """初回・月変更・旧形式の年から正しい連番を決める。"""
        now = datetime(2026, 10, 18, tzinfo=JST)
        for current, expected in [
            ("0.1.0", "26.10.100"),
            ("26.10.100", "26.10.101"),
            ("2026.10.107", "26.10.108"),
            ("26.9.108", "26.10.100"),
            ("25.10.108", "26.10.100"),
        ]:
            with self.subTest(current=current):
                self.assertEqual(next_version(current, now), expected)

    def test_month_and_year_rollover_use_jst(self):
        """UTC では前日でも JST の新しい年月で採番する。"""
        for current, now, expected in [
            ("26.9.108", datetime(2026, 9, 30, 15, tzinfo=UTC), "26.10.100"),
            ("26.12.108", datetime(2026, 12, 31, 15, tzinfo=UTC), "27.1.100"),
        ]:
            with self.subTest(now=now):
                self.assertEqual(next_version(current, now), expected)

    def test_versions_are_updated_without_changing_dependencies(self):
        """公開版だけをそろえ、入力不備ではファイルを更新しない。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cargo = root / "Cargo.toml"
            cargo.write_text(
                '[workspace]\nmembers = ["kotonoha"]\n\n'
                '[workspace.package]\nversion = "0.1.0"\nedition = "2024"\n\n'
                '[workspace.dependencies.example]\nversion = "1.2.3"\n',
                encoding="utf-8",
            )
            python = root / "kotonoha-python/pyproject.toml"
            python.parent.mkdir()
            python.write_text(
                '[project]\nname = "kotonoha"\nversion = "0.1.0"\n\n'
                '[tool.example]\nversion = "4.5.6"\n',
                encoding="utf-8",
            )
            version = prepare_release(root, datetime(2026, 10, 18, tzinfo=JST))
            self.assertEqual(version, "26.10.100")
            rust = tomllib.loads(cargo.read_text(encoding="utf-8"))
            py = tomllib.loads(python.read_text(encoding="utf-8"))
            self.assertEqual(rust["workspace"]["package"]["version"], version)
            self.assertEqual(py["project"]["version"], version)
            self.assertEqual(
                rust["workspace"]["dependencies"]["example"]["version"], "1.2.3"
            )
            self.assertEqual(py["tool"]["example"]["version"], "4.5.6")

            # 不正な Python メタデータでは Cargo.toml も更新しない。
            before = cargo.read_bytes()
            python.write_text('[project]\nname = "kotonoha"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected one version"):
                prepare_release(root, datetime(2026, 10, 18, tzinfo=JST))
            self.assertEqual(cargo.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
