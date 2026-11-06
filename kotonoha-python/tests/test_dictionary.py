"""辞書の取得・検証の Python API を検査する。

利用側が PATH の hasami CLI に頼らず、組み込みの hasami と同じ版の辞書を用意できることを
確かめる。通信はローカルの HTTP サーバー（目録と辞書のミラー）だけに向ける。
"""

import functools
import hashlib
import http.server
import json
import os
import subprocess
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import kotonoha


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "kotonoha" / "tests" / "fixtures" / "quantities"
NAME = "quantities"


def build_dictionary(output: Path) -> None:
    """数量のテストと同じ辞書ソースから、現行の CLI で小さな HSD を作る。

    Raises:
        RuntimeError: CLI によるテスト辞書の構築が失敗した場合。

    """
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "-p",
            "kotonoha-cli",
            "--",
            "build-dict",
            "--csv-dir",
            str(FIXTURE),
            "--matrix",
            str(FIXTURE / "matrix.def"),
            "--unk",
            str(FIXTURE / "unk.def"),
            "--char-def",
            str(FIXTURE / "char.def"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """テストの出力を汚さないよう、アクセスログを出さない。"""

    def log_message(self, format, *args):  # noqa: A002
        """何も出力しない。"""


class Mirror:
    """一時フォルダーを配る HTTP サーバー（hasami の目録と辞書のミラー）。"""

    def __init__(self, root: Path) -> None:
        """`root` の中身を 127.0.0.1 の空いたポートで配り始める。"""
        handler = functools.partial(_QuietHandler, directory=str(root))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        """サーバーを止める。2 回目以降は何もしない。"""
        if not self.thread.is_alive():
            return
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def write_catalog(root: Path, dictionary: Path, **overrides) -> None:
    """`dictionary` だけを載せた目録 (dictionaries.json) を書く。"""
    data = dictionary.read_bytes()
    catalog = {
        "hasami_version": kotonoha.hasami_version(),
        "format_version": kotonoha.dictionary_format_version(),
        "recommended": NAME,
        "dictionaries": [
            {
                "name": NAME,
                "file": dictionary.name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    catalog.update(overrides)
    (root / "dictionaries.json").write_text(json.dumps(catalog), encoding="utf-8")


class DictionaryVersionTests(unittest.TestCase):
    """組み込みの hasami の版と、辞書の置き場所を返す。"""

    def test_versions_describe_the_embedded_hasami(self):
        """版は x.y.z の文字列、辞書の形式の版は 6 以上の整数。"""
        self.assertRegex(kotonoha.hasami_version(), r"^\d+\.\d+\.\d+$")
        self.assertIsInstance(kotonoha.dictionary_format_version(), int)
        self.assertGreaterEqual(kotonoha.dictionary_format_version(), 6)

    def test_dictionary_dir_uses_shared_override(self):
        """HASAMI_DATA_DIR を最優先に使い、ディレクトリは作らない。"""
        with TemporaryDirectory() as temp:
            target = Path(temp) / "not-created"
            with patch.dict(os.environ, {"HASAMI_DATA_DIR": str(target)}):
                self.assertEqual(kotonoha.dictionary_dir(), target)
            self.assertFalse(target.exists())

    def test_dictionary_error_is_a_runtime_error(self):
        """既存の RuntimeError を捕まえるコードでも扱える。"""
        self.assertTrue(issubclass(kotonoha.DictionaryError, RuntimeError))


class DictionaryApiTests(unittest.TestCase):
    """実際に構築した辞書で、検証と取得の結果を確かめる。"""

    @classmethod
    def setUpClass(cls):
        """テスト辞書を 1 度だけ構築する。"""
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.source = Path(temp.name) / f"{NAME}.hsd"
        build_dictionary(cls.source)

    def setUp(self):
        """辞書を配るミラーと、取得先の空のフォルダーを用意する。"""
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.mirror_root = Path(temp.name) / "mirror"
        self.mirror_root.mkdir()
        (self.mirror_root / self.source.name).write_bytes(self.source.read_bytes())
        self.target = Path(temp.name) / "shared"
        self.mirror = Mirror(self.mirror_root)
        self.addCleanup(self.mirror.close)

    def download(self, **kwargs):
        """ミラーから取得する。

        Returns:
            Path: 置いた辞書のパス。

        """
        return kotonoha.download_dictionary(
            self.target, base_url=self.mirror.url, **kwargs
        )

    def test_built_dictionary_is_valid(self):
        """組み込みの hasami で読める辞書は例外を出さない。"""
        self.assertIsNone(kotonoha.validate_dictionary(self.source))

    def test_missing_file_is_file_not_found(self):
        """無いファイルは FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            kotonoha.validate_dictionary(self.target / "missing.hsd")

    def test_non_dictionary_is_corrupt(self):
        """HSD でないファイルは code が corrupt。"""
        self.target.mkdir()
        path = self.target / "not.hsd"
        path.write_bytes(b"not a dictionary" * 8)
        with self.assertRaises(kotonoha.DictionaryError) as error:
            kotonoha.validate_dictionary(path)
        self.assertEqual(error.exception.code, "corrupt")
        self.assertEqual(Path(error.exception.path), path)

    def test_old_format_is_reported_as_format(self):
        """先頭の版が古い辞書 (v5) は code が format で、版がメッセージに出る。"""
        self.target.mkdir()
        path = self.target / "old.hsd"
        path.write_bytes(b"HSMDICT\0" + (5).to_bytes(4, "little") + bytes(52))
        with self.assertRaises(kotonoha.DictionaryError) as error:
            kotonoha.validate_dictionary(path)
        self.assertEqual(error.exception.code, "format")
        self.assertIn("v5", str(error.exception))

    def test_download_places_the_recommended_dictionary(self):
        """名前を省くと目録の推奨を取り、読める辞書を置き場所に置く。"""
        write_catalog(self.mirror_root, self.source)
        path = self.download()
        self.assertEqual(path, self.target / self.source.name)
        self.assertEqual(path.read_bytes(), self.source.read_bytes())
        self.assertIsNone(kotonoha.validate_dictionary(path))

    def test_download_keeps_a_matching_file(self):
        """正しいファイルがすでにあれば、取り直さずに同じパスを返す。"""
        write_catalog(self.mirror_root, self.source)
        first = self.download(name=NAME)
        mtime = first.stat().st_mtime_ns
        self.assertEqual(self.download(name=NAME), first)
        self.assertEqual(first.stat().st_mtime_ns, mtime)

    def test_different_file_is_kept_unless_forced(self):
        """中身の違うファイルは force なしでは変えず、force で置き換える。"""
        write_catalog(self.mirror_root, self.source)
        self.target.mkdir()
        existing = self.target / self.source.name
        existing.write_bytes(b"HSMDICT\0" + (5).to_bytes(4, "little") + bytes(52))
        with self.assertRaises(kotonoha.DictionaryError) as error:
            self.download()
        self.assertEqual(error.exception.code, "conflict")
        self.assertEqual(Path(error.exception.path), existing)
        self.assertEqual(existing.read_bytes()[8:12], (5).to_bytes(4, "little"))

        self.assertEqual(self.download(force=True), existing)
        self.assertEqual(existing.read_bytes(), self.source.read_bytes())

    def test_broken_download_keeps_the_existing_dictionary(self):
        """取得した中身が目録と合わなければ失敗し、既存の辞書を残す。"""
        write_catalog(self.mirror_root, self.source)
        self.download()
        (self.mirror_root / self.source.name).write_bytes(b"broken" * 8)
        write_catalog(
            self.mirror_root,
            self.source,
            dictionaries=[
                {
                    "name": NAME,
                    "file": self.source.name,
                    "size": 48,
                    "sha256": hashlib.sha256(b"broken" * 8).hexdigest(),
                }
            ],
        )
        with self.assertRaises(kotonoha.DictionaryError) as error:
            self.download(force=True)
        self.assertEqual(error.exception.code, "corrupt")
        self.assertEqual(
            (self.target / self.source.name).read_bytes(), self.source.read_bytes()
        )

    def test_catalog_for_another_format_is_rejected(self):
        """組み込みの hasami が読めない形式の目録は code が format。"""
        write_catalog(self.mirror_root, self.source, format_version=5)
        with self.assertRaises(kotonoha.DictionaryError) as error:
            self.download()
        self.assertEqual(error.exception.code, "format")
        self.assertFalse((self.target / self.source.name).exists())

    def test_catalog_for_another_hasami_is_rejected(self):
        """ミラーでも、組み込みの hasami と別の版の目録は使わない。"""
        write_catalog(self.mirror_root, self.source, hasami_version="0.0.1")
        with self.assertRaises(kotonoha.DictionaryError) as error:
            self.download()
        self.assertEqual(error.exception.code, "download")
        self.assertIn("0.0.1", str(error.exception))

    def test_unknown_name_is_a_value_error(self):
        """目録に無い名前は ValueError。"""
        write_catalog(self.mirror_root, self.source)
        with self.assertRaises(ValueError):
            self.download(name="missing")

    def test_unreachable_mirror_is_a_download_error(self):
        """目録を取れないときは code が download。"""
        url = self.mirror.url
        self.mirror.close()
        with self.assertRaises(kotonoha.DictionaryError) as error:
            kotonoha.download_dictionary(self.target, base_url=url)
        self.assertEqual(error.exception.code, "download")


if __name__ == "__main__":
    unittest.main()
