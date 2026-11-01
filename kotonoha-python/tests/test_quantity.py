"""共通の構築辞書で数量補正が Python のテキスト解析にも届くことを確認する。"""

import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import kotonoha


class QuantityApiTests(unittest.TestCase):
    """手作りトークンを入力せず、実際の Analyzer から数量を解析する。"""

    @classmethod
    def setUpClass(cls):
        """Rust・CLI と共通の辞書ソースを現行 CLI で HSD へ変換する。

        Raises:
            RuntimeError: CLI によるテスト辞書の構築が失敗した場合。

        """
        root = Path(__file__).resolve().parents[2]
        source = root / "kotonoha" / "tests" / "fixtures" / "quantities"
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        dictionary = Path(temp.name) / "quantity.hsd"
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
                str(source),
                "--matrix",
                str(source / "matrix.def"),
                "--unk",
                str(source / "unk.def"),
                "--char-def",
                str(source / "char.def"),
                "--output",
                str(dictionary),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        with patch.dict(os.environ):
            os.environ["XDG_DATA_HOME"] = temp.name
            for key in (
                "KOTONOHA_MODEL_PATH",
                "KOTONOHA_MODEL_BUNDLE",
                "KOTONOHA_MODEL_VARIANT",
            ):
                os.environ.pop(key, None)
            cls.engine = kotonoha.KotonohaEngine(dict_path=str(dictionary))
        cls.addClassCleanup(delattr, cls, "engine")
        cls.normalizer = kotonoha.DictionaryNormalizer(str(dictionary), symbols="")
        cls.addClassCleanup(delattr, cls, "normalizer")
        cls.cases = json.loads((source / "cases.json").read_text(encoding="utf-8"))

    def test_context_spans_use_utf8_byte_positions_and_complete_quantities(self):
        """全文の文脈とUTF-8位置をPythonのタプルへ保持する。"""
        text = "前1階と6冊"
        self.assertEqual(
            self.normalizer.quantity_spans(text),
            [(3, 7, "イッカイ"), (10, 14, "ロクサツ")],
        )
        known = self.normalizer.known_word_spans(text)
        for span in [(3, 4), (4, 7), (10, 11), (11, 14)]:
            self.assertIn(span, known)
        encoded = text.encode("utf-8")
        for start, end in known:
            self.assertTrue(encoded[start:end].decode("utf-8"))
        selected = self.normalizer.token_pronunciations(text)
        self.assertIn((3, 4, "イッ"), selected)
        self.assertIn((4, 7, "カイ"), selected)
        self.assertEqual(self.normalizer.token_pronunciations(""), [])
        for rejected in ["01階", "1 階", "1.1.1階", ""]:
            with self.subTest(rejected=rejected):
                self.assertEqual(self.normalizer.quantity_spans(rejected), [])

    def test_numeric_identifiers_keep_digits_and_reject_dates(self):
        """郵便・電話形を桁読みし、日付や不完全な範囲を対象外にする。"""
        for text, expected in [
            ("123-4567", "イチニサンヨンゴロクナナ"),
            ("03-1234-5678", "ゼロサンイチニサンヨンゴロクナナハチ"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.normalizer.normalize_identifier(text), expected)
        for text in ["2026-01-05", "12-34", "1234567"]:
            with self.subTest(text=text):
                self.assertIsNone(self.normalizer.normalize_identifier(text))

    def test_quantity_readings_reach_nodes_phones_and_labels(self):
        """促音化する数量と、六・小数・識別子・人名の対照条件を照合する。"""
        for text, reading, pronunciation in self.cases:
            with self.subTest(text=text):
                nodes = self.engine.text_to_analyze(text)
                self.assertEqual("".join(node.reading for node in nodes), reading)
                self.assertEqual(
                    "".join(node.pronunciation for node in nodes), pronunciation
                )
                canonical = SimpleNamespace(
                    surface="語",
                    pos="名詞",
                    reading=pronunciation,
                    pronunciation=pronunciation,
                )
                expected = [phone for phone, _ in self.engine.phone_tones([canonical])]
                phones = self.engine.text_to_phone_tones(text)
                self.assertEqual([phone for phone, _ in phones], expected)
                self.assertEqual(len(self.engine.text_to_labels(text)), len(phones))
                self.assertTrue(self.engine.text_to_prosody_symbols(text))


if __name__ == "__main__":
    unittest.main()
