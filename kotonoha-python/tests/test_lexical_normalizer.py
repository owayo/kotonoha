"""構築辞書から英字語の候補と厳密な音素列を公開Python APIで取得する。"""

import csv
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import kotonoha


class LexicalNormalizerTests(unittest.TestCase):
    """語義を推測せず、全文の語境界と辞書内の異音候補を照合する。"""

    @classmethod
    def setUpClass(cls):
        """一時ディレクトリに独立した小辞書を構築する。"""
        root = Path(__file__).resolve().parents[2]
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        source = directory / "source"
        source.mkdir()
        fixture = root / "kotonoha" / "tests" / "fixtures" / "quantities"
        for name in ("matrix.def", "char.def", "unk.def"):
            shutil.copyfile(fixture / name, source / name)
        with (source / "lexical.csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            for surface, pronunciation, subtype in [
                ("前", "マエ", "一般"),
                ("後", "アト", "一般"),
                ("River Cloud", "リバークラウド", "一般"),
                ("river cloud", "リバアクラウド", "一般"),
                ("Maple Harbor", "メープルハーバー", "一般"),
                ("maple harbor", "メープルハーバー", "人名"),
            ]:
                writer.writerow(
                    [
                        surface,
                        0,
                        0,
                        0,
                        "名詞",
                        "固有名詞",
                        subtype,
                        "一般" if subtype == "人名" else "*",
                        "*",
                        "*",
                        surface,
                        pronunciation,
                        pronunciation,
                    ]
                )
        path = directory / "lexical.hsd"
        run = subprocess.run(
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
                str(path),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
        cls.normalizer = kotonoha.DictionaryNormalizer(str(path), symbols="")
        cls.addClassCleanup(delattr, cls, "normalizer")

    def test_whole_spans_are_utf8_tuples_without_rewriting_input(self):
        """繰り返す語の各範囲を保ち、識別子の一部と異音候補は辞退する。"""
        text = "前River Cloud後River Cloud"
        selected = self.normalizer.token_pronunciations(text)
        self.assertEqual(
            self.normalizer.lexical_pronunciation_spans(text),
            [(3, 14, "リバークラウド"), (17, 28, "リバークラウド")],
        )
        self.assertEqual(self.normalizer.normalize(text), text)
        self.assertEqual(self.normalizer.token_pronunciations(text), selected)
        for text in [
            "xRiver Cloud",
            "River Cloud2",
            "path/River Cloud",
            "Maple Harbor",
            "",
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.normalizer.lexical_pronunciation_spans(text), [])

    def test_phoneme_api_preserves_vowel_length_and_rejects_silent_loss(self):
        """音素列だけを比較し、未知文字の無視やアクセントの同値化は行わない。"""
        phones = kotonoha.DictionaryNormalizer.pronunciation_phonemes
        self.assertEqual(phones("アール"), ["a", "a", "r", "u"])
        self.assertEqual(phones("アー"), phones("アア"))
        self.assertNotEqual(phones("ア"), phones("アー"))
        for value in ["", "アA", "ーア", "ンー", "ッー", "ア〇"]:
            with self.subTest(value=value):
                self.assertIsNone(phones(value))


if __name__ == "__main__":
    unittest.main()
