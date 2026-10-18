"""Python 入力の既定値と公開 API の変換結果を検査する。"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import kotonoha


class EngineApiTests(unittest.TestCase):
    """辞書やモデルなしで入力変換と出力の契約を確認する。"""

    def setUp(self):
        """外部モデルの設定を除いてエンジンを用意する。"""
        # 手元のモデル設定が辞書不要のテストに混ざらないようにする。
        model_data = TemporaryDirectory()
        self.addCleanup(model_data.cleanup)
        with patch.dict(os.environ):
            os.environ["XDG_DATA_HOME"] = model_data.name
            for key in (
                "KOTONOHA_MODEL_PATH",
                "KOTONOHA_MODEL_BUNDLE",
                "KOTONOHA_MODEL_VARIANT",
            ):
                os.environ.pop(key, None)
            self.engine = kotonoha.KotonohaEngine()

    def test_missing_and_none_fields_match_explicit_defaults(self):
        """省略・None・明示した既定値で出力が一致し、入力を変更しない。"""
        required = dict(surface="猫", pos="名詞", reading="ネコ")
        optional = dict(
            pos_detail1="*",
            pos_detail2="*",
            pos_detail3="*",
            ctype="*",
            cform="*",
            lemma="猫",
            pronunciation="ネコ",
        )
        explicit = SimpleNamespace(**required, **optional)
        for token in (
            SimpleNamespace(**required),
            SimpleNamespace(**required, **dict.fromkeys(optional)),
        ):
            original = vars(token).copy()
            for method in (
                "make_label",
                "phone_tones",
                "phone_tones_with_punct",
                "prosody_symbols",
                "predict_accent_types",
            ):
                with self.subTest(method=method, token=token):
                    call = getattr(self.engine, method)
                    self.assertEqual(call([token]), call([explicit]))
            node = self.engine.analyze([token])[0]
            self.assertEqual(
                (node.surface, node.reading, node.pronunciation), ("猫", "ネコ", "ネコ")
            )
            self.assertEqual(node.mora_count, 2)
            self.assertEqual(vars(token), original)

    def test_explicit_pronunciation_is_preserved(self):
        """読みと異なる発音を指定した場合も、発音が採用される。"""
        token = SimpleNamespace(
            surface="今日", pos="名詞", reading="キョウ", pronunciation="キョー"
        )
        node = self.engine.analyze([token])[0]
        self.assertEqual(node.reading, "キョウ")
        self.assertEqual(node.pronunciation, "キョオ")
        self.assertEqual(node.mora_count, 2)
        self.assertEqual(token.pronunciation, "キョー")

    def test_predictions_are_integer_lists_including_empty_input(self):
        """予測結果は bytes ではなく、空入力でも整数のリストで返す。"""
        self.assertEqual(self.engine.predict_accent_types([]), [])
        tokens = [
            SimpleNamespace(surface="猫", pos="名詞", reading="ネコ"),
            SimpleNamespace(surface="犬", pos="名詞", reading="イヌ"),
        ]
        predictions = self.engine.predict_accent_types(tokens)
        self.assertEqual(predictions, [0, 0])
        self.assertTrue(all(type(value) is int for value in predictions))

    def test_conjugation_distinguishes_dictionary_form_and_volitional_u(self):
        """活用形を渡すと基本形の「う」を長音化せず、意志形だけを長音化する。"""
        for surface, lemma, cform, reading, expected in (
            ("まよう", "迷う", "終止形-一般", "マヨウ", "マヨウ"),
            ("映ろう", "映ろう", "基本形", "ウツロウ", "ウツロウ"),
            ("つくろう", "つくろう", "意志推量形", "ツクロウ", "ツクロオ"),
        ):
            with self.subTest(surface=surface, cform=cform):
                token = SimpleNamespace(
                    surface=surface,
                    pos="動詞",
                    reading=reading,
                    lemma=lemma,
                    cform=cform,
                )
                node = self.engine.analyze([token])[0]
                self.assertEqual(node.pronunciation, expected)
                self.assertEqual(
                    [phone for phone, _ in self.engine.phone_tones([token])][-2],
                    "o" if "意志推量" in cform else "u",
                )

    def test_long_pronunciation_keeps_mora_count(self):
        """256 モーラでも Python API の個数と出力位置を保持する。"""
        reading = "カ" * 256
        token = SimpleNamespace(surface=reading, pos="名詞", reading=reading)
        self.assertEqual(self.engine.analyze([token])[0].mora_count, 256)
        self.assertEqual(len(self.engine.phone_tones([token])), 514)
        self.assertEqual(self.engine.prosody_symbols([token]).count("カ"), 256)

    def test_punctuation_output_is_distinct(self):
        """句読点を保持する API の出力を通常の PhoneTone と区別する。"""
        tokens = [
            SimpleNamespace(surface="猫", pos="名詞", reading="ネコ"),
            SimpleNamespace(surface="、", pos="記号", reading="、"),
            SimpleNamespace(surface="犬", pos="名詞", reading="イヌ"),
        ]
        plain = [phone for phone, _ in self.engine.phone_tones(tokens)]
        punct = [phone for phone, _ in self.engine.phone_tones_with_punct(tokens)]
        self.assertNotIn("、", plain)
        self.assertIn("、", punct)

    def test_invalid_tokens_raise(self):
        """必須属性の欠落と辞書形式の入力を受け付けない。"""
        for token in (
            SimpleNamespace(surface="猫", pos="名詞"),
            {"surface": "猫", "pos": "名詞", "reading": "ネコ"},
        ):
            with self.subTest(token=token):
                with self.assertRaises((TypeError, AttributeError)):
                    self.engine.make_label([token])

    def test_text_methods_report_missing_dictionary(self):
        """辞書未設定のテキスト解析が RuntimeError を返す。"""
        for method in (
            "text_to_labels",
            "text_to_phone_tones",
            "text_to_phone_tones_with_punct",
            "text_to_prosody_symbols",
            "text_to_analyze",
        ):
            with self.subTest(method=method):
                with self.assertRaisesRegex(RuntimeError, "辞書が読み込まれていません"):
                    getattr(self.engine, method)("猫")


class DictionaryNormalizerApiTests(unittest.TestCase):
    """モデルを起動せず辞書読み込みの失敗を明示する。"""

    def test_missing_dictionary_is_an_error(self):
        """文字列と Path のどちらも読み込みエラーを返す。"""
        with TemporaryDirectory() as tmp:
            for path in (str(Path(tmp) / "missing.hsd"), Path(tmp) / "missing.hsd"):
                with self.subTest(path=path):
                    with self.assertRaises(RuntimeError):
                        kotonoha.DictionaryNormalizer(path, symbols="+#")


class ModelApiTests(unittest.TestCase):
    """共有モデルのパス解決と既存ファイル保護を確認する。"""

    def test_model_dir_uses_shared_override(self):
        """環境変数の共有先を返し、ディレクトリを勝手に作らない。"""
        with TemporaryDirectory() as tmp:
            shared = Path(tmp) / "models"
            with patch.dict(os.environ, {"KOTONOHA_MODEL_BUNDLE": str(shared)}):
                self.assertEqual(Path(kotonoha.model_dir()), shared)
            self.assertFalse(shared.exists())

    def test_model_bundle_accepts_string_and_pathlike(self):
        """文字列と Path を受け付け、存在しないバンドルはロード時に失敗する。"""
        with TemporaryDirectory() as tmp:
            for bundle in (str(Path(tmp) / "missing"), Path(tmp) / "missing"):
                with self.subTest(bundle=bundle):
                    with self.assertRaises(RuntimeError):
                        kotonoha.KotonohaEngine(model_bundle=bundle)

    def test_download_protects_existing_model(self):
        """既存モデルが異なる場合は通信せず、内容を保持する。"""
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "teacher.knm"
            model.write_bytes(b"existing model")
            with self.assertRaisesRegex(RuntimeError, "--force"):
                kotonoha.download_models(dir=tmp, base_url="http://127.0.0.1:1")
            self.assertEqual(model.read_bytes(), b"existing model")
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
