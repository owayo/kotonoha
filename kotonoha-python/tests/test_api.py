"""Python 入力の既定値と公開 API の変換結果を検査する。"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import kotonoha


class EngineApiTests(unittest.TestCase):
    """辞書やモデルなしで入力変換と出力の契約を確認する。"""

    def setUp(self):
        """外部モデルの設定を除いてエンジンを用意する。"""
        # 手元のモデル設定が辞書不要のテストに混ざらないようにする。
        with patch.dict(os.environ):
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


if __name__ == "__main__":
    unittest.main()
