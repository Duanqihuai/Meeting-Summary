import unittest

from common.asr import (
    _group_timestamped_items,
    _result_tokens,
    _resolve_model_location,
)


class SenseVoiceParsingTests(unittest.TestCase):
    def test_english_words_and_punctuation_are_grouped(self):
        result = {
            "words": ["Hello", "world", "."],
            "timestamp": [[100, 600], [700, 1200], [1200, 1300]],
        }
        items = _result_tokens([result], duration=2.0)
        words, segments = _group_timestamped_items(items)
        self.assertEqual([w["text"] for w in words], ["Hello", " world"])
        self.assertEqual(segments[0]["text"], "Hello world.")
        self.assertEqual(segments[0]["end"], 1.2)

    def test_chinese_words_do_not_gain_spaces(self):
        result = {
            "words": ["你", "好", "。"],
            "timestamp": [[100, 300], [300, 500], [500, 600]],
        }
        items = _result_tokens([result], duration=1.0)
        words, segments = _group_timestamped_items(items)
        self.assertEqual([w["text"] for w in words], ["你", "好"])
        self.assertEqual(segments[0]["text"], "你好。")

    def test_non_lexical_symbols_are_ignored(self):
        result = {
            "words": ["Yes", "😊", "."],
            "timestamp": [[100, 300], [300, 400], [400, 500]],
        }
        items = _result_tokens([result], duration=1.0)
        words, segments = _group_timestamped_items(items)
        self.assertEqual([w["text"] for w in words], ["Yes"])
        self.assertEqual(segments[0]["text"], "Yes.")

    def test_contractions_do_not_gain_internal_spaces(self):
        result = {
            "words": ["How", "'", "s", "your"],
            "timestamp": [
                [100, 200],
                [200, 220],
                [220, 300],
                [350, 500],
            ],
        }
        items = _result_tokens([result], duration=1.0)
        _, segments = _group_timestamped_items(items)
        self.assertEqual(segments[0]["text"], "How's your")

    def test_old_whisper_names_are_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_model_location("small")


if __name__ == "__main__":
    unittest.main()
