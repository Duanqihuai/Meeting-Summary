"""说话人对齐边界：不能把无覆盖、重叠和换人情况硬贴到同一人。"""
import unittest
import json
from common.diarization import attribute_words, group_words, union_duration


class DiarizationTests(unittest.TestCase):
    def word(self, start, end, text=" word"):
        return {"start": start, "end": end, "text": text}

    def test_clear_change_splits_sentence(self):
        words = [self.word(0, 1, " Hello"), self.word(1, 2, " there")]
        turns = [{"start": 0, "end": 1, "speaker": "SPEAKER_01"},
                 {"start": 1, "end": 2, "speaker": "SPEAKER_02"}]
        rows = group_words(attribute_words(words, turns))
        self.assertEqual([r["speaker"] for r in rows], ["SPEAKER_01", "SPEAKER_02"])
        self.assertEqual([r["text"] for r in rows], ["Hello", "there"])

    def test_overlap_is_unknown(self):
        turns = [{"start": 0, "end": 2, "speaker": s} for s in ["SPEAKER_01", "SPEAKER_02"]]
        w = attribute_words([self.word(0.5, 1)], turns)[0]
        self.assertIsNone(w["speaker"])
        self.assertTrue(w["speaker_uncertain"])
        self.assertEqual(len(w["speaker_candidates"]), 2)

    def test_gap_does_not_borrow_nearest_speaker(self):
        w = attribute_words([self.word(3, 4)], [{"start": 0, "end": 2, "speaker": "SPEAKER_01"}])[0]
        self.assertIsNone(w["speaker"])

    def test_union_does_not_double_count(self):
        self.assertEqual(union_duration([(0, 2), (1, 3), (4, 5)]), 4)

    def test_same_speaker_long_pause_splits(self):
        turns = [{"start": 0, "end": 10, "speaker": "SPEAKER_01"}]
        self.assertEqual(len(group_words(attribute_words([self.word(0, 1), self.word(4, 5)], turns))), 2)

    def test_numpy_timestamps_are_serializable(self):
        import numpy as np
        words = [self.word(np.float64(0), np.float64(1))]
        aligned = attribute_words(words, [{"start": 0, "end": 2, "speaker": "SPEAKER_01"}])
        self.assertIs(type(aligned[0]["speaker_uncertain"]), bool)
        json.dumps(aligned)


if __name__ == "__main__":
    unittest.main()
