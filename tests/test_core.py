"""针对会影响结论可信度的边界测试，无须 API Key。"""
import base64
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from common.core import parse_summary, check_evidence, summarize, validate_segments
from common.audio import chunks
from scripts.evaluate import edit_distance, rouge_l, overlap_ratio


def summary(sid):
    return {"overview": "测试", "decisions": [{"text": "决定", "evidence_ids": [sid]}], "actions": [], "risks": []}


class Tests(unittest.TestCase):
    def test_unknown_evidence_rejected(self):
        with self.assertRaises(ValueError):
            check_evidence(summary("不存在"), {"s1"})

    def test_schema_rejected(self):
        with self.assertRaises(ValueError):
            parse_summary('{"overview":"x", "decisions":[], "actions":[{"text":"x","evidence_ids":[]}],"risks":[]}')

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate_segments([{"id": "s1", "text": "a"}, {"id": "s1", "text": "b"}])

    def test_hallucination_retry(self):
        class Fake:
            count = 0
            def text(self, system, payload):
                self.count += 1
                return json.dumps(summary("fake" if self.count == 1 else "s1"))
        client = Fake()
        result = summarize([{"id": "s1", "text": "会议内容"}], client)
        self.assertEqual(result["decisions"][0]["evidence_ids"], ["s1"])
        self.assertEqual(client.count, 2)

    def test_audio_tail_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.wav"
            with wave.open(str(path), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\x00\x00" * (16000 * 7))
            items = list(chunks(path, seconds=4, overlap=1))
            self.assertEqual([(x[1], x[2]) for x in items], [(0, 4), (3, 7)])
            for item in items:
                with wave.open(io.BytesIO(base64.b64decode(item[3].split(",")[1]))) as f:
                    self.assertEqual(f.getnframes(), 64000)

    def test_long_meeting_reduces_without_losing_valid_ids(self):
        class Fake:
            def text(self, system, payload):
                data = json.loads(payload)
                sid = data[0].get("id") or data[0]["decisions"][0]["evidence_ids"][0]
                return json.dumps(summary(sid))
        rows = [{"id": f"s{i}", "text": "讨论内容" * 80} for i in range(8)]
        result = summarize(rows, Fake(), budget=1000)
        self.assertEqual(result["decisions"][0]["evidence_ids"], ["s0"])

    def test_metrics(self):
        self.assertEqual(edit_distance("abc", "axc"), 1)
        self.assertEqual(rouge_l("abc", "abc"), 1)
        self.assertAlmostEqual(overlap_ratio([
            {"start": 0, "end": 4, "speaker": "A"},
            {"start": 1, "end": 2, "speaker": "A"},
            {"start": 2, "end": 6, "speaker": "B"}]), 2 / 6)


if __name__ == "__main__":
    unittest.main()
