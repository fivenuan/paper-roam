import contextlib
import io
import json
import unittest
from unittest.mock import patch

import paper_roam


RECORD = {
    "id": "https://openalex.org/W4403571340",
    "doi": "https://doi.org/10.48550/arXiv.2410.10813",
    "display_name": "LongMemEval",
    "publication_year": 2024,
    "type": "preprint",
    "is_retracted": False,
    "abstract_inverted_index": {
        "memory.": [3],
        "We": [0],
        "evaluate": [1],
        "dialogue": [2],
    },
}


def model_result(value):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}


class RecommendationTests(unittest.TestCase):
    def test_chinese_interest_returns_index_link_and_quoted_abstract(self):
        responses = [
            model_result({"route": "recommend", "query": "conversational long-term memory"}),
            {"results": [RECORD]},
            model_result({"picks": [{
                "id": RECORD["id"],
                "evidence_id": 1,
                "evidence": "这是一段模型翻译，不能用作原文。",
                "url": "https://example.invalid/",
            }]}),
            model_result({"reasons": ["这篇论文评估对话记忆。"]}),
        ]
        output = io.StringIO()
        with patch("paper_roam.request_json", side_effect=responses) as request:
            with contextlib.redirect_stdout(output):
                code = paper_roam.main(["我想了解 AI 长期记忆"], api_key="test")
        self.assertEqual(code, 0)
        report = output.getvalue()
        self.assertIn(RECORD["doi"], report)
        self.assertIn("We evaluate dialogue memory.", report)
        self.assertIn("这篇论文评估对话记忆。", report)
        self.assertNotIn("example.invalid", report)
        self.assertIn("search=conversational+long-term+memory", request.call_args_list[1].args[0])
        self.assertIn("We evaluate dialogue memory.", request.call_args_list[2].kwargs["body"]["messages"][1]["content"])
        evidence_input = json.loads(request.call_args_list[3].kwargs["body"]["messages"][1]["content"])
        self.assertEqual(evidence_input["evidence"], ["We evaluate dialogue memory."])

    def test_unrelated_input_does_not_call_openalex(self):
        output = io.StringIO()
        with patch("paper_roam.request_json", return_value=model_result({"route": "other", "query": ""})) as request:
            with contextlib.redirect_stdout(output):
                code = paper_roam.main(["我去吃饭了"], api_key="test")
        self.assertEqual(code, 0)
        self.assertEqual(request.call_count, 1)
        self.assertIn("研究兴趣", output.getvalue())

    def test_missing_abstract_does_not_call_selection_model(self):
        responses = [
            model_result({"route": "recommend", "query": "memory"}),
            {"results": [{**RECORD, "abstract_inverted_index": None}]},
        ]
        with patch("paper_roam.request_json", side_effect=responses) as request:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = paper_roam.main(["memory"], api_key="test")
        self.assertEqual(code, 0)
        self.assertEqual(request.call_count, 2)
        self.assertIn("作者摘要", output.getvalue())

    def test_conference_recording_is_not_recommended_as_a_paper(self):
        responses = [
            model_result({"route": "recommend", "query": "memory"}),
            {"results": [{**RECORD, "type": "other", "doi": "https://doi.org/10.48448/vgbf-t005"}]},
            model_result({"picks": []}),
        ]
        with patch("paper_roam.request_json", side_effect=responses) as request:
            with contextlib.redirect_stdout(io.StringIO()):
                code = paper_roam.main(["memory"], api_key="test")
        self.assertEqual(code, 0)
        self.assertEqual(request.call_count, 2)

    def test_invalid_paper_or_missing_passage_never_reaches_report(self):
        for pick in [
            {"id": "unknown", "reason": "bad", "evidence_id": 1},
            {"id": RECORD["id"], "reason": "bad", "evidence_id": 2},
        ]:
            with self.subTest(pick=pick):
                responses = [
                    model_result({"route": "recommend", "query": "memory"}),
                    {"results": [RECORD]},
                    model_result({"picks": [pick]}),
                ]
                with patch("paper_roam.request_json", side_effect=responses):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        with contextlib.redirect_stderr(io.StringIO()) as errors:
                            code = paper_roam.main(["memory"], api_key="test")
                self.assertEqual(code, 1)
                self.assertNotIn(RECORD["doi"], output.getvalue())
                self.assertIn("无法核对", errors.getvalue())

    def test_model_cannot_add_a_link_in_its_recommendation_reason(self):
        responses = [
            model_result({"route": "recommend", "query": "memory"}),
            {"results": [RECORD]},
            model_result({"picks": [{
                "id": RECORD["id"],
                "evidence_id": 1,
            }]}),
            model_result({"reasons": ["Read the [results](https://example.invalid/unverified)."]}),
        ]
        with patch("paper_roam.request_json", side_effect=responses):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = paper_roam.main(["memory"], api_key="test")
        self.assertEqual(code, 1)
        self.assertNotIn("example.invalid", output.getvalue())
