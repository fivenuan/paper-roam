import contextlib
import io
import unittest
from unittest.mock import patch

import paper_roam
import roam_fulltext
from test_paper_roam import RECORD, model_result


class FullTextTests(unittest.TestCase):
    def test_versioned_arxiv_uses_original_pdf_page(self):
        question = "核对 https://arxiv.org/abs/2410.10813v2 的实验"
        with (
            patch("roam_fulltext.read_pdf", return_value=[(3, "This experiment used dialogue histories.")]) as read,
            patch("paper_roam.request_json", return_value=model_result({
                "claims": [{"id": "E1", "text": "实验使用了对话历史。", "quote": "This experiment used dialogue histories."}],
            })) as call,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            code = paper_roam.main(["--deep", "--focus", "experiment", question], api_key="test")
        self.assertEqual(code, 0)
        self.assertEqual(read.call_args.args[0], "https://arxiv.org/pdf/2410.10813v2")
        self.assertEqual(call.call_count, 1)
        self.assertIn("第 3 页", output.getvalue())
        self.assertIn("This experiment used dialogue histories.", output.getvalue())

    def test_missing_pdf_reports_abstract_and_uses_only_one_followup(self):
        responses = [
            model_result({"route": "deep", "query": "conversation memory tests"}),
            {"results": [RECORD]},
            model_result({"claims": [], "missing": ["实验条件"], "followup_query": "longmemeval evaluation"}),
            {"results": []},
            model_result({"claims": [{"id": "E1", "text": "该论文评估对话记忆。", "quote": "We evaluate dialogue memory."}], "missing": ["实验条件"],
                          "followup_query": "ignored"}),
        ]
        with (
            patch("paper_roam.request_json", side_effect=responses) as call,
            patch("roam_fulltext.read_pdf", side_effect=roam_fulltext.SourceError("无法获取全文")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            code = paper_roam.main(["对话记忆实验做了什么"], api_key="test")
        self.assertEqual(code, 0)
        self.assertEqual(call.call_count, 5)
        self.assertIn("摘要", output.getvalue())
        self.assertIn("实验条件", output.getvalue())

    def test_invalid_claim_is_dropped_but_valid_sources_remain(self):
        with (
            patch("roam_fulltext.read_pdf", return_value=[(1, "The paper describes the study.")]),
            patch("paper_roam.request_json", return_value=model_result({"claims": [
                {"id": "E91", "text": "unsupported statement"},
                {"id": "E1", "text": "verified link https://example.invalid"},
                {"id": "E1", "text": "论文描述了一项研究。", "quote": "The paper describes the study."},
                {"id": "E1", "text": "100 人参加。", "quote": "The paper describes the study."},
                {"id": "E1", "text": "错误的摘录", "quote": "Fabricated evidence."},
            ]})),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            paper_roam.main(["--deep", "--focus", "study", "https://arxiv.org/abs/2410.10813v2"], api_key="test")
        self.assertIn("论文描述了一项研究。", output.getvalue())
        self.assertNotIn("unsupported statement", output.getvalue())
        self.assertNotIn("example.invalid", output.getvalue())
        self.assertNotIn("100 人参加", output.getvalue())
        self.assertNotIn("错误的摘录", output.getvalue())

    def test_focus_does_not_prevent_topic_based_search(self):
        responses = [
            model_result({"route": "recommend", "query": "long term memory"}),
            {"results": []},
            model_result({"claims": [], "missing": ["no evidence"]}),
        ]
        with (
            patch("paper_roam.request_json", side_effect=responses) as call,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = paper_roam.main(["--deep", "--focus", "experiment", "long term memory"], api_key="test")
        self.assertEqual(code, 0)
        self.assertIn("search=long+term+memory", call.call_args_list[1].args[0])

    def test_unavailable_explicit_pdf_falls_back_to_same_version_abstract(self):
        abstract = "This study evaluates knowledge updates."
        with (
            patch("roam_fulltext.read_pdf", side_effect=roam_fulltext.SourceError("无法获取全文")),
            patch("roam_fulltext.read_arxiv_abstract", return_value=abstract) as read,
            patch("paper_roam.request_json", return_value=model_result({
                "claims": [{"id": "E1", "text": "作者称其评估知识更新。", "quote": abstract}],
            })),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            code = paper_roam.main(["--deep", "--focus", "knowledge updates",
                                    "https://arxiv.org/abs/2410.10813v2"], api_key="test")
        self.assertEqual(code, 0)
        read.assert_called_once_with("2410.10813v2")
        self.assertIn("作者摘要", output.getvalue())
        self.assertIn("无法获取全文", output.getvalue())

    def test_failed_supplement_keeps_first_claim(self):
        responses = [
            model_result({"route": "deep", "query": "conversation memory tests"}),
            {"results": [RECORD]},
            model_result({"claims": [{"id": "E1", "text": "该论文评估对话记忆。",
                                     "quote": "We evaluate dialogue memory."}],
                          "missing": ["实验条件"], "followup_query": "longmemeval evaluation"}),
            paper_roam.PaperRoamError("补搜超时"),
            model_result({"claims": [], "missing": ["实验条件"]}),
        ]
        with (
            patch("paper_roam.request_json", side_effect=responses) as call,
            patch("roam_fulltext.read_pdf", side_effect=roam_fulltext.SourceError("无法获取全文")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            code = paper_roam.main(["对话记忆实验做了什么"], api_key="test")
        self.assertEqual(code, 0)
        self.assertEqual(call.call_count, 5)
        self.assertIn("该论文评估对话记忆。", output.getvalue())
        self.assertIn("补搜超时", output.getvalue())
        self.assertIn("实验条件", output.getvalue())

    def test_deep_task_can_continue_after_one_calculation(self):
        original = "The authors report three groups in the experiment."
        replies = [
            model_result({"claims": [], "calculation": "100 * (9 - 6) / 12"}),
            model_result({"claims": [{"id": "E1", "text": "作者报告了三个实验组。", "quote": original}]}),
        ]
        with (
            patch("roam_fulltext.read_pdf", return_value=[(3, original)]),
            patch("paper_roam.request_json", side_effect=replies) as call,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            paper_roam.main(["--deep", "--focus", "groups", "https://arxiv.org/abs/2410.10813v2"],
                            api_key="test")
        self.assertEqual(call.call_count, 2)
        second_prompt = call.call_args_list[-1].kwargs["body"]["messages"][-1]["content"]
        self.assertIn("100 * (9 - 6) / 12 = 25.0", second_prompt)
        self.assertIn("作者报告了三个实验组。", output.getvalue())
