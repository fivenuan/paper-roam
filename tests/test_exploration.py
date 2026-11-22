import contextlib
import io
import json
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import paper_roam
from test_paper_roam import RECORD, model_result


QUERIES = {
    "direct": "long-term memory language models",
    "adjacent": "human memory interference",
    "impact": "ecological resilience",
    "novel": "urban biodiversity acoustic monitoring",
}


def record(number, field, **extra):
    return {
        **RECORD,
        "id": f"https://openalex.org/W{number}",
        "doi": f"https://doi.org/10.1000/test-{number}",
        "display_name": f"Test paper {number}",
        "cited_by_count": 37,
        "primary_topic": {"field": {"display_name": field}},
        **extra,
    }


def pick(route, number):
    return {"route": route, "id": f"https://openalex.org/W{number}", "evidence_id": 1}


class ExplorationTests(unittest.TestCase):
    def test_topic_free_prompt_receives_distinct_discipline_seeds(self):
        response = model_result({
            "route": "explore", "queries": QUERIES, "limit": 4, "has_topic": False,
        })
        with patch("paper_roam.request_json", return_value=response) as request:
            paper_roam.search_plan("我不知道今天该看什么论文", "test")
        payload = json.loads(request.call_args.kwargs["body"]["messages"][1]["content"])
        seeds = payload["discovery_seeds"]
        self.assertEqual(set(seeds), set(QUERIES))
        self.assertEqual(len(set(seeds.values())), 4)

    def test_topic_free_selection_prefers_different_fields_over_model_rank(self):
        primary = paper_roam.Paper(
            id="lead", url="https://doi.org/10.1000/lead", title="Learning",
            year=2024, kind="article", abstract="Abstract", field="Psychology", citations=7,
        )
        same_field = paper_roam.Paper(
            id="same", url="https://doi.org/10.1000/same", title="Risk perception",
            year=2024, kind="article", abstract="Abstract", field="Psychology", citations=9,
        )
        different = paper_roam.Paper(
            id="other", url="https://doi.org/10.1000/other", title="Acoustic archives",
            year=2024, kind="article", abstract="Abstract", field="History", citations=2,
        )
        choices = {
            "direct": [(primary, "Abstract")],
            "adjacent": [(same_field, "Abstract"), (different, "Abstract")],
            "impact": [],
            "novel": [],
        }
        selected, covered = paper_roam.assemble_exploration(choices, {"limit": 2, "has_topic": False})
        self.assertEqual([paper.id for _, paper, _ in selected], ["lead", "other"])
        self.assertTrue(covered)

    def test_four_searches_and_one_cross_field_slot_for_three_papers(self):
        plan = model_result({
            "route": "explore", "queries": QUERIES, "limit": 3, "require_cross_field": True,
        })
        responses = [
            plan,
            {"results": [record(1, "Computer Science")]},
            {"results": [record(2, "Psychology")]},
            {"results": [record(3, "Environmental Science")]},
            {"results": [record(4, "Environmental Science")]},
            model_result({"picks": [pick("direct", 1), pick("adjacent", 2), pick("impact", 3), pick("novel", 4)]}),
            model_result({"reasons": ["第一条理由。", "第二条理由。", "第三条理由。"]}),
        ]
        with patch("paper_roam.request_json", side_effect=responses) as request:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = paper_roam.main(["推荐三篇 AI 长期记忆论文，至少一篇来自陌生领域"], api_key="test")
        self.assertEqual(code, 0)
        report = output.getvalue()
        self.assertEqual(report.count("   - 来源："), 3)
        self.assertIn("跨领域影响力", report)
        self.assertIn("https://doi.org/10.1000/test-3", report)
        self.assertIn("37", report)
        params = [parse_qs(urlsplit(call.args[0]).query) for call in request.call_args_list[1:5]]
        self.assertEqual([query["search.title_abstract_keywords"][0] for query in params], list(QUERIES.values()))
        self.assertNotIn("sort", params[2])
        self.assertEqual(params[2]["per_page"], ["12"])
        self.assertIn("from_publication_date:", params[3]["filter"][0])
        data = json.loads(request.call_args_list[5].kwargs["body"]["messages"][1]["content"])
        self.assertEqual(data["queries"], QUERIES)
        self.assertEqual(data["routes"]["impact"][0]["field"], "Environmental Science")

    def test_open_ended_exploration_deduplicates_and_explains_unmet_coverage(self):
        responses = [
            model_result({
                "route": "explore", "queries": QUERIES, "limit": 3,
                "require_cross_field": True, "has_topic": False,
            }),
            {"results": [record(1, "Computer Science")]},
            {"results": [record(2, "Psychology")]},
            {"results": [record(3, "Computer Science")]},
            {"results": [record(4, "Psychology", doi="https://doi.org/10.1000/test-2")]},
            model_result({"picks": [pick("direct", 1), pick("adjacent", 2), pick("impact", 3), pick("novel", 4)]}),
            model_result({"reasons": ["第一条理由。", "第二条理由。"]}),
        ]
        with patch("paper_roam.request_json", side_effect=responses):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = paper_roam.main(["我不知道今天该看什么论文，想看陌生方向"], api_key="test")
        self.assertEqual(code, 0)
        report = output.getvalue()
        self.assertIn("【探索起点】", report)
        self.assertEqual(report.count("https://doi.org/10.1000/test-2"), 1)
        self.assertIn("已覆盖不同学科", report)
        self.assertNotIn("https://doi.org/10.1000/test-3", report)

    def test_rejected_extra_route_and_empty_results_are_visible(self):
        responses = [
            model_result({"route": "explore", "queries": QUERIES, "limit": 3, "require_cross_field": True}),
            {"results": [record(1, "Computer Science")]},
            {"results": []},
            {"results": [record(3, "Computer Science")]},
            {"results": []},
            model_result({"picks": [pick("direct", 1), pick("impact", 3), pick("adjacent", 3)]}),
            model_result({"reasons": ["第一条理由。"]}),
        ]
        with patch("paper_roam.request_json", side_effect=responses):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = paper_roam.main(["推荐三篇论文，至少一篇来自陌生领域"], api_key="test")
        self.assertEqual(code, 0)
        report = output.getvalue()
        self.assertIn("未找到可确认的跨领域推荐", report)
        self.assertIn("相似机制：没有可用摘要", report)
        self.assertEqual(report.count("   - 来源："), 1)

    def test_impact_ranks_citations_inside_relevant_candidate_pool(self):
        response = {"results": [
            record(1, "Ecology", cited_by_count=12),
            record(2, "Ecology", cited_by_count=99),
            record(3, "Ecology", cited_by_count=None),
        ]}
        with patch("paper_roam.request_json", return_value=response):
            papers = paper_roam.search_papers("ecological resilience", route="impact")
        self.assertEqual([paper.citations for paper in papers], [99, 12, None])

    def test_same_paper_with_two_field_labels_cannot_meet_cross_field_request(self):
        direct = paper_roam.Paper(
            id="direct", url="https://doi.org/10.1000/demo", title="Example",
            year=2024, kind="article", abstract="Abstract", field="Computer Science", citations=7,
        )
        duplicate = paper_roam.Paper(
            id="duplicate", url=direct.url, title=direct.title, year=2024,
            kind="preprint", abstract="Abstract", field="Neuroscience", citations=0,
        )
        selected, covered = paper_roam.assemble_exploration(
            {"direct": [(direct, "Abstract")], "adjacent": [(duplicate, "Abstract")], "impact": [], "novel": []},
            {"limit": 1, "require_cross_field": True},
        )
        self.assertEqual([paper.id for _, paper, _ in selected], ["direct"])
        self.assertFalse(covered)
