import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paper_roam
from roam_memory import MemoryStore
from test_paper_roam import model_result


class AssistantCLITests(unittest.TestCase):
    def test_memory_is_opt_in_and_provides_latest_confirmed_value(self):
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "memory.sqlite3")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(paper_roam.main(["--db", db, "--remember", "语言", "English"]), 0)
                self.assertEqual(paper_roam.main(["--db", db, "--remember", "语言", "中文"]), 0)
                with patch("paper_roam.run_interest", return_value=0) as research:
                    paper_roam.main(["--db", db, "语言"], api_key="test")
                    self.assertEqual(research.call_args.kwargs["memory"], ())
                    paper_roam.main(["--db", db, "--with-memory", "语言"], api_key="test")
                    self.assertEqual(research.call_args.kwargs["memory"][0]["value"], "中文")

    def test_capture_candidates_requires_confirmation_and_verbatim_source(self):
        text = "以后优先读中文论文"
        candidates = {"candidates": [
            {"key": "论文语言", "value": "中文", "source": "优先读中文论文"},
            {"key": "实验模型", "value": "DeepSeek", "source": "杜撰的原话"},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "memory.sqlite3")
            with (
                patch("paper_roam.request_json", return_value=model_result(candidates)),
                patch("builtins.input", side_effect=["n", "y"]) as confirm,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                paper_roam.main(["--db", db, "--capture", text], api_key="test")
                paper_roam.main(["--db", db, "--capture", text], api_key="test")
            with MemoryStore(db) as store:
                self.assertEqual(store.get("论文语言").value, "中文")
                self.assertIsNone(store.get("实验模型"))
            self.assertEqual(confirm.call_count, 2)

    def test_calculator_does_not_call_the_model(self):
        with (
            patch("paper_roam.request_json") as call,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(paper_roam.main(["--calc", "100 * (9 - 6) / 12"]), 0)
        self.assertIn("25", output.getvalue())
        call.assert_not_called()
