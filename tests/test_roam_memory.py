import tempfile
import unittest
from pathlib import Path

from roam_memory import MemoryStore, calculate


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "memory.sqlite3"

    def test_update_preserves_history_and_reads_current_value_after_reopen(self):
        with MemoryStore(self.path) as store:
            store.record("论文语言", "优先英文")
            store.record("论文语言", "优先中文")
        with MemoryStore(self.path) as store:
            self.assertEqual(store.get("论文语言").value, "优先中文")
            self.assertEqual(
                [(row.version, row.value, row.current) for row in store.history("论文语言")],
                [(2, "优先中文", True), (1, "优先英文", False)],
            )

    def test_unchanged_record_does_not_create_another_version(self):
        with MemoryStore(self.path) as store:
            first = store.record(" Research Topic ", "AI memory")
            again = store.record("research topic", " AI memory ")
            self.assertEqual(first, again)
            self.assertEqual(store.history("RESEARCH TOPIC"), [first])

    def test_search_only_returns_current_records_and_missing_lookup_is_empty(self):
        with MemoryStore(self.path) as store:
            store.record("论文语言", "优先英文")
            store.record("论文语言", "优先中文")
            store.record("研究主题", "PaperQA evidence retrieval")
            self.assertEqual(store.search("英文"), [])
            self.assertEqual([row.key for row in store.search("paperqa retrieval")], ["研究主题"])
            self.assertEqual(len(store.search()), 2)
            self.assertIsNone(store.get("未知"))

    def test_blank_inputs_do_not_write_a_record(self):
        with MemoryStore(self.path) as store:
            for key, value in [("", "偏好"), ("论文语言", "  ")]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    store.record(key, value)
            self.assertEqual(store.search(), [])


class CalculatorTests(unittest.TestCase):
    def test_calculates_precedence_exact_integers_and_decimal_arithmetic(self):
        for expression, expected in [
            ("(3000 - 780) - 465", 1755),
            ("2 × (3 + 4) ÷ 2", 7.0),
            ("2^5 + 7 // 3 + 7 % 3", 35),
            ("5e-1 * 8", 4.0),
            ("-3**2", -9),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression), expected)

    def test_rejects_code_and_unbounded_or_invalid_arithmetic(self):
        for expression in [
            "",
            "abs(-4)",
            "(1).__class__",
            "[1, 2]",
            "True",
            "1 / 0",
            "2 ** (2 ** 20)",
            "1e999",
            "(-1) ** 0.5",
            "+" * 202 + "1",
        ]:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculate(expression)
