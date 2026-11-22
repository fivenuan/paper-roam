"""Explicitly confirmed research notes and bounded arithmetic for PaperRoam."""

import ast
import math
import operator
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    value: str
    version: int
    created_at: str
    current: bool


class MemoryStore:
    """Store user-confirmed values while retaining their previous versions."""

    def __init__(self, path: str | Path):
        self._connection = sqlite3.connect(path)
        try:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_versions (
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (key, version)
                )
            """)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._connection.close()

    def record(self, key: str, value: str) -> MemoryRecord:
        key = key.strip().casefold()
        value = value.strip()
        if not key or not value:
            raise ValueError("记忆的名称和值都不能为空。")

        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            current = self.get(key)
            if current is not None and current.value == value:
                return current
            version = 1 if current is None else current.version + 1
            created_at = datetime.now(timezone.utc).isoformat()
            self._connection.execute(
                "INSERT INTO memory_versions (key, value, version, created_at) VALUES (?, ?, ?, ?)",
                (key, value, version, created_at),
            )
            return MemoryRecord(key, value, version, created_at, current=True)

    def get(self, key: str) -> MemoryRecord | None:
        row = self._connection.execute("""
            SELECT key, value, version, created_at
            FROM memory_versions
            WHERE key = ?
            ORDER BY version DESC
            LIMIT 1
        """, (key.strip().casefold(),)).fetchone()
        return MemoryRecord(*row, current=True) if row is not None else None

    def search(self, query: str = "") -> list[MemoryRecord]:
        words = query.casefold().split()
        rows = self._connection.execute("""
            SELECT record.key, record.value, record.version, record.created_at
            FROM memory_versions AS record
            JOIN (
                SELECT key, MAX(version) AS version
                FROM memory_versions
                GROUP BY key
            ) AS latest USING (key, version)
            ORDER BY record.key
        """)
        return [
            MemoryRecord(*row, current=True)
            for row in rows
            if all(word in f"{row[0]} {row[1]}".casefold() for word in words)
        ]

    def history(self, key: str) -> list[MemoryRecord]:
        rows = self._connection.execute("""
            SELECT key, value, version, created_at
            FROM memory_versions
            WHERE key = ?
            ORDER BY version DESC
        """, (key.strip().casefold(),))
        return [
            MemoryRecord(*row, current=index == 0)
            for index, row in enumerate(rows)
        ]


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
CALCULATOR_TRANSLATION = str.maketrans({"×": "*", "÷": "/", "−": "-", "^": "**"})
MAX_MAGNITUDE = 10**100


def _bounded_number(value: int | float) -> int | float:
    if (
        type(value) not in (int, float)
        or (isinstance(value, float) and not math.isfinite(value))
        or abs(value) > MAX_MAGNITUDE
    ):
        raise ValueError("计算结果超出支持的实数范围。")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        return _bounded_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return _bounded_number(UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数的绝对值不能超过 100。")
        return _bounded_number(BINARY_OPERATORS[type(node.op)](left, right))
    raise ValueError("只支持实数、括号和基本算术运算。")


def calculate(expression: str) -> int | float:
    """Evaluate bounded literal-only arithmetic without executing Python code."""
    expression = expression.strip()
    if not expression or len(expression) > 200:
        raise ValueError("请输入 1 到 200 个字符的算式。")
    try:
        tree = ast.parse(expression.translate(CALCULATOR_TRANSLATION), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("请输入有效的算式。") from None
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("算式过于复杂。")
    try:
        return _evaluate(tree.body)
    except (ZeroDivisionError, OverflowError):
        raise ValueError("无法计算该算式。") from None
