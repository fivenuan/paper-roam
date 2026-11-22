"""PaperRoam command-line research, chat, and confirmed memory."""

import argparse
import contextlib
import io
import sqlite3
import sys
from collections import deque
from pathlib import Path

from roam_memory import MemoryStore, calculate
from roam_retrieval import Document, RetrievalIndex


def current_memory(store, interest):
    records = {row.key: row for row in store.search()}
    index = RetrievalIndex(Document(row.key, f"{row.key} {row.value}") for row in records.values())
    keys = [hit.document.id for hit in index.rank(interest, limit=6)]
    # Confirmed preferences can matter even without a shared search term.
    keys.extend(key for key in records if key not in keys)
    return tuple({"key": key, "value": records[key].value} for key in keys[:6])


def print_memory(records):
    if not records:
        print("没有匹配的已确认记忆。")
    for row in records:
        state = "当前" if row.current else "历史"
        print(f"- {row.key} = {row.value}（v{row.version}，{state}）")


def capture_memory(text, *, ask, api_key, store):
    result = ask(api_key, """
从用户原话提取适合写入研究记忆的候选。只取明确表达的长期偏好、已读论文、研究进度、待追问题或已定决定。
不要推测。复用提供的匹配键，改口沿用该键。输出 JSON，candidates 最多三项，每项含 key、value 和 source。
source 必须是原话中的连续逐字片段。value 是可独立使用的简短中文摘要。没有候选时返回空数组。
""", {"text": text, "existing_keys": [row.key for row in store.search()[:40]]}, max_tokens=550)
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return False
    confirmed = False
    for item in candidates[:3]:
        if not isinstance(item, dict):
            continue
        key, value, source = (item.get(field) for field in ("key", "value", "source"))
        if not all(isinstance(part, str) and part.strip() for part in (key, value, source)):
            continue
        key, value, source = key.strip(), value.strip(), source.strip()
        if source not in text or len(key) > 100 or len(value) > 600 or len(source) > 600:
            continue
        existing = store.get(key)
        if existing and existing.value == value:
            print(f"已存在：{key}（v{existing.version}）")
            continue
        if existing:
            print(f"当前：{existing.key} = {existing.value}")
        print(f"候选：{key} = {value}\n原话：{source}")
        try:
            accept = input("写入这条记忆？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return confirmed
        if accept in {"y", "yes", "是"}:
            print_memory([store.record(key, value)])
            confirmed = True
    if not confirmed:
        print("没有写入新记忆。")
    return confirmed


def chat(*, research, ask, api_key, path, with_memory, service_error, first="", options=None):
    history = deque(maxlen=2)
    print("PaperRoam。输入 /退出、/记住 键=值、/提取 原话、/记忆 关键词或 /计算 算式。")
    pending = first
    while True:
        try:
            text = pending or input("\n你> ").strip()
            pending = ""
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        command, _, argument = text.partition(" ")
        if command in {"/退出", "/exit", "/quit"}:
            return 0
        try:
            if command in {"/记住", "/提取", "/记忆"}:
                with open_store(path) as store:
                    if command == "/记住":
                        key, separator, value = argument.partition("=")
                        if not separator:
                            print("请输入 /记住 键=值。")
                            continue
                        print_memory([store.record(key, value)])
                        with_memory = True
                    elif command == "/提取":
                        if capture_memory(argument, ask=ask, api_key=api_key, store=store):
                            with_memory = True
                    else:
                        print_memory(store.search(argument))
            elif command == "/计算":
                print(calculate(argument))
            elif text:
                memory = ()
                if with_memory:
                    with open_store(path) as store:
                        memory = current_memory(store, text)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    research(text, api_key, context=tuple(history), memory=memory, **(options or {}))
                rendered = output.getvalue()
                print(rendered, end="")
                if rendered:
                    history.append({"ask": text[:700], "result": rendered[:1600]})
        except (ValueError, OSError, sqlite3.Error, service_error) as error:
            print(f"本轮操作失败：{error}", file=sys.stderr)


def open_store(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return MemoryStore(path)


def run(argv, *, api_key, research, ask, setting, service_error):
    parser = argparse.ArgumentParser(description="从 OpenAlex 找论文、核对原文和保存已确认的研究记忆。")
    parser.add_argument("interest", nargs="?", help="研究兴趣、问题或论文链接")
    parser.add_argument("--deep", action="store_true", help="查正文并报告证据位置")
    parser.add_argument("--pdf", action="append", default=[], help="本地 PDF 路径，可重复指定")
    parser.add_argument("--focus", default="", help="指定用于全文检索的短关键词")
    parser.add_argument("--retrieval", choices=("bm25", "hybrid", "reranked"), default="bm25",
                        help="全文段落检索方式")
    parser.add_argument("--reranker", choices=("local", "api"), default="local")
    parser.add_argument("--db", default=str(Path.home() / ".paperroam" / "memory.sqlite3"), help="SQLite 路径")
    parser.add_argument("--with-memory", action="store_true", help="显式使用此前确认的研究记忆")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chat", action="store_true", help="在同一个聊天中继续问")
    mode.add_argument("--calc", help="运行本地算术计算器")
    mode.add_argument("--remember", nargs=2, metavar=("KEY", "VALUE"), help="确认写入或更新记忆")
    mode.add_argument("--capture", help="从原话提取候选并逐条确认")
    mode.add_argument("--memory-search", help="查当前记忆；空字符串列出全部")
    mode.add_argument("--memory-history", help="查键的历史版本")
    args = parser.parse_args(argv)
    if len(args.pdf) > 3:
        parser.error("同一任务最多指定三个本地 PDF。")
    interest = (args.interest or "").strip()
    try:
        if args.calc is not None:
            print(calculate(args.calc))
            return 0
        if args.remember or args.memory_search is not None or args.memory_history is not None:
            with open_store(args.db) as store:
                if args.remember:
                    records = [store.record(*args.remember)]
                elif args.memory_history is not None:
                    records = store.history(args.memory_history)
                else:
                    records = store.search(args.memory_search)
                print_memory(records)
            return 0
        if not (interest or args.pdf or args.chat or args.capture is not None):
            parser.error("请输入研究问题、提供 PDF 或使用 --chat。")
        api_key = api_key or setting("DEEPSEEK_API_KEY")
        if not api_key:
            print("请先配置 DEEPSEEK_API_KEY。", file=sys.stderr)
            return 1
        if args.capture is not None:
            with open_store(args.db) as store:
                capture_memory(args.capture, ask=ask, api_key=api_key, store=store)
            return 0
        options = {"deep": args.deep, "pdf_paths": args.pdf, "focus": args.focus,
                   "retrieval": args.retrieval, "reranker": args.reranker}
        if args.chat:
            return chat(research=research, ask=ask, api_key=api_key, path=args.db,
                        with_memory=args.with_memory, service_error=service_error,
                        first=interest, options=options)
        memory = ()
        if args.with_memory:
            with open_store(args.db) as store:
                memory = current_memory(store, interest)
        return research(interest or "拆解指定论文的研究动机、方法、结果和局限", api_key,
                        memory=memory, **options)
    except (ValueError, OSError, sqlite3.Error, service_error) as error:
        print(f"操作失败：{error}", file=sys.stderr)
        return 1
