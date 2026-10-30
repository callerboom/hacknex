"""
Memory feature test helper (HNX26EPS08).
Uses --fake-date to seed past conversations for testing day-based recall.

Usage
-----
Step 1 — seed fake history across 3 days:
    python bench/test_memory.py --seed

Step 2 — run memory queries (no LLM, no mic needed):
    python bench/test_memory.py --query

Step 3 — verify forget:
    python bench/test_memory.py --forget
"""

import argparse
import sys
import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.edge.memory import Memory
from src.edge.router import route_intent


DB_PATH = "data/memory_test.db"


def seed(mem: Memory) -> None:
    """Insert fake turns across 3 days using _fake_day parameter."""
    today = datetime.date.today()
    day0 = (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
    day1 = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    day2 = today.strftime("%Y-%m-%d")

    pairs = [
        # Day 1 (two days ago)
        (day0, "What is the capital of France?", "The capital of France is Paris."),
        (day0, "How do I treat a burn?", "Cool it under running water for 10 minutes."),
        # Day 2 (yesterday)
        (day1, "What is 48 times 12?", "The answer is 576."),
        (day1, "Why is the sky blue?", "Because of Rayleigh scattering of sunlight."),
        # Day 3 (today)
        (day2, "Check battery status", "Battery is at 82%, discharging."),
    ]

    for fake_day, user, bot in pairs:
        mem.save(user, bot, _fake_day=fake_day)

    print(f"[Seed] Inserted {len(pairs)} turns across days: {day0}, {day1}, {day2}")


def run_queries(mem: Memory) -> None:
    """Test memory-backed fast-path intents through route_intent."""
    queries = [
        "what did I ask today",
        "what did I ask yesterday",
        "what did I ask on day 1",
        "what did I ask on day 2",
    ]
    print("\n[Queries]")
    for q in queries:
        reply = route_intent(q, memory=mem)
        status = "FAST-PATH" if reply else "-> LLM"
        print(f"  [{status}] '{q}'\n    -> {reply or '[no fast-path match]'}\n")


def run_search(mem: Memory) -> None:
    """Test FTS5 search."""
    hits = mem.search("burn treatment", k=2)
    print(f"\n[Search] 'burn treatment' -> {hits}")

    hits2 = mem.search("capital France", k=2)
    print(f"[Search] 'capital France' -> {hits2}")


def run_forget(mem: Memory) -> None:
    reply = route_intent("forget everything", memory=mem)
    print(f"\n[Forget] reply: {reply}")
    print(f"[Forget] recent after clear: {mem.recent(5)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory feature test")
    parser.add_argument("--seed",   action="store_true", help="Insert fake history")
    parser.add_argument("--query",  action="store_true", help="Run memory queries")
    parser.add_argument("--search", action="store_true", help="Run FTS5 search")
    parser.add_argument("--forget", action="store_true", help="Test forget everything")
    parser.add_argument("--all",    action="store_true", help="Run all steps")
    args = parser.parse_args()

    mem = Memory(path=DB_PATH)
    if not mem.is_available:
        print("[ERROR] Memory DB could not be opened.")
        sys.exit(1)

    if args.all or args.seed:
        seed(mem)
    if args.all or args.query:
        run_queries(mem)
    if args.all or args.search:
        run_search(mem)
    if args.all or args.forget:
        run_forget(mem)

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
