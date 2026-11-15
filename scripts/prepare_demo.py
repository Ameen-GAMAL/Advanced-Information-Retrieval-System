"""Prepare the demo's real models and index, and issue a smoke-test query."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_system.demo import build_demo


def main() -> None:
    demo = build_demo(ROOT)
    query = next(iter(demo.examples.values()))
    for profile in ("bm25", "hybrid+minilm"):
        result = demo.engine.search(query, profile=profile, top_k=3)
        if result.results.empty:
            raise RuntimeError(f"{profile} returned no results for the demo's warm-up query")
        if demo.dataset == "demo/fixture" and "fusion" not in result.results["docno"].tolist():
            raise RuntimeError(f"{profile} skipped the fixture's relevant document")
    print(f"Prepared {demo.label}: {len(demo.documents):,} documents. Smoke search passed.")


if __name__ == "__main__":
    main()
