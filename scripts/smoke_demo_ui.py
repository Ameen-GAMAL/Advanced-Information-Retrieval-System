"""Exercise the search UI against the fixture in the built demo image."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run()
    assert not app.exception, app.exception
    app.text_input(key="query_input").set_value("reciprocal rank fusion")
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert [heading.value for heading in app.subheader] == ["BM25 baseline", "Hybrid + MiniLM"]
    assert any("Judged relevant" in caption.value for caption in app.caption)
    print("Fixture UI search passed.")


if __name__ == "__main__":
    main()
