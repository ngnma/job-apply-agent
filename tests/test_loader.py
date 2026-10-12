import pytest

from job_agent.rag.loader import _clean_url, load_cv_text


def test_headings_are_marked(fake_cv_docx):
    text = load_cv_text(fake_cv_docx)
    lines = text.splitlines()
    assert "# Jane Doe" in lines
    assert "# Work History" in lines
    assert "### Programming Languages" in lines
    assert "Data Analyst | Acme Ltd | London, UK | 01/2024 – 06/2025" in lines


def test_bullets_start_with_dash(fake_cv_docx):
    lines = load_cv_text(fake_cv_docx).splitlines()
    assert "- Built a Python ETL pipeline with Airflow." in lines  # Word list bullet
    assert "- Ran 10+ A/B tests on pricing." in lines  # typed "•" bullet
    assert not any(line.startswith("•") for line in lines)


def test_hyperlinks_keep_url(fake_cv_docx):
    text = load_cv_text(fake_cv_docx)
    assert "GitHub (https://github.com/jane/demo)" in text
    assert "GitHub (https://github.com/jane/broken)" in text  # damaged link repaired


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://github.com/a", "https://github.com/a"),
        ("○%09https:/github.com/a", "https://github.com/a"),
        ("http:///github.com/a", "http://github.com/a"),
    ],
)
def test_clean_url(raw, expected):
    assert _clean_url(raw) == expected


def test_unsupported_file_type_raises(tmp_path):
    bad = tmp_path / "cv.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_cv_text(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cv_text(tmp_path / "nope.docx")
