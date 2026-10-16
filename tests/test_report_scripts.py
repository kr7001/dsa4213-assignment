"""Tests for the report scripts: the results tables and figures, and the number check."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    """Import a script from scripts/ as a module (scripts/ is not a package)."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


check = load_script("check_report_numbers")


def test_lookup_matches_the_rounding_the_text_shows():
    lookup = check.Lookup([0.053496, 1115210, 0.005977, 582.3, 4.0267])
    assert "0.053" in lookup
    assert "0.054" not in lookup  # a value rounded from rounded numbers is caught
    assert "5.3" in lookup  # as a percentage
    assert "1,115,210" in lookup
    assert "1.12" in lookup  # in millions
    assert "0.60" in lookup  # 0.5977% shown as a percentage
    assert "9.7" in lookup  # 582.3 seconds in minutes
    assert "4.027" in lookup
    assert "4.028" not in lookup


def test_prose_lines_skip_citations_references_and_inputs():
    tex = "\n".join(
        [
            r"\documentclass{article}",
            r"\begin{document}",
            r"\begin{abstract}",
            r"Test perplexity 40.8 \citep[Sec.~5]{radford2018improving}, Table~\ref{tab:x}.",
            r"\end{abstract}",
            r"\input{tables/part1.tex} % 99.9 in a comment",
            r"\bibliography{references}",
            r"\appendix",
            r"Appendix number 12.5.",
        ]
    )
    lines = check.prose_lines(tex)
    text = " ".join(line for _, line, _ in lines)
    assert "40.8" in text
    assert "5" not in text.replace("40.8", "").replace("12.5", "")
    assert "99.9" not in text
    body = [line for _, line, in_body in lines if in_body]
    assert any("40.8" in line for line in body)
    assert all("12.5" not in line for line in body)


def test_check_fails_on_an_untraced_body_number(tmp_path, monkeypatch):
    report = tmp_path / "report"
    (report / "tables").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    (report / "facts.json").write_text(json.dumps({"ppl": 40.77, "acc": 0.9330}))
    tex = report / "main.tex"
    tex.write_text(
        "\\begin{abstract}\nPerplexity 40.8 and accuracy 93.3\\%.\n\\end{abstract}\n"
        "\\bibliography{references}\n"
    )
    monkeypatch.chdir(tmp_path)
    assert check.main(["--tex", str(tex)]) == 0
    tex.write_text(
        "\\begin{abstract}\nPerplexity 40.9 and accuracy 93.3\\%.\n\\end{abstract}\n"
        "\\bibliography{references}\n"
    )
    assert check.main(["--tex", str(tex)]) == 1


@pytest.mark.skipif(
    not (ROOT / "results/eval/test.json").exists(), reason="needs the committed result files"
)
def test_results_tables_and_figures_build_from_the_result_files(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    results = load_script("report_results")
    out = tmp_path / "report"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir()
    committed = ROOT / "report/tables/samples.tex"
    if committed.exists():
        shutil.copy(committed, out / "tables/samples.tex")
    results.build_results(out)
    for name in ("curves", "curves_appendix", "positions", "part3", "part3_length"):
        assert (out / "figures" / f"{name}.pdf").exists()
    for name in ("part1", "part2", "part3", "generation", "errors", "comparisons", "tuning"):
        assert (out / "tables" / f"{name}.tex").read_text().startswith("\\begin{tabular")
    facts = json.loads((out / "facts.json").read_text())
    derived = facts["derived"]
    assert derived["neural_models"] == 19
    assert round(facts["part2"]["decomposition_seed_4213"]["glove_wins_back"], 3) == 0.053
    assert derived["discordant_last"] == 12
    categories = facts["errors"]["categories"]
    assert sum(c["count"] for c in categories.values()) == facts["errors"]["category_total"] == 40
