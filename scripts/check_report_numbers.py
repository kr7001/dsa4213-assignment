"""Check that every number in the report's text can be traced to a result file.

Usage:
    uv run python scripts/make_report_figures.py      # writes report/facts.json
    uv run python scripts/check_report_numbers.py [--tex report/main.tex] [--all]

It reads the prose of report/main.tex (the generated tables are skipped: they
are written from the result files by make_report_figures.py) and looks up each
number among:

1. report/facts.json: results and every derived number the text quotes
   (differences, ratios, shares), computed by scripts/report_results.py;
2. the numbers in the generated tables (report/tables/*.tex);
3. the settings in configs/*.yaml;
4. the counts-only data files: results/data, results/eda, results/preprocess,
   results/ngram, results/pilot;
5. for the appendices only, every other number in results/.

A number counts as traced when some value from these files, shown with the same
number of decimals, gives the same digits: as it is, as a percentage, or in
thousands, millions, billions, minutes or hours. Numbers quoted from other papers
are listed in EXTERNAL with their source. The check is one-sided: a traced number
can still be the wrong one of two similar values, so a person should also read the
text against the tables (the report's "unsure list" says which claims to check).

It prints every number it could not trace, with its line and context, and exits
with status 1 if any number in the body (abstract to Section 5) is untraced.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

RESULTS = Path("results")
TIER1_RESULT_FOLDERS = ("data", "eda", "preprocess", "ngram", "pilot")

# Numbers quoted from other sources, with where they come from.
EXTERNAL = {
    "90.34": "Carlson et al. (2025): GloVe 2024, CoNLL-03 F1 at 100 dimensions",
    "90.72": "Carlson et al. (2025): GloVe 2024, CoNLL-03 F1 at 300 dimensions",
    "11.9": "GloVe project page: 11.9B tokens of Wikipedia and Gigaword",
    "2.3": "report/tables/core_spread.tex (core parameters within 2.3%)",
    "26": "Merity et al. (2018), AWD-LSTM parameters per training word (Penn Treebank)",
    "16": "Merity et al. (2017, 2018), AWD-LSTM parameters per training word (WikiText-2)",
    "03": "the name of the CoNLL-03 dataset",
    "2014": "the name of GloVe's 2014 release",
}
# Powers of ten written as settings or bounds in the text.
POWERS = {
    "1e-05": "a bound: Kendall's tau p-value 5.2e-6 (part3.json); the tuning ladder's lowest rate",
    "1e-08": "Adam's epsilon (run.json); the lower limit of C (configs/part3.yaml, lowest: -8)",
    "1e+06": "the upper limit of C (configs/part3.yaml, highest: 6)",
}

# Patterns removed before numbers are read: commands whose arguments are not prose.
STRIP = [
    r"\\cite[a-z]*\*?(\[[^\]]*\])*\{[^}]*\}",
    r"\\(ref|label|input|url|includegraphics|graphicspath|bibliography|bibliographystyle)(\[[^\]]*\])?\{[^}]*\}",
    r"\\texttt\{[^}]*\}",
    r"\\setlength\{[^}]*\}\{[^}]*\}",
    r"\\addlinespace\[[^\]]*\]",
    r"\\(begin|end)\{[a-zA-Z*]+\}(\{[^}]*\})?",
    r"[pm]\{0\.\d+\\(textwidth|columnwidth)\}",
    r"\\titlebox\{[^}]*\}",
    r"\\counterwithin\{[^}]*\}\{[^}]*\}",
    r"\\(sub)?section\*?\{",
    r"\[(\d+(\.\d+)?)(pt|cm|mm|em)\]",
    r"\d+(\.\d+)?(pt|cm|mm|em)\b",
    r"\\[a-zA-Z]+\{[a-z]+\d*[a-z]*\}",  # \tok{num}, \tok{bos}, ...
]
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:(?:\{,\}|,)\d{3})+|\d+)(\.\d+)?(?![\w.]*\d)")
POWER = re.compile(r"10\^\{?(-?\d+)\}?")


def numbers_in(value: object) -> Iterator[float]:
    """Every number inside a JSON or YAML structure."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from numbers_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from numbers_in(item)
    elif isinstance(value, str):
        for match in re.finditer(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", value):
            with contextlib.suppress(ValueError):
                yield float(match.group(0))


def tex_numbers(path: Path) -> Iterator[float]:
    for match in NUMBER.finditer(path.read_text(encoding="utf-8").replace("{,}", ",")):
        yield float(match.group(0).replace(",", ""))


def json_files(folders: Iterable[Path]) -> Iterator[Path]:
    for folder in folders:
        if folder.is_file():
            yield folder
        elif folder.exists():
            yield from sorted(folder.rglob("*.json"))


class Lookup:
    """Formatted forms of every known value, by number of decimals."""

    SCALES = (1, 100, 1e-3, 1e-6, 1e-9, 1 / 60, 1 / 3600)

    def __init__(self, values: Iterable[float]) -> None:
        self.forms: dict[int, set[str]] = {d: set() for d in range(5)}
        for value in values:
            for scale in self.SCALES:
                for v in (value * scale, abs(value * scale)):
                    if abs(v) > 1e12:
                        continue
                    for d in range(5):
                        self.forms[d].add(f"{v:.{d}f}")

    def __contains__(self, token: str) -> bool:
        clean = token.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        return decimals <= 4 and clean in self.forms[decimals]


def prose_lines(tex: str) -> list[tuple[int, str, bool]]:
    """(line number, cleaned text, in the body) for every line of prose."""
    lines = []
    in_body = False
    in_appendix = False
    for number, raw in enumerate(tex.splitlines(), start=1):
        if "\\begin{abstract}" in raw:
            in_body = True
        if raw.startswith("\\bibliography{"):
            in_body = False
        if raw.startswith("\\appendix"):
            in_appendix = True
        if not (in_body or in_appendix):
            continue
        line = re.sub(r"(?<!\\)%.*", "", raw)
        for pattern in STRIP:
            line = re.sub(pattern, " ", line)
        lines.append((number, line, in_body and not in_appendix))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tex", type=Path, default=Path("report/main.tex"))
    parser.add_argument("--all", action="store_true", help="also list the traced numbers")
    args = parser.parse_args(argv)
    facts_path = args.tex.parent / "facts.json"
    if not facts_path.exists():
        print(
            "report/facts.json is missing: run scripts/make_report_figures.py first",
            file=sys.stderr,
        )
        return 2

    tier1: list[float] = list(numbers_in(json.loads(facts_path.read_text(encoding="utf-8"))))
    for table in sorted((args.tex.parent / "tables").glob("*.tex")):
        tier1.extend(tex_numbers(table))
    for config in sorted(Path("configs").glob("*.yaml")):
        tier1.extend(numbers_in(yaml.safe_load(config.read_text(encoding="utf-8"))))
    for path in json_files(RESULTS / f for f in TIER1_RESULT_FOLDERS):
        tier1.extend(numbers_in(json.loads(path.read_text(encoding="utf-8"))))
    body_lookup = Lookup(tier1)
    tier2 = list(tier1)
    for path in json_files([RESULTS]):
        tier2.extend(numbers_in(json.loads(path.read_text(encoding="utf-8"))))
    appendix_lookup = Lookup(tier2)

    tex = args.tex.read_text(encoding="utf-8")
    untraced_body, untraced_appendix, traced = [], [], 0
    for number, line, in_body in prose_lines(tex):
        line = line.replace("{,}", ",")
        for power in POWER.finditer(line):
            token = f"{10.0 ** int(power.group(1)):g}"
            if token in POWERS or token in (body_lookup if in_body else appendix_lookup):
                traced += 1
                continue
            context = line[max(0, power.start() - 50) : power.end() + 30].strip()
            (untraced_body if in_body else untraced_appendix).append((number, token, context))
        line = POWER.sub(" ", line)
        for match in NUMBER.finditer(line):
            token = match.group(0)
            if token in EXTERNAL:
                continue
            lookup = body_lookup if in_body else appendix_lookup
            if token in lookup:
                traced += 1
                if args.all:
                    print(f"  ok   line {number}: {token}")
                continue
            context = line[max(0, match.start() - 50) : match.end() + 30].strip()
            (untraced_body if in_body else untraced_appendix).append((number, token, context))

    print(f"Traced {traced} numbers in the text to the result files.")
    for title, rows in (("Body", untraced_body), ("Appendices", untraced_appendix)):
        if rows:
            print(f"\n{title}: {len(rows)} numbers not traced (read these against the results):")
            for number, token, context in rows:
                print(f"  line {number}: {token:>10s}   ...{context}...")
    print("\nNumbers quoted from other sources or written as settings (not checked):")
    for token, source in {**EXTERNAL, **POWERS}.items():
        print(f"  {token}: {source}")
    return 1 if untraced_body else 0


if __name__ == "__main__":
    raise SystemExit(main())
