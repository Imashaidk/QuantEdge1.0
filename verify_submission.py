"""Checks the submission before it is packaged.

Run after run_all.py and after building the PDF:

    python verify_submission.py

It checks that:
  1. every required file is present
  2. every table, figure and number the report uses exists
  3. tables/key_numbers.tex matches the CSV files in results/
  4. the report body is at most 10 pages, not counting the cover and references
  5. no source or text file contains em dashes or emoji
  6. the ZIP, if built, is under 25 MB

Author: Sameera Ekanayaka
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED = [
    "run_all.py",
    "requirements.txt",
    "README.md",
    "docs/AI_DISCLOSURE.md",
    "data/raw_prices.csv",
    "data/log_returns.csv",
    "report/report.tex",
    "report/report.pdf",
    "tables/key_numbers.tex",
]
MAX_BODY_PAGES = 10
ZIP_PATH = ROOT / "dist" / "Nexora_risk_across_tails_and_timescales.zip"


def check_required_files() -> list:
    return [f"missing file: {name}" for name in REQUIRED if not (ROOT / name).exists()]


def check_report_inputs() -> list:
    tex = (ROOT / "report" / "report.tex").read_text(encoding="utf-8")
    problems = []
    for name in re.findall(r"\\input\{\.\./([^}]+)\}", tex):
        if not (ROOT / name).exists():
            problems.append(f"report inputs a missing file: {name}")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (ROOT / "figures" / name).exists():
            problems.append(f"report uses a missing figure: {name}")

    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "tables" / "key_numbers.tex").read_text()))
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", tex)) | set(re.findall(r"\\([A-Z][A-Za-z]+)\b", tex))
    # Only names that look like our macros: capitalised and not standard LaTeX
    standard = {"LARGE", "Large", "Pr", "Phi"}
    for name in sorted(used - defined - standard):
        problems.append(f"report uses an undefined macro: \\{name}")
    return problems


def check_key_numbers() -> list:
    from src.key_numbers import key_numbers_from_results

    written = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (ROOT / "tables" / "key_numbers.tex").read_text()))
    rebuilt = key_numbers_from_results()
    problems = []
    for name, value in rebuilt.items():
        if written.get(name) != value:
            problems.append(f"key number {name}: file says {written.get(name)!r}, results give {value!r}")
    return problems


def check_page_count() -> list:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf is not installed, skipping the page count")
        return []
    reader = PdfReader(str(ROOT / "report" / "report.pdf"))
    first_ref = next((i for i, page in enumerate(reader.pages) if "References" in (page.extract_text() or "")), len(reader.pages))
    # Page 0 is the cover. The page where references start still counts, to be safe.
    body = first_ref
    print(f"  {len(reader.pages)} pages in total, {body} counted (cover and reference-only pages excluded)")
    return [] if body <= MAX_BODY_PAGES else [f"report body is {body} pages, the limit is {MAX_BODY_PAGES}"]


def check_characters() -> list:
    bad = re.compile("[\u2014\U0001F300-\U0001FAFF\u2600-\u27BF]")
    problems = []
    for path in ROOT.rglob("*"):
        if any(part in {".git", ".venv", "venv", "__pycache__", "dist"} for part in path.parts):
            continue
        if path.is_file() and path.suffix in {".py", ".tex", ".md", ".txt"}:
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if bad.search(line):
                    problems.append(f"em dash or emoji in {path.relative_to(ROOT)}:{n}")
    return problems


def check_zip() -> list:
    if not ZIP_PATH.exists():
        print("  ZIP not built yet (python scripts/make_zip.py)")
        return []
    size_mb = ZIP_PATH.stat().st_size / 1024 ** 2
    print(f"  {ZIP_PATH.name}: {size_mb:.1f} MB")
    return [] if size_mb <= 25 else [f"ZIP is {size_mb:.1f} MB, the limit is 25 MB"]


def main() -> int:
    checks = [
        ("Required files", check_required_files),
        ("Report inputs and macros", check_report_inputs),
        ("Key numbers match results/", check_key_numbers),
        ("Report length", check_page_count),
        ("No em dashes or emoji", check_characters),
        ("ZIP size", check_zip),
    ]
    failed = 0
    for title, check in checks:
        print(f"{title}")
        problems = check()
        for p in problems:
            print(f"  FAIL {p}")
        print("  ok" if not problems else "")
        failed += bool(problems)
    print("All checks passed." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
