"""Builds the submission ZIP from the files tracked in git.

Using git's file list keeps out the virtual environment, caches and LaTeX
build files. Run it from the repository root after committing the final PDF:

    python scripts/make_zip.py

Author: Sameera Ekanayaka
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "Nexora_risk_across_tails_and_timescales.zip"
LIMIT_MB = 25.0


def tracked_files() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def main() -> int:
    files = tracked_files()
    if "report/report.pdf" not in files:
        print("report/report.pdf is not committed. Build and commit the PDF first.")
        return 1

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in files:
            zf.write(ROOT / name, arcname=f"QuantEdge1.0/{name}")

    size_mb = OUT.stat().st_size / 1024 ** 2
    print(f"{len(files)} files -> {OUT.relative_to(ROOT)} ({size_mb:.1f} MB)")
    if size_mb > LIMIT_MB:
        print(f"Too big: the limit is {LIMIT_MB:.0f} MB.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
