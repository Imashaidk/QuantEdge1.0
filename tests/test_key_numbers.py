"""Tests for the report macros.

Author: Sameera Ekanayaka
"""

import pytest

from src.key_numbers import write_key_numbers


def test_macros_are_written_one_per_line(tmp_path):
    path = write_key_numbers({"GapTwenty": "+5.0\\%", "SampleDays": "4,835"}, out_dir=tmp_path)
    lines = path.read_text().splitlines()
    assert lines[1] == "\\newcommand{\\GapTwenty}{+5.0\\%}"
    assert lines[2] == "\\newcommand{\\SampleDays}{4,835}"


def test_macro_names_with_digits_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_key_numbers({"Gap20": "1"}, out_dir=tmp_path)
