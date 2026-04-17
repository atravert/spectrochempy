# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import io

import pytest

from spectrochempy import read_jcamp
from spectrochempy.core.readers.read_jcamp import _parse_xydata_format
from spectrochempy.core.readers.read_jcamp import _readl
from spectrochempy.core.readers.read_jcamp import _tokenize_numeric_line


class TestReadl:
    """Test _readl helper for JCAMP header parsing with spaces and multiple =."""

    @pytest.mark.parametrize(
        "line,expected_keyword,expected_text",
        [
            ("##CLASS=IUPAC A", "##CLASS", "IUPAC A"),
            ("##TITLE=Methanol", "##TITLE", "Methanol"),
            ("##END=", "##END", ""),
            ("##XYDATA=(X++(Y..Y))", "##XYDATA", "(X++(Y..Y))"),
            (
                "##ORIGIN=NIST, Analytical Chemistry",
                "##ORIGIN",
                "NIST, Analytical Chemistry",
            ),
            (
                "##$UNCERTAINTY IN Y=2.0 % (B=1.0E-04)",
                "##$UNCERTAINTY IN Y",
                "2.0 % (B=1.0E-04)",
            ),
            ("4000 0.1 0.2 0.3", "", "4000 0.1 0.2 0.3"),
            ("", "EOF", ""),
        ],
        ids=[
            "class_with_space",
            "simple_title",
            "end_keyword",
            "xydata_format",
            "origin_with_comma",
            "multiple_equals",
            "data_line",
            "empty_line",
        ],
    )
    def test_readl_parsing(self, line, expected_keyword, expected_text):
        """Test _readl correctly parses various JCAMP header formats."""
        fid = io.StringIO(line)
        keyword, text = _readl(fid)
        assert keyword == expected_keyword
        assert text == expected_text


class TestNumericTokenizer:
    """Test numeric tokenizer for PAC parsing with adjacent signed values."""

    @pytest.mark.parametrize(
        "line,mode,expected",
        [
            ("4000 0.1 0.2 0.3", "whitespace", ["4000", "0.1", "0.2", "0.3"]),
            ("1452604-144958", "pac", ["1452604", "-144958"]),
            (
                "782.00 20260 1452604-144958",
                "pac",
                ["782.00", "20260", "1452604", "-144958"],
            ),
            ("1 2+3 4+5", "pac", ["1", "2", "+3", "4", "+5"]),
            ("", "pac", []),
            ("1452604-144958", "whitespace", ["1452604-144958"]),
        ],
        ids=[
            "whitespace_separated",
            "pac_adjacent_negative",
            "pac_mixed_adjacent",
            "pac_explicit_plus",
            "empty_line",
            "whitespace_preserves_adjacent",
        ],
    )
    def test_tokenize_numeric_line(self, line, mode, expected):
        """Test _tokenize_numeric_line handles PAC and whitespace modes correctly."""
        tokens = _tokenize_numeric_line(line, mode=mode)
        assert tokens == expected

    @pytest.mark.parametrize(
        "format_spec,expected",
        [
            ("(X++(Y..Y))", "pac"),
            ("(XYDATA)", "whitespace"),
        ],
        ids=["pac_format", "whitespace_format"],
    )
    def test_parse_xydata_format(self, format_spec, expected):
        """Test format specifier detection."""
        assert _parse_xydata_format(format_spec) == expected


class TestReadJcamp:
    """Integration tests for JCAMP parsing with edge cases."""

    def test_jcamp_class_with_space(self):
        """Test JCAMP with ##CLASS=IUPAC A (header with space after =)."""
        jcamp = b"""##TITLE=Methanol
##JCAMP-DX=4.24
##DATA TYPE=INFRARED SPECTRUM
##CLASS=IUPAC A
##ORIGIN=NIST
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000
##LASTX=3998
##NPOINTS=3
##XFACTOR=1.0
##YFACTOR=1.0
##XYDATA=(X++(Y..Y))
4000 0.1
3999 0.15
3998 0.2
##END=
"""
        ds = read_jcamp({"methanol.jdx": jcamp})
        assert ds.name == "Methanol"
        assert ds.shape == (1, 3)

    def test_jcamp_pac_adjacent_negative(self):
        """
        Test PAC-style adjacent negative values are tokenized correctly.

        Reproduces the original NIST issue where lines like:
        1452604-144958 should be parsed as two values: 1452604 and -144958.
        """
        line = "1452604-144958"
        tokens = _tokenize_numeric_line(line, mode="pac")
        assert tokens == ["1452604", "-144958"]
