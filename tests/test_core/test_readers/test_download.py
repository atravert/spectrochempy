# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from unittest import mock

import pytest

VALID_JCAMP = b"""##TITLE=Methanol
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##ORIGIN=NIST
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000
##LASTX=400
##NPOINTS=100
##XFACTOR=1.0
##YFACTOR=0.001
##XYDATA=(X++(Y..Y))
4000 0.1 0.2 0.3
3999 0.15 0.25 0.35
3998 0.2 0.3 0.4
##END=
"""

HTML_NOT_FOUND = b"""<!DOCTYPE html>
<html><body><p>Spectrum not found.</p></body></html>
"""
HTML_GENERIC = b"""<!DOCTYPE html>
<html><body><h1>IR Spectrum</h1></body></html>
"""
RATE_LIMITED = b"""##TITLE=Rate limit exceeded.
##END=
"""
SPECTRUM_NOT_FOUND = b"""##TITLE=Spectrum not found.
##END=
"""


@pytest.fixture
def mock_session():
    """Create a mock requests.Session."""
    with mock.patch(
        "spectrochempy.core.readers.download._create_session"
    ) as mock_create_session:
        mock_session = mock.MagicMock()
        mock_create_session.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_read_jcamp():
    """Mock read_jcamp that returns valid datasets."""
    with mock.patch("spectrochempy.core.readers.download.read_jcamp") as mock_jcamp:

        def make_ds(path):
            ds = mock.MagicMock()
            ds.history = ["Imported from jdx file"]
            return ds

        mock_jcamp.side_effect = make_ds
        yield mock_jcamp


class TestNistIr:
    """Core tests for NIST IR download functionality."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            (VALID_JCAMP, "VALID"),
            (SPECTRUM_NOT_FOUND, "NOT_FOUND"),
            (RATE_LIMITED, "RATE_LIMITED"),
            (HTML_NOT_FOUND, "NOT_FOUND"),
            (HTML_GENERIC, "INVALID_CONTENT"),
            (b"", "HTTP_ERROR"),
        ],
    )
    def test_response_classification(self, content, expected):
        """Test that responses are classified correctly during discovery."""
        from spectrochempy.core.readers.download import _check_index_result

        assert _check_index_result(content).name == expected

    def test_discovery_after_initial_miss(self, mock_session):
        """Test discovery when valid indices come after an initial miss (methanol pattern)."""
        from spectrochempy.core.readers.download import _discover_nist_ir_indices

        def mock_get(url, **kwargs):
            idx = int(url.split("Index=")[1].split("&")[0])
            response = mock.MagicMock()
            response.content = (
                VALID_JCAMP if idx > 0 and idx < 29 else SPECTRUM_NOT_FOUND
            )
            return response

        mock_session.get.side_effect = mock_get
        result = _discover_nist_ir_indices(
            mock_session, "67561", max_index=30, delay=0.01
        )
        assert result == list(range(1, 29))

    def test_discovery_under_rate_limiting(self, mock_session):
        """Test that rate-limiting doesn't prematurely stop discovery."""
        from spectrochempy.core.readers.download import _discover_nist_ir_indices

        def mock_get(url, **kwargs):
            idx = int(url.split("Index=")[1].split("&")[0])
            response = mock.MagicMock()
            if idx == 5:
                response.content = RATE_LIMITED
            elif idx < 5:
                response.content = VALID_JCAMP
            else:
                response.content = SPECTRUM_NOT_FOUND
            return response

        mock_session.get.side_effect = mock_get
        result = _discover_nist_ir_indices(
            mock_session, "67561", max_index=10, delay=0.01
        )
        assert result == [0, 1, 2, 3, 4]

    def test_discovery_under_http_errors(self, mock_session):
        """Test that transient HTTP errors don't crash discovery."""
        import requests

        from spectrochempy.core.readers.download import _discover_nist_ir_indices

        def mock_get(url, **kwargs):
            idx = int(url.split("Index=")[1].split("&")[0])
            if idx == 1:
                raise requests.RequestException("Connection reset")
            response = mock.MagicMock()
            response.content = VALID_JCAMP if idx <= 4 else SPECTRUM_NOT_FOUND
            return response

        mock_session.get.side_effect = mock_get
        result = _discover_nist_ir_indices(
            mock_session, "67561", max_index=10, delay=0.01
        )
        assert result == [0, 2, 3, 4]

    def test_download_single_index(self, mock_session, mock_read_jcamp):
        """Test successful download of a single spectrum."""
        from spectrochempy.core.readers.download import download_nist_ir

        mock_session.get.return_value.content = VALID_JCAMP
        result = download_nist_ir("67-56-1", index=0, delay=0.01)
        assert result is not None
        assert "C67561" in result.history[0]

    def test_download_multiple_indices(self, mock_session, mock_read_jcamp):
        """Test downloading multiple spectra by explicit indices."""
        from spectrochempy.core.readers.download import download_nist_ir

        mock_session.get.return_value.content = VALID_JCAMP
        result = download_nist_ir("67-56-1", index=[0, 1], delay=0.01)
        assert len(result) == 2

    def test_index_all_pipeline(self, mock_session, mock_read_jcamp):
        """Test the full index='all' discovery + download pipeline."""
        from spectrochempy.core.readers.download import download_nist_ir

        def mock_get(url, **kwargs):
            idx = int(url.split("Index=")[1].split("&")[0])
            response = mock.MagicMock()
            response.content = VALID_JCAMP if idx < 3 else SPECTRUM_NOT_FOUND
            return response

        mock_session.get.side_effect = mock_get
        result = download_nist_ir("67561", index="all", delay=0.01, max_index=5)
        assert result is not None
        assert len(result) == 3

    @pytest.mark.parametrize(
        "cas_input,expected_cas",
        [("67-56-1", "C67561"), ("67561", "C67561"), (67561, "C67561")],
    )
    def test_cas_normalization(
        self, mock_session, mock_read_jcamp, cas_input, expected_cas
    ):
        """Test CAS number normalization regardless of input format."""
        from spectrochempy.core.readers.download import download_nist_ir

        mock_session.get.return_value.content = VALID_JCAMP
        download_nist_ir(cas_input, index=0, delay=0.01)
        assert expected_cas in mock_session.get.call_args[0][0]

    def test_invalid_index_type_raises_error(self, mock_session):
        """Test that invalid index types raise ValueError."""
        from spectrochempy.core.readers.download import download_nist_ir

        with pytest.raises(ValueError, match="index must be"):
            download_nist_ir("67561", index="invalid")

    def test_returns_none_for_missing(self, mock_session):
        """Test that None is returned when spectrum is not found."""
        from spectrochempy.core.readers.download import download_nist_ir

        mock_session.get.return_value.content = HTML_NOT_FOUND
        result = download_nist_ir("000-00-0", index=0, delay=0.01)
        assert result is None

    def test_session_has_retry_strategy(self):
        """Test that _create_session creates a session with retry logic."""
        from spectrochempy.core.readers.download import _create_session

        session = _create_session(total_retries=5, backoff_factor=2.0)
        assert session.headers["User-Agent"] is not None
        adapters = list(session.adapters.values())
        assert len(adapters) == 2

    def test_delay_between_requests(self, mock_session, mock_read_jcamp):
        """Test that delay is applied between requests."""
        import time

        from spectrochempy.core.readers.download import download_nist_ir

        mock_session.get.return_value.content = VALID_JCAMP
        start = time.time()
        result = download_nist_ir("67561", index=[0, 1], delay=0.1)
        elapsed = time.time() - start
        assert elapsed >= 0.1
        assert result is not None


class TestNistIrIntegration:
    """Integration tests requiring real NIST server access (skipped by default)."""

    @pytest.mark.skip(reason="Requires network access; may rate-limit NIST server")
    def test_methanol_index_all(self):
        """
        Full integration test: download all methanol spectra.

        Methanol (CAS 67-56-1) has 29 valid spectra at indices 0-28.
        """
        from spectrochempy.core.readers.download import download_nist_ir

        result = download_nist_ir("67-56-1", index="all", delay=2.0, max_index=50)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 29
