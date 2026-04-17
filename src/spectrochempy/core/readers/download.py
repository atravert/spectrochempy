# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""In this module, methods are provided to download external datasets from public database."""

__all__ = ["load_iris", "download_nist_ir"]
__dataset_methods__ = __all__

import tempfile
import time
from enum import Enum
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.read_jcamp import read_jcamp
from spectrochempy.utils._logging import error_
from spectrochempy.utils._logging import info_
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.typeutils import is_iterable


class IndexCheckResult(Enum):
    """Result of checking if a spectrum index is valid."""

    VALID = "valid"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"
    INVALID_CONTENT = "invalid_content"


def _create_session(
    total_retries=3,
    backoff_factor=1.0,
    status_forcelist=(500, 502, 503, 504),
):
    """Create a requests Session with retry logic."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; SpectroChemPy/1.0; +https://www.spectrochempy.com)",  # noqa: E501
            "Accept": "application/x-jcamp-dx, text/html, */*",
        }
    )
    retry_strategy = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _is_valid_jcamp(content: bytes) -> bool:
    """Check if content appears to be valid JCAMP-DX (starts with ##)."""
    if not content:
        return False
    stripped = content.strip()[:50]
    return stripped.startswith(b"##")


def _is_spectrum_not_found(content: bytes) -> bool:
    """Check if the response indicates 'Spectrum not found' (genuine miss)."""
    if not content:
        return True
    text = content[:500].lower()
    return b"spectrum not found" in text or b"no ir spectrum" in text


def _is_rate_limited(content: bytes) -> bool:
    """Check if the response indicates rate limiting."""
    if not content:
        return True
    text = content[:500].lower()
    return b"rate limit" in text or b"request blocked" in text


def _check_index_result(content: bytes) -> IndexCheckResult:
    """
    Classify the response for a spectrum index check.

    Returns
    -------
    IndexCheckResult
        Classification of the response.
    """
    if not content:
        return IndexCheckResult.HTTP_ERROR

    if _is_rate_limited(content):
        return IndexCheckResult.RATE_LIMITED

    if _is_spectrum_not_found(content):
        return IndexCheckResult.NOT_FOUND

    if _is_valid_jcamp(content):
        return IndexCheckResult.VALID

    return IndexCheckResult.INVALID_CONTENT


def _discover_nist_ir_indices(
    session: requests.Session,
    CAS: str,
    max_index: int = 50,
    delay: float = 1.0,
    consecutive_miss_threshold: int = 5,
) -> list[int]:
    """
    Discover all valid spectrum indices for a CAS number.

    Scans indices from 0 to max_index-1, stopping when:
    - consecutive_miss_threshold consecutive genuine misses are found, OR
    - max_index is reached

    Parameters
    ----------
    session : requests.Session
        Session with retry logic configured.
    CAS : str
        CAS number (without dashes).
    max_index : int, optional
        Maximum index to try. Default is 50.
    delay : float, optional
        Delay in seconds between requests. Default is 1.0.
    consecutive_miss_threshold : int, optional
        Number of consecutive genuine misses before stopping. Default is 5.

    Returns
    -------
    list[int]
        List of valid spectrum indices found, in ascending order.
    """
    valid_indices = []
    consecutive_misses = 0
    http_errors = 0
    max_http_errors = 10

    for i in range(max_index):
        url = f"https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C{CAS}&Index={i}&Type=IR"
        try:
            response = session.get(url, timeout=30)
            result = _check_index_result(response.content)
        except requests.RequestException:
            http_errors += 1
            if http_errors >= max_http_errors:
                warning_(
                    f"NIST IR: Too many HTTP errors ({http_errors}), stopping discovery"
                )
                break
            consecutive_misses = 0
            if i < max_index - 1:
                time.sleep(delay)
            continue

        http_errors = 0

        if result == IndexCheckResult.VALID:
            valid_indices.append(i)
            consecutive_misses = 0
        elif result in (IndexCheckResult.NOT_FOUND, IndexCheckResult.INVALID_CONTENT):
            consecutive_misses += 1
            if consecutive_misses >= consecutive_miss_threshold:
                break
        else:
            consecutive_misses = 0

        if i < max_index - 1:
            time.sleep(delay)

    return valid_indices


def _fetch_url(session: requests.Session, url: str, timeout: int = 30) -> bytes:
    """Fetch URL content with the given session."""
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def load_iris():
    """
    Upload the classical "iris" dataset.

    The "IRIS" dataset is a classical example for machine learning.
    It is read from the `scikit-learn` package.

    Returns
    -------
    dataset
        The `IRIS` dataset.

    See Also
    --------
    read : Read data from experimental data.

    """
    from sklearn.datasets import load_iris as sklearn_iris

    data = sklearn_iris()
    coordx = Coord(
        labels=["sepal_length", "sepal width", "petal_length", "petal_width"],
        title="features",
    )
    labels = [data.target_names[i] for i in data.target]
    coordy = Coord(labels=labels, title="samples")
    new = NDDataset(
        data.data,
        coordset=[coordy, coordx],
        title="size",
        name="`IRIS` Dataset",
        units="cm",
    )
    new.history = "Loaded from scikit-learn datasets"
    return new


def download_nist_ir(CAS, index="all", delay=1.0, max_index=50):
    """
    Upload IR spectra from NIST webbook.

    Parameters
    ----------
    CAS : int or str
        the CAS number, can be given as "XXXX-XX-X" (str), "XXXXXXX" (str), XXXXXXX (int)

    index : str or int or tuple of ints
        If set to 'all' (default, import all available spectra for the compound
        corresponding to the index, or a single spectrum, or selected spectra.

    delay : float, optional
        Delay in seconds between requests to NIST server. Default is 1.0.
        Increase this value if getting rate-limited.

    max_index : int, optional
        Maximum index to try when searching for spectra with index="all".
        Default is 50. Prevents excessive requests.

    Returns
    -------
    list of NDDataset or NDDataset
        The dataset(s).

    See Also
    --------
    read : Read data from experimental data.

    """
    info_("download_nist_ir")
    if isinstance(CAS, str) and "-" in CAS:
        CAS = CAS.replace("-", "")

    session = _create_session()

    if index == "all":
        index_list = _discover_nist_ir_indices(
            session, CAS, max_index=max_index, delay=delay
        )

        if len(index_list) == 0:
            error_(IOError, "NIST IR: no spectrum found")
            return None
        if len(index_list) == 1:
            info_("NIST IR: 1 spectrum found")
        else:
            info_(f"NIST IR: {len(index_list)} spectra found")
        index = index_list

    elif isinstance(index, int):
        index = [index]
    elif isinstance(index, str) or not is_iterable(index):
        raise ValueError("index must be 'all', int or iterable of int")

    out = []
    prev_index = None
    for i in index:
        if prev_index is not None and i < prev_index:
            info_(f"Skipping index {i} (non-sequential after {prev_index})")
            continue
        url = f"https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C{CAS}&Index={i}&Type=IR"
        try:
            content = _fetch_url(session, url)
        except requests.RequestException:
            error_(f"Request failed for index {i}")
            continue

        if _is_spectrum_not_found(content) or not _is_valid_jcamp(content):
            error_(
                IOError,
                f"NIST IR: Spectrum {i} does not exist... please check !",
            )
            if i == index[-1] and out == []:
                return None
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jdx", delete=False) as f:
            temp_path = f.name
            try:
                txtdata = content.decode("utf-8")
                f.write(txtdata)
            except UnicodeDecodeError:
                error_("Received non-UTF-8 content from NIST server")
                continue

        try:
            ds = read_jcamp(temp_path)
            Path(temp_path).unlink()
            ds.history[0] = f"Downloaded from NIST: {url}"
            out.append(ds)
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            error_(
                f"Can't read JCAMP file for index {i}: received HTML or invalid data"
            )
            continue

        prev_index = i
        time.sleep(delay)

    if len(out) == 0:
        return None
    if len(out) == 1:
        return out[0]
    return out
