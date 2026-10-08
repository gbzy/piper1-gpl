"""Tests for downloading voices."""

import io
import json
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from piper import download_voices

_VOICES_JSON = {
    "de_DE-karlsson-low": {"aliases": ["de-karlsson-low"]},
    "en_US-lessac-medium": {"aliases": []},
    "ja_JP-hi_fi_captain-medium": {"aliases": ["ja_JA-hi_fi_captain-medium"]},
}


class FakeUrlopen:
    """Stand-in for urlopen that serves a fixed set of URLs."""

    def __init__(self, available: Optional[Dict[str, bytes]] = None) -> None:
        """Serve voices.json plus any URL substrings in `available`."""
        self.available = available or {}
        self.requested: list = []

    def __call__(self, url: str):
        """Return a file-like object for the URL, or raise a 404."""
        self.requested.append(url)

        if url == download_voices.VOICES_JSON:
            return io.BytesIO(json.dumps(_VOICES_JSON).encode("utf-8"))

        for url_part, data in self.available.items():
            if url_part in url:
                return io.BytesIO(data)

        raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def clear_voices_cache():
    """Keep the voices.json cache from leaking between tests."""
    download_voices._VOICES_DICT = None  # pylint: disable=protected-access
    yield
    download_voices._VOICES_DICT = None  # pylint: disable=protected-access


def test_download_current_name(tmp_path: Path) -> None:
    """Test that a current voice name downloads without consulting voices.json."""
    fake = FakeUrlopen({"en_US-lessac-medium": b"fake"})
    with patch.object(download_voices, "urlopen", fake):
        download_voices.download_voice("en_US-lessac-medium", tmp_path)

    assert (tmp_path / "en_US-lessac-medium.onnx").read_bytes() == b"fake"
    assert (tmp_path / "en_US-lessac-medium.onnx.json").read_bytes() == b"fake"

    # voices.json is only fetched when a name has to be resolved
    assert download_voices.VOICES_JSON not in fake.requested
    assert "en/en_US/lessac/medium/en_US-lessac-medium.onnx" in fake.requested[0]


def test_download_legacy_name(tmp_path: Path) -> None:
    """Test that a pre-1.0 name is resolved through its alias."""
    fake = FakeUrlopen({"de_DE-karlsson-low": b"fake"})
    with patch.object(download_voices, "urlopen", fake):
        download_voices.download_voice("de-karlsson-low", tmp_path)

    # Files are saved under the current name
    assert (tmp_path / "de_DE-karlsson-low.onnx").exists()
    assert not (tmp_path / "de-karlsson-low.onnx").exists()


def test_download_renamed_voice(tmp_path: Path) -> None:
    """Test that a renamed voice whose old name still parses is resolved."""
    # ja_JA-hi_fi_captain-medium matches VOICE_PATTERN, so it is tried directly
    # first and only resolved after the download 404s.
    fake = FakeUrlopen({"ja_JP-hi_fi_captain-medium": b"fake"})
    with patch.object(download_voices, "urlopen", fake):
        download_voices.download_voice("ja_JA-hi_fi_captain-medium", tmp_path)

    assert (tmp_path / "ja_JP-hi_fi_captain-medium.onnx").exists()
    assert not (tmp_path / "ja_JA-hi_fi_captain-medium.onnx").exists()

    # The old path was tried before voices.json was consulted
    assert "ja/ja_JA/" in fake.requested[0]
    assert download_voices.VOICES_JSON in fake.requested


def test_unknown_name(tmp_path: Path) -> None:
    """Test that an unparseable name with no alias is still an error."""
    fake = FakeUrlopen()
    with patch.object(download_voices, "urlopen", fake):
        with pytest.raises(ValueError):
            download_voices.download_voice("not-a-voice", tmp_path)


def test_missing_voice(tmp_path: Path) -> None:
    """Test that a parseable name with no alias reports the download failure."""
    fake = FakeUrlopen()
    with patch.object(download_voices, "urlopen", fake):
        with pytest.raises(HTTPError):
            download_voices.download_voice("en_US-nonexistent-medium", tmp_path)

    assert not list(tmp_path.iterdir())
