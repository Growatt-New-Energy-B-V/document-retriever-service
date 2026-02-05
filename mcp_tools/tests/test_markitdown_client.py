"""Tests for the markitdown client (mocked HTTP)."""
import pytest
import json


def test_markitdown_client_import():
    """Verify the client module can be imported."""
    from server.markitdown_client import MarkitdownClient
    client = MarkitdownClient(base_url="http://localhost:9999")
    assert client.base_url == "http://localhost:9999"
