import pytest
import httpx
from unittest.mock import Mock, patch

from microcore.mcp import HeaderAuth  # Replace with actual import path


class TestHeaderAuth:
    """Test suite for HeaderAuth class."""

    def test_init_with_header_name_and_value(self):
        auth = HeaderAuth("Authorization", "Bearer token123")
        assert auth.headers == {"Authorization": "Bearer token123"}

    def test_init_with_dict(self):
        headers = {"Authorization": "Bearer token123", "X-API-Key": "secret"}
        auth = HeaderAuth(headers)
        assert auth.headers == headers

    def test_init_with_dict_and_value_raises_error(self):
        with pytest.raises(ValueError, match="header_value should not be set"):
            HeaderAuth({"Authorization": "Bearer token"}, "extra_value")

    def test_init_with_empty_string_value(self):
        """Test initialization with empty string header value."""
        auth = HeaderAuth("X-Custom-Header", "")
        assert auth.headers == {"X-Custom-Header": ""}

    def test_auth_flow_single_header(self):
        """Test auth flow with single header."""
        auth = HeaderAuth("Authorization", "Bearer token123")
        request = Mock()
        request.headers = {}

        # Execute auth flow
        flow = auth.auth_flow(request)
        result = next(flow)

        assert result is request
        assert request.headers["Authorization"] == "Bearer token123"

    def test_auth_flow_multiple_headers(self):
        """Test auth flow with multiple headers."""
        headers = {"Authorization": "Bearer token", "X-API-Key": "secret"}
        auth = HeaderAuth(headers)
        request = Mock()
        request.headers = {}

        # Execute auth flow
        flow = auth.auth_flow(request)
        result = next(flow)

        assert result is request
        assert request.headers["Authorization"] == "Bearer token"
        assert request.headers["X-API-Key"] == "secret"

    def test_auth_flow_overwrites_existing_headers(self):
        """Test that auth flow overwrites existing headers."""
        auth = HeaderAuth("Authorization", "Bearer new_token")
        request = Mock()
        request.headers = {"Authorization": "Bearer old_token"}

        # Execute auth flow
        flow = auth.auth_flow(request)
        next(flow)

        assert request.headers["Authorization"] == "Bearer new_token"

    @pytest.mark.asyncio
    async def test_integration_with_httpx_client(self):
        """Test integration with httpx.Client (mock test)."""
        auth = HeaderAuth("X-API-Key", "test-key-123")
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_request.return_value = httpx.Response(200)

            async with httpx.AsyncClient(auth=auth) as client:
                await client.get("https://example.com")
            assert mock_request.called
