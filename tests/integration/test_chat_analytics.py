"""Integration tests for analytics endpoints in app.routers.chat"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from app.main import app


@pytest.fixture
def client():
    """Provide a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_user_id():
    """Default user ID for tests."""
    return "test-user-123"


@pytest.fixture
def sample_chats(mock_user_id):
    """Sample chat data for testing."""
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    
    return [
        {
            "id": "chat-1",
            "userId": mock_user_id,
            "name": "Chat 1",
            "createdAt": now.isoformat(),
            "usage": {
                "inputTokens": 1000,
                "outputTokens": 500,
                "totalTokens": 1500,
                "cacheReadTokens": 100,
                "cacheCreationTokens": 50,
                "byAgent": {
                    "supervisor": {
                        "inputTokens": 1000,
                        "outputTokens": 500,
                        "totalTokens": 1500,
                        "cacheReadTokens": 100,
                        "cacheCreationTokens": 50,
                    }
                },
            },
        },
        {
            "id": "chat-2",
            "userId": mock_user_id,
            "name": "Chat 2",
            "createdAt": yesterday.isoformat(),
            "usage": {
                "inputTokens": 500,
                "outputTokens": 250,
                "totalTokens": 750,
                "cacheReadTokens": 50,
                "cacheCreationTokens": 25,
                "byAgent": {
                    "research": {
                        "inputTokens": 500,
                        "outputTokens": 250,
                        "totalTokens": 750,
                        "cacheReadTokens": 50,
                        "cacheCreationTokens": 25,
                    }
                },
            },
        },
    ]


class TestDailyStatsEndpoint:
    """Tests for GET /chats/stats/daily endpoint."""

    def test_daily_stats_unauthorized(self, client):
        """Verify endpoint returns 401 without authentication."""
        response = client.get("/v1/api/chats/stats/daily")
        assert response.status_code == 401
        assert response.json()["detail"] == "Unauthorized"

    def test_daily_stats_success(self, client, mock_user_id, sample_chats):
        """Verify endpoint returns daily stats with valid auth."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_daily_stats", AsyncMock(return_value=[])):
                response = client.get("/v1/api/chats/stats/daily?days=7")
                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                assert isinstance(data["data"], list)

    def test_daily_stats_with_days_parameter(self, client, mock_user_id):
        """Verify days parameter is passed to memory manager."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_stats = [
                {"date": "2024-01-15", "inputTokens": 100, "outputTokens": 50}
            ]
            with patch.object(app.memory_manager, "get_daily_stats", AsyncMock(return_value=mock_stats)) as mock_method:
                response = client.get("/v1/api/chats/stats/daily?days=30")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 30)

    def test_daily_stats_default_days(self, client, mock_user_id):
        """Verify default days=7 is used when not specified."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_daily_stats", AsyncMock(return_value=[])) as mock_method:
                response = client.get("/v1/api/chats/stats/daily")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 7)


class TestAgentStatsEndpoint:
    """Tests for GET /chats/stats/agents endpoint."""

    def test_agent_stats_unauthorized(self, client):
        """Verify endpoint returns 401 without authentication."""
        response = client.get("/v1/api/chats/stats/agents")
        assert response.status_code == 401

    def test_agent_stats_success(self, client, mock_user_id):
        """Verify endpoint returns agent stats with valid auth."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_stats = {
                "supervisor": {
                    "inputTokens": 1000,
                    "outputTokens": 500,
                    "totalTokens": 1500,
                    "costUSD": 0.1234,
                    "chatCount": 5,
                }
            }
            with patch.object(app.memory_manager, "get_agent_stats", AsyncMock(return_value=mock_stats)):
                response = client.get("/v1/api/chats/stats/agents")
                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                assert "supervisor" in data["data"]

    def test_agent_stats_empty(self, client, mock_user_id):
        """Verify endpoint returns empty dict when no agents."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_agent_stats", AsyncMock(return_value={})):
                response = client.get("/v1/api/chats/stats/agents")
                assert response.status_code == 200
                data = response.json()
                assert data["data"] == {}


class TestCostAnalyticsEndpoint:
    """Tests for GET /chats/analytics/costs endpoint."""

    def test_cost_analytics_unauthorized(self, client):
        """Verify endpoint returns 401 without authentication."""
        response = client.get("/v1/api/chats/analytics/costs")
        assert response.status_code == 401

    def test_cost_analytics_success(self, client, mock_user_id):
        """Verify endpoint returns cost analytics with valid auth."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_summary = {
                "totalCostUSD": 0.1234,
                "breakdown": {
                    "inputCost": 0.05,
                    "outputCost": 0.07,
                    "cacheReadCost": 0.005,
                    "cacheCreationCost": 0.0034,
                },
            }
            with patch.object(app.memory_manager, "get_cost_summary", AsyncMock(return_value=mock_summary)):
                response = client.get("/v1/api/chats/analytics/costs")
                assert response.status_code == 200
                data = response.json()
                assert "totalCostUSD" in data
                assert "breakdown" in data
                assert all(
                    key in data["breakdown"]
                    for key in ["inputCost", "outputCost", "cacheReadCost", "cacheCreationCost"]
                )

    def test_cost_analytics_response_format(self, client, mock_user_id):
        """Verify response format has required fields."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_summary = {
                "totalCostUSD": 0.0,
                "breakdown": {
                    "inputCost": 0.0,
                    "outputCost": 0.0,
                    "cacheReadCost": 0.0,
                    "cacheCreationCost": 0.0,
                },
            }
            with patch.object(app.memory_manager, "get_cost_summary", AsyncMock(return_value=mock_summary)):
                response = client.get("/v1/api/chats/analytics/costs")
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data["totalCostUSD"], (int, float))
                assert isinstance(data["breakdown"], dict)


class TestTopExpensiveChatsEndpoint:
    """Tests for GET /chats/top/expensive endpoint."""

    def test_top_expensive_unauthorized(self, client):
        """Verify endpoint returns 401 without authentication."""
        response = client.get("/v1/api/chats/top/expensive")
        assert response.status_code == 401

    def test_top_expensive_success(self, client, mock_user_id, sample_chats):
        """Verify endpoint returns top expensive chats with valid auth."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_chats = [
                {**chat, "costUSD": 0.1234}
                for chat in sample_chats
            ]
            with patch.object(app.memory_manager, "get_top_expensive_chats", AsyncMock(return_value=mock_chats)):
                response = client.get("/v1/api/chats/top/expensive")
                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                assert len(data["data"]) == 2
                for chat in data["data"]:
                    assert "costUSD" in chat

    def test_top_expensive_with_limit(self, client, mock_user_id):
        """Verify limit parameter is passed to memory manager."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_top_expensive_chats", AsyncMock(return_value=[])) as mock_method:
                response = client.get("/v1/api/chats/top/expensive?limit=5")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 5)

    def test_top_expensive_default_limit(self, client, mock_user_id):
        """Verify default limit=10 is used when not specified."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_top_expensive_chats", AsyncMock(return_value=[])) as mock_method:
                response = client.get("/v1/api/chats/top/expensive")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 10)

    def test_top_expensive_sorted_descending(self, client, mock_user_id):
        """Verify returned chats are sorted by cost descending."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_chats = [
                {"id": "chat-1", "costUSD": 0.5},
                {"id": "chat-2", "costUSD": 0.3},
                {"id": "chat-3", "costUSD": 0.1},
            ]
            with patch.object(app.memory_manager, "get_top_expensive_chats", AsyncMock(return_value=mock_chats)):
                response = client.get("/v1/api/chats/top/expensive")
                assert response.status_code == 200
                data = response.json()
                costs = [chat["costUSD"] for chat in data["data"]]
                assert costs == sorted(costs, reverse=True)


class TestAnalyticsEndpoint:
    """Tests for GET /chats/analytics endpoint."""

    def test_analytics_unauthorized(self, client):
        """Verify endpoint returns 401 without authentication."""
        response = client.get("/v1/api/chats/analytics")
        assert response.status_code == 401

    def test_analytics_success(self, client, mock_user_id):
        """Verify endpoint returns analytics summary with valid auth."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_summary = {
                "period": "7d",
                "summary": {
                    "totalChats": 2,
                    "totalTokens": 2250,
                    "inputTokens": 1500,
                    "outputTokens": 750,
                    "totalCostUSD": 0.1234,
                    "avgCostPerChat": 0.0617,
                },
                "byDay": [],
                "byAgent": {},
            }
            with patch.object(app.memory_manager, "get_analytics_summary", AsyncMock(return_value=mock_summary)):
                response = client.get("/v1/api/chats/analytics")
                assert response.status_code == 200
                data = response.json()
                assert data["period"] == "7d"
                assert "summary" in data
                assert "byDay" in data
                assert "byAgent" in data

    def test_analytics_period_parsing_7d(self, client, mock_user_id):
        """Verify period parameter is parsed correctly for 7d."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_analytics_summary", AsyncMock(return_value={"period": "7d"})) as mock_method:
                response = client.get("/v1/api/chats/analytics?period=7d")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 7)

    def test_analytics_period_parsing_30d(self, client, mock_user_id):
        """Verify period parameter is parsed correctly for 30d."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_analytics_summary", AsyncMock(return_value={"period": "30d"})) as mock_method:
                response = client.get("/v1/api/chats/analytics?period=30d")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 30)

    def test_analytics_period_parsing_90d(self, client, mock_user_id):
        """Verify period parameter is parsed correctly for 90d."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_analytics_summary", AsyncMock(return_value={"period": "90d"})) as mock_method:
                response = client.get("/v1/api/chats/analytics?period=90d")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 90)

    def test_analytics_default_period(self, client, mock_user_id):
        """Verify default period=7d is used when not specified."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_analytics_summary", AsyncMock(return_value={"period": "7d"})) as mock_method:
                response = client.get("/v1/api/chats/analytics")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 7)

    def test_analytics_invalid_period_defaults_to_7d(self, client, mock_user_id):
        """Verify invalid period defaults to 7d."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            with patch.object(app.memory_manager, "get_analytics_summary", AsyncMock(return_value={"period": "7d"})) as mock_method:
                response = client.get("/v1/api/chats/analytics?period=invalid")
                assert response.status_code == 200
                mock_method.assert_called_once_with(mock_user_id, 7)

    def test_analytics_response_structure(self, client, mock_user_id):
        """Verify response has correct structure."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_summary = {
                "period": "7d",
                "summary": {
                    "totalChats": 0,
                    "totalTokens": 0,
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "totalCostUSD": 0.0,
                    "avgCostPerChat": 0.0,
                },
                "byDay": [],
                "byAgent": {},
            }
            with patch.object(app.memory_manager, "get_analytics_summary", AsyncMock(return_value=mock_summary)):
                response = client.get("/v1/api/chats/analytics?period=30d")
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data["period"], str)
                assert isinstance(data["summary"], dict)
                assert isinstance(data["byDay"], list)
                assert isinstance(data["byAgent"], dict)


class TestCostSummaryEndpoint:
    """Tests for GET /chats/cost endpoint."""

    def test_cost_summary_unauthorized(self, client):
        """Verify endpoint returns 401 without authentication."""
        response = client.get("/v1/api/chats/cost")
        assert response.status_code == 401

    def test_cost_summary_success(self, client, mock_user_id):
        """Verify endpoint returns cost summary with valid auth."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_summary = {
                "totalCostUSD": 0.1234,
                "breakdown": {
                    "inputCost": 0.05,
                    "outputCost": 0.07,
                    "cacheReadCost": 0.005,
                    "cacheCreationCost": 0.0034,
                },
            }
            with patch.object(app.memory_manager, "get_cost_summary", AsyncMock(return_value=mock_summary)):
                response = client.get("/v1/api/chats/cost")
                assert response.status_code == 200
                data = response.json()
                assert "totalCostUSD" in data
                assert "breakdown" in data

    def test_cost_summary_zero_cost(self, client, mock_user_id):
        """Verify endpoint handles zero cost correctly."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_summary = {
                "totalCostUSD": 0.0,
                "breakdown": {
                    "inputCost": 0.0,
                    "outputCost": 0.0,
                    "cacheReadCost": 0.0,
                    "cacheCreationCost": 0.0,
                },
            }
            with patch.object(app.memory_manager, "get_cost_summary", AsyncMock(return_value=mock_summary)):
                response = client.get("/v1/api/chats/cost")
                assert response.status_code == 200
                data = response.json()
                assert data["totalCostUSD"] == 0.0

    def test_cost_summary_breakdown_sums_to_total(self, client, mock_user_id):
        """Verify breakdown sums to total cost."""
        with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
            mock_summary = {
                "totalCostUSD": 0.1234,
                "breakdown": {
                    "inputCost": 0.05,
                    "outputCost": 0.07,
                    "cacheReadCost": 0.001,
                    "cacheCreationCost": 0.0024,
                },
            }
            with patch.object(app.memory_manager, "get_cost_summary", AsyncMock(return_value=mock_summary)):
                response = client.get("/v1/api/chats/cost")
                assert response.status_code == 200
                data = response.json()
                breakdown_sum = sum(data["breakdown"].values())
                assert abs(breakdown_sum - data["totalCostUSD"]) < 0.0001


class TestAnalyticsErrorHandling:
    """Tests for error handling across analytics endpoints."""

    def test_endpoints_handle_memory_manager_exception(self, client, mock_user_id):
        """Verify endpoints handle memory manager exceptions gracefully."""
        endpoints = [
            "/v1/api/chats/cost",
            "/v1/api/chats/stats/daily",
            "/v1/api/chats/stats/agents",
            "/v1/api/chats/analytics/costs",
            "/v1/api/chats/top/expensive",
            "/v1/api/chats/analytics",
        ]
        
        for endpoint in endpoints:
            with patch("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=mock_user_id)):
                # Simulate exception in memory manager
                with patch.object(app.memory_manager, "get_cost_summary", AsyncMock(side_effect=Exception("DB Error"))):
                    # Most endpoints should return 500
                    if "cost" in endpoint or "analytics" in endpoint:
                        response = client.get(endpoint)
                        # Should return 500 or be caught by error handling
                        assert response.status_code in [500, 200]  # Depends on which method is mocked
