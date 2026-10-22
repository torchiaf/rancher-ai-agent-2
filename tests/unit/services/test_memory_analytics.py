"""Unit tests for analytics methods in app.services.memory"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.memory import MemoryManager, TOKEN_PRICING


@pytest.fixture
def memory_manager():
    """Create a MemoryManager instance for testing."""
    manager = MemoryManager()
    return manager


class TestCalculateCost:
    """Tests for _calculate_cost method."""

    def test_calculate_cost_all_zeros(self, memory_manager):
        """Verify cost calculation with all zero tokens."""
        usage = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
        }
        cost = memory_manager._calculate_cost(usage)
        assert cost == 0.0

    def test_calculate_cost_input_only(self, memory_manager):
        """Verify cost calculation with input tokens only."""
        usage = {
            "inputTokens": 1000,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
        }
        cost = memory_manager._calculate_cost(usage)
        expected = 1000 * TOKEN_PRICING["input"]
        assert cost == expected

    def test_calculate_cost_output_only(self, memory_manager):
        """Verify cost calculation with output tokens only."""
        usage = {
            "inputTokens": 0,
            "outputTokens": 1000,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
        }
        cost = memory_manager._calculate_cost(usage)
        expected = 1000 * TOKEN_PRICING["output"]
        assert cost == expected

    def test_calculate_cost_cache_read_only(self, memory_manager):
        """Verify cost calculation with cache read tokens only."""
        usage = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 1000,
            "cacheCreationTokens": 0,
        }
        cost = memory_manager._calculate_cost(usage)
        expected = 1000 * TOKEN_PRICING["cache_read"]
        assert cost == expected

    def test_calculate_cost_all_types(self, memory_manager):
        """Verify cost calculation with all token types."""
        usage = {
            "inputTokens": 1000,
            "outputTokens": 500,
            "cacheReadTokens": 200,
            "cacheCreationTokens": 300,
        }
        cost = memory_manager._calculate_cost(usage)
        expected = (
            1000 * TOKEN_PRICING["input"]
            + 500 * TOKEN_PRICING["output"]
            + 200 * TOKEN_PRICING["cache_read"]
            + 300 * TOKEN_PRICING["cache_creation"]
        )
        assert abs(cost - expected) < 0.0001  # Float comparison tolerance

    def test_calculate_cost_missing_keys(self, memory_manager):
        """Verify cost calculation with missing keys (should default to 0)."""
        usage = {"inputTokens": 1000}  # Missing other keys
        cost = memory_manager._calculate_cost(usage)
        expected = 1000 * TOKEN_PRICING["input"]
        assert cost == expected


class TestGetDailyStats:
    """Tests for get_daily_stats method."""

    @pytest.mark.asyncio
    async def test_get_daily_stats_empty(self, memory_manager):
        """Verify daily stats with no chats."""
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[])):
            stats = await memory_manager.get_daily_stats("user1", days=7)
            assert stats == []

    @pytest.mark.asyncio
    async def test_get_daily_stats_single_chat(self, memory_manager):
        """Verify daily stats with single chat."""
        now = datetime.now(timezone.utc)
        chat = {
            "id": "chat1",
            "createdAt": now.isoformat(),
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cacheReadTokens": 10,
                "cacheCreationTokens": 5,
            },
        }
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[chat])):
            stats = await memory_manager.get_daily_stats("user1", days=7)
            assert len(stats) == 1
            assert stats[0]["date"] == now.date().isoformat()
            assert stats[0]["inputTokens"] == 100
            assert stats[0]["chatCount"] == 1

    @pytest.mark.asyncio
    async def test_get_daily_stats_multiple_days(self, memory_manager):
        """Verify daily stats aggregates multiple days."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        
        chats = [
            {
                "id": "chat1",
                "createdAt": now.isoformat(),
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "totalTokens": 150,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
            {
                "id": "chat2",
                "createdAt": yesterday.isoformat(),
                "usage": {
                    "inputTokens": 200,
                    "outputTokens": 100,
                    "totalTokens": 300,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
        ]
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=chats)):
            stats = await memory_manager.get_daily_stats("user1", days=7)
            assert len(stats) == 2
            # Should be sorted descending by date
            assert stats[0]["date"] == now.date().isoformat()
            assert stats[1]["date"] == yesterday.date().isoformat()

    @pytest.mark.asyncio
    async def test_get_daily_stats_filters_old_dates(self, memory_manager):
        """Verify daily stats filters out chats older than period."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=10)
        
        chats = [
            {
                "id": "chat1",
                "createdAt": now.isoformat(),
                "usage": {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150, "cacheReadTokens": 0, "cacheCreationTokens": 0},
            },
            {
                "id": "chat2",
                "createdAt": old.isoformat(),
                "usage": {"inputTokens": 200, "outputTokens": 100, "totalTokens": 300, "cacheReadTokens": 0, "cacheCreationTokens": 0},
            },
        ]
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=chats)):
            stats = await memory_manager.get_daily_stats("user1", days=7)
            # Should only include recent chat
            assert len(stats) == 1
            assert stats[0]["date"] == now.date().isoformat()


class TestGetAgentStats:
    """Tests for get_agent_stats method."""

    @pytest.mark.asyncio
    async def test_get_agent_stats_empty(self, memory_manager):
        """Verify agent stats with no chats."""
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[])):
            stats = await memory_manager.get_agent_stats("user1")
            assert stats == {}

    @pytest.mark.asyncio
    async def test_get_agent_stats_single_agent(self, memory_manager):
        """Verify agent stats with single agent."""
        chat = {
            "id": "chat1",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {
                    "supervisor": {
                        "inputTokens": 100,
                        "outputTokens": 50,
                        "totalTokens": 150,
                        "cacheReadTokens": 0,
                        "cacheCreationTokens": 0,
                    }
                },
            },
        }
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[chat])):
            stats = await memory_manager.get_agent_stats("user1")
            assert "supervisor" in stats
            assert stats["supervisor"]["inputTokens"] == 100
            assert stats["supervisor"]["chatCount"] == 1
            assert stats["supervisor"]["avgInputPerChat"] == 100.0
            assert stats["supervisor"]["avgOutputPerChat"] == 50.0

    @pytest.mark.asyncio
    async def test_get_agent_stats_multiple_agents(self, memory_manager):
        """Verify agent stats with multiple agents."""
        chat = {
            "id": "chat1",
            "usage": {
                "inputTokens": 300,
                "outputTokens": 150,
                "totalTokens": 450,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
                "byAgent": {
                    "supervisor": {
                        "inputTokens": 100,
                        "outputTokens": 50,
                        "totalTokens": 150,
                        "cacheReadTokens": 0,
                        "cacheCreationTokens": 0,
                    },
                    "research": {
                        "inputTokens": 200,
                        "outputTokens": 100,
                        "totalTokens": 300,
                        "cacheReadTokens": 0,
                        "cacheCreationTokens": 0,
                    },
                },
            },
        }
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[chat])):
            stats = await memory_manager.get_agent_stats("user1")
            assert len(stats) == 2
            assert "supervisor" in stats
            assert "research" in stats
            assert stats["supervisor"]["inputTokens"] == 100
            assert stats["research"]["inputTokens"] == 200

    @pytest.mark.asyncio
    async def test_get_agent_stats_calculates_averages(self, memory_manager):
        """Verify agent stats calculates correct averages."""
        chats = [
            {
                "id": "chat1",
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "totalTokens": 150,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                    "byAgent": {
                        "supervisor": {
                            "inputTokens": 100,
                            "outputTokens": 50,
                            "totalTokens": 150,
                            "cacheReadTokens": 0,
                            "cacheCreationTokens": 0,
                        }
                    },
                },
            },
            {
                "id": "chat2",
                "usage": {
                    "inputTokens": 200,
                    "outputTokens": 100,
                    "totalTokens": 300,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                    "byAgent": {
                        "supervisor": {
                            "inputTokens": 200,
                            "outputTokens": 100,
                            "totalTokens": 300,
                            "cacheReadTokens": 0,
                            "cacheCreationTokens": 0,
                        }
                    },
                },
            },
        ]
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=chats)):
            stats = await memory_manager.get_agent_stats("user1")
            assert stats["supervisor"]["chatCount"] == 2
            assert stats["supervisor"]["avgInputPerChat"] == 150.0  # (100+200)/2
            assert stats["supervisor"]["avgOutputPerChat"] == 75.0  # (50+100)/2


class TestGetTopExpensiveChats:
    """Tests for get_top_expensive_chats method."""

    @pytest.mark.asyncio
    async def test_get_top_expensive_chats_empty(self, memory_manager):
        """Verify top expensive chats with no chats."""
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[])):
            chats = await memory_manager.get_top_expensive_chats("user1", limit=10)
            assert chats == []

    @pytest.mark.asyncio
    async def test_get_top_expensive_chats_sorting(self, memory_manager):
        """Verify top expensive chats are sorted by cost descending."""
        chats = [
            {
                "id": "chat1",
                "name": "Chat 1",
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
            {
                "id": "chat2",
                "name": "Chat 2",
                "usage": {
                    "inputTokens": 1000,
                    "outputTokens": 500,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
            {
                "id": "chat3",
                "name": "Chat 3",
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
        ]
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=chats)):
            result = await memory_manager.get_top_expensive_chats("user1", limit=10)
            # Should be sorted by cost descending
            assert result[0]["id"] == "chat2"
            assert result[1]["id"] == "chat1"
            assert result[2]["id"] == "chat3"
            # All should have costUSD calculated
            assert all("costUSD" in chat for chat in result)

    @pytest.mark.asyncio
    async def test_get_top_expensive_chats_limit(self, memory_manager):
        """Verify top expensive chats respects limit parameter."""
        chats = [
            {
                "id": f"chat{i}",
                "usage": {
                    "inputTokens": 100 * i,
                    "outputTokens": 50 * i,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            }
            for i in range(1, 6)
        ]
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=chats)):
            result = await memory_manager.get_top_expensive_chats("user1", limit=3)
            assert len(result) == 3


class TestGetAnalyticsSummary:
    """Tests for get_analytics_summary method."""

    @pytest.mark.asyncio
    async def test_get_analytics_summary_empty(self, memory_manager):
        """Verify analytics summary with no chats."""
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[])):
            with patch.object(memory_manager, "get_daily_stats", AsyncMock(return_value=[])):
                with patch.object(memory_manager, "get_agent_stats", AsyncMock(return_value={})):
                    summary = await memory_manager.get_analytics_summary("user1", 7)
                    assert summary["period"] == "7d"
                    assert summary["summary"]["totalChats"] == 0
                    assert summary["summary"]["totalTokens"] == 0
                    assert summary["summary"]["totalCostUSD"] == 0.0

    @pytest.mark.asyncio
    async def test_get_analytics_summary_calculates_totals(self, memory_manager):
        """Verify analytics summary calculates correct totals."""
        chat = {
            "id": "chat1",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cacheReadTokens": 0,
                "cacheCreationTokens": 0,
            },
        }
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[chat])):
            with patch.object(memory_manager, "get_daily_stats", AsyncMock(return_value=[])):
                with patch.object(memory_manager, "get_agent_stats", AsyncMock(return_value={})):
                    summary = await memory_manager.get_analytics_summary("user1", 7)
                    assert summary["summary"]["totalChats"] == 1
                    assert summary["summary"]["totalTokens"] == 150
                    assert summary["summary"]["inputTokens"] == 100
                    assert summary["summary"]["outputTokens"] == 50


class TestGetCostSummary:
    """Tests for get_cost_summary method."""

    @pytest.mark.asyncio
    async def test_get_cost_summary_empty(self, memory_manager):
        """Verify cost summary with no chats."""
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[])):
            summary = await memory_manager.get_cost_summary("user1")
            assert summary["totalCostUSD"] == 0.0
            assert all(cost == 0.0 for cost in summary["breakdown"].values())

    @pytest.mark.asyncio
    async def test_get_cost_summary_breakdown(self, memory_manager):
        """Verify cost summary breakdown calculation."""
        chat = {
            "id": "chat1",
            "usage": {
                "inputTokens": 1000,
                "outputTokens": 500,
                "cacheReadTokens": 200,
                "cacheCreationTokens": 100,
            },
        }
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=[chat])):
            summary = await memory_manager.get_cost_summary("user1")
            
            expected_input = 1000 * TOKEN_PRICING["input"]
            expected_output = 500 * TOKEN_PRICING["output"]
            expected_cache_read = 200 * TOKEN_PRICING["cache_read"]
            expected_cache_creation = 100 * TOKEN_PRICING["cache_creation"]
            
            assert abs(summary["breakdown"]["inputCost"] - expected_input) < 0.0001
            assert abs(summary["breakdown"]["outputCost"] - expected_output) < 0.0001
            assert abs(summary["breakdown"]["cacheReadCost"] - expected_cache_read) < 0.0001
            assert abs(summary["breakdown"]["cacheCreationCost"] - expected_cache_creation) < 0.0001

    @pytest.mark.asyncio
    async def test_get_cost_summary_multiple_chats(self, memory_manager):
        """Verify cost summary aggregates multiple chats."""
        chats = [
            {
                "id": "chat1",
                "usage": {
                    "inputTokens": 1000,
                    "outputTokens": 500,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
            {
                "id": "chat2",
                "usage": {
                    "inputTokens": 500,
                    "outputTokens": 250,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                },
            },
        ]
        with patch.object(memory_manager, "fetch_chats", AsyncMock(return_value=chats)):
            summary = await memory_manager.get_cost_summary("user1")
            
            expected_input = 1500 * TOKEN_PRICING["input"]
            expected_output = 750 * TOKEN_PRICING["output"]
            
            assert abs(summary["breakdown"]["inputCost"] - expected_input) < 0.0001
            assert abs(summary["breakdown"]["outputCost"] - expected_output) < 0.0001
