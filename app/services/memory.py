import os
import re
import logging
from enum import Enum
from typing import cast
from datetime import datetime, timezone, timedelta
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import CheckpointMetadata, CheckpointTuple
from langchain_core.runnables.config import RunnableConfig

from ..services.agent.loader import DEFAULT_AGENT_NAME
from ..constants import CONTEXT_PARAMETERS_SUFFIX

# Token pricing in USD per token (adjust based on your LLM provider)
TOKEN_PRICING = {
    "input": 0.0005,           # $0.0005 per input token
    "output": 0.0015,          # $0.0015 per output token
    "cache_read": 0.00005,     # $0.00005 per cache read token (90% discount)
    "cache_creation": 0.0015,  # $0.0015 per cache creation token (same as output)
}

class StorageType(str, Enum):
    """Enum for different types of memory storage."""
    IN_MEMORY = "in-memory"
    POSTGRES = "postgres"

class MemoryManager:
    """
    Manages chat memories using a database or in-memory storage.
    """
    def __init__(self):
        self.db_enabled = os.environ.get("DB_ENABLED", "false").lower() == "true"
        self.db_url = os.environ.get("DB_CONNECTION_STRING", "")
        self.db_pool: AsyncConnectionPool | None = None
        self.checkpointer: AsyncPostgresSaver | InMemorySaver | None = None
        self.storage_type: StorageType | None = None

    async def initialize(self):
        if os.environ.get("DB_ENABLED", "false").lower() == "true":
            logging.info("Initializing PostgreSQL Checkpointer...")

            self.db_pool = AsyncConnectionPool(
                conninfo=self.db_url, 
                max_size=20,
                kwargs={"autocommit": True},
                open=False
            )
            
            await self.db_pool.open()

            self.checkpointer = AsyncPostgresSaver(self.db_pool)  # type: ignore[arg-type]

            await self.checkpointer.setup()
            self.storage_type = StorageType.POSTGRES

            logging.info("Using PostgreSQL Checkpointer for checkpoints.")
        else:
            self.checkpointer = InMemorySaver()
            self.storage_type = StorageType.IN_MEMORY
            
            logging.info("Using InMemorySaver for checkpoints.")
                        
    async def destroy(self):
        if self.db_pool:
            await self.db_pool.close()
            logging.info("PostgreSQL pool closed.")

    def get_checkpointer(self) -> AsyncPostgresSaver | InMemorySaver:
        if not self.checkpointer:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")
        return self.checkpointer

    def _filter_by_tags(self, tag_filters: list[dict], tags: list, role: str) -> bool:
        """
        TODO: add custom tag filtering logic here.
        Check if any of the tags are in the excluded tags list.
        
        Args:
            tag_filters: list of tag filter dicts
            tags: list of tags to check
            role: "human" or "ai"
        Returns:
            bool: True if none of the tags are in the excluded tags, False otherwise.
        """
        for tag in tags:
            for excluded in tag_filters:
                for tag_role, excluded_tags in excluded.items():
                    if tag_role == role and tag in excluded_tags:
                        return False
        return True
    
    def _is_empty_chat(self, checkpoint_tuple: CheckpointTuple, tag_filters: list[dict] = []) -> bool:
        """
        Check if a chat is empty from the checkpoint tuple.
        A chat is considered empty if it does not contain any messages besides tag-filtered messages.
        
        Example:
            tag_filters = [
                { "human": ["welcome"] },
                { "ai": ["welcome"] },
            ]
        """        
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        agent_metatdata = channel_values.get("agent_metadata", {})
        messages = channel_values.get("messages", [])
        tags = agent_metatdata.get("tags", [])

        is_empty = True
        for msg in messages:
            if self._filter_by_tags(tag_filters, tags, msg.type):
                is_empty = False
                break
            
        return is_empty
    
    def _get_chat_metadata(self, checkpointTuple: CheckpointTuple) -> dict:
        """
        Extract chat metadata from a checkpoint tuple.
        """
        metadata = checkpointTuple.metadata
        channel_values = checkpointTuple.checkpoint.get("channel_values", {})

        name = metadata.get("chat_name", "")
        created_at = None
        
        messages = channel_values.get("messages_history", [])
        if messages and len(messages) > 0:
            # First message is used for chat metadata
            message = messages[0]

            additional_kwargs = message.additional_kwargs
            created_at = additional_kwargs.get("created_at") if additional_kwargs else None

            if not name:
                request_metadata = message.additional_kwargs.get("request_metadata", {})

                user_input = request_metadata.get("user_input", "")
                labels = request_metadata.get("labels", {})

                summary_label = labels.get("summary", "")
                if summary_label:
                    # Use summary label as chat name if available, sanitizing it by removing any HTML tags
                    name = re.sub(r'<[^>]+>', '', summary_label)
                else:
                    # Use content of first User message as default chat name
                    name = user_input if user_input else "Untitled Chat"

        return {
            "name": name,
            "created_at": created_at
        }

    async def fetch_chats(self, user_id: str, filters: dict = {}) -> list:
        """
        TODO: implement filtering logic.
        Fetch chat threads from the database for a specific user.

        Args:
            user_id: The ID of the user.
            filters: A dictionary to filter chat threads.

        Returns:
            A list of chat thread records.
        """
        rows = []

        chat_ids = set()
        async for checkpoint_tuple in self.get_checkpointer().alist(config=None, filter={"user_id": user_id}):
            chat_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id", "")
            user_id = checkpoint_tuple.metadata.get("user_id", "")
            
            if not chat_id or not user_id:
                continue

            if "::child::" in chat_id or "::planner::" in chat_id: # skip child and plan threads
                continue

            if self._is_empty_chat(checkpoint_tuple):
                logging.debug(f"Chat_id: {chat_id}, user_id: {user_id} is empty, skipping")
                continue
            
            logging.debug(f"Processing checkpoint_tuple for chat_id: {chat_id}, user_id: {user_id}")

            if chat_id not in chat_ids:
                chat_ids.add(chat_id)
                
                chat_metadata = self._get_chat_metadata(checkpoint_tuple)
                rows.append({
                    "id": chat_id,
                    "userId": user_id,
                    "name": chat_metadata.get("name"),
                    "createdAt": chat_metadata.get("created_at")
                })

        return rows
    
    async def delete_chats(self, user_id: str) -> None:
        """
        Delete all chat threads for a specific user.

        Args:
            user_id: The ID of the user.
        """
        thread_ids = []
        async for checkpoint_tuple in self.get_checkpointer().alist(config=None, filter={"user_id": user_id}):
            thread_id = checkpoint_tuple.config.get('configurable', {}).get('thread_id', '')
            if thread_id not in thread_ids:
                thread_ids.append(thread_id)
        
        # Then delete them after iteration is complete
        for thread_id in thread_ids:
            await self.get_checkpointer().adelete_thread(thread_id)
            logging.debug(f"Deleted thread: {thread_id}, user_id: {user_id}")

    async def _aggregate_token_usage(self, chat_id: str, user_id: str) -> dict:
        """Aggregate token usage for a conversation, broken down per source.

        Sums the ``token_usage`` state channel across the main thread (supervisor)
        and all child-agent sub-threads (``chat_id::child::<name>``). Entries are
        bucketed per source: ``ui-tools`` and ``summarization`` get their own
        buckets; agent-turn entries are bucketed as ``supervisor`` (main thread) or
        the child agent name (child threads).

        Returns grand totals plus a ``byAgent`` breakdown, all camelCase.
        """
        fields = ("input_tokens", "output_tokens", "total_tokens", "cache_read", "cache_creation")
        field_map = {
            "input_tokens": "inputTokens",
            "output_tokens": "outputTokens",
            "total_tokens": "totalTokens",
            "cache_read": "cacheReadTokens",
            "cache_creation": "cacheCreationTokens",
        }

        def _empty() -> dict:
            return {out: 0 for out in field_map.values()}

        totals = _empty()
        by_agent: dict[str, dict] = {}
        child_prefix = f"{chat_id}::child::"
        seen_threads: set[str] = set()

        async for checkpoint_tuple in self.get_checkpointer().alist(config=None, filter={"user_id": user_id}):
            thread_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id", "")

            if thread_id != chat_id and not thread_id.startswith(child_prefix):
                continue

            # alist yields newest first; the first checkpoint per thread carries the
            # complete (cumulative) token_usage channel.
            if thread_id in seen_threads:
                continue
            seen_threads.add(thread_id)

            token_usage = checkpoint_tuple.checkpoint.get("channel_values", {}).get("token_usage", {})
            if not token_usage:
                continue

            for entry in token_usage.values():
                source = entry.get("source", "agent")
                if source in ("ui-tools", "summarization"):
                    bucket = source
                elif thread_id == chat_id:
                    bucket = "supervisor"
                else:
                    bucket = thread_id.split("::child::", 1)[1]

                agent_totals = by_agent.setdefault(bucket, _empty())
                for field in fields:
                    value = entry.get(field, 0)
                    agent_totals[field_map[field]] += value
                    totals[field_map[field]] += value

        return {**totals, "byAgent": by_agent}

    def _calculate_cost(self, usage: dict) -> float:
        """Calculate USD cost from token usage dict.
        
        Args:
            usage: Token usage dict with inputTokens, outputTokens, cacheReadTokens, cacheCreationTokens
            
        Returns:
            Cost in USD as float
        """
        cost = 0.0
        cost += usage.get("inputTokens", 0) * TOKEN_PRICING["input"]
        cost += usage.get("outputTokens", 0) * TOKEN_PRICING["output"]
        cost += usage.get("cacheReadTokens", 0) * TOKEN_PRICING["cache_read"]
        cost += usage.get("cacheCreationTokens", 0) * TOKEN_PRICING["cache_creation"]
        return cost

    async def get_daily_stats(self, user_id: str, days: int = 7) -> list[dict]:
        """Get aggregated token usage statistics grouped by date.
        
        Args:
            user_id: The ID of the user
            days: Number of days to look back (default 7)
            
        Returns:
            List of dicts with date, tokens, and cost breakdown
        """
        daily_stats: dict[str, dict] = {}
        
        # Get all chats for the user
        chats = await self.fetch_chats(user_id)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        for chat in chats:
            created_at = chat.get("createdAt")
            if not created_at:
                continue
                
            # Parse ISO format datetime
            try:
                if isinstance(created_at, str):
                    chat_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    chat_date = created_at
                    
                if chat_date < cutoff_date:
                    continue
                    
                date_key = chat_date.date().isoformat()
            except (ValueError, AttributeError):
                continue
            
            usage = chat.get("usage", {})
            if not usage:
                continue
                
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "date": date_key,
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "totalTokens": 0,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                    "costUSD": 0.0,
                    "chatCount": 0,
                }
            
            daily_stats[date_key]["inputTokens"] += usage.get("inputTokens", 0)
            daily_stats[date_key]["outputTokens"] += usage.get("outputTokens", 0)
            daily_stats[date_key]["totalTokens"] += usage.get("totalTokens", 0)
            daily_stats[date_key]["cacheReadTokens"] += usage.get("cacheReadTokens", 0)
            daily_stats[date_key]["cacheCreationTokens"] += usage.get("cacheCreationTokens", 0)
            daily_stats[date_key]["chatCount"] += 1
        
        # Calculate costs for each day
        for day_stat in daily_stats.values():
            day_stat["costUSD"] = self._calculate_cost(day_stat)
        
        # Sort by date descending
        return sorted(daily_stats.values(), key=lambda x: x["date"], reverse=True)

    async def get_agent_stats(self, user_id: str) -> dict[str, dict]:
        """Get aggregated statistics broken down by agent.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dict with agent names as keys and their stats as values
        """
        agent_stats: dict[str, dict] = {}
        chats = await self.fetch_chats(user_id)
        
        for chat in chats:
            usage = chat.get("usage", {})
            if not usage:
                continue
                
            by_agent = usage.get("byAgent", {})
            for agent_name, agent_usage in by_agent.items():
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {
                        "agent": agent_name,
                        "inputTokens": 0,
                        "outputTokens": 0,
                        "totalTokens": 0,
                        "cacheReadTokens": 0,
                        "cacheCreationTokens": 0,
                        "costUSD": 0.0,
                        "chatCount": 0,
                        "avgInputPerChat": 0.0,
                        "avgOutputPerChat": 0.0,
                    }
                
                agent_stats[agent_name]["inputTokens"] += agent_usage.get("inputTokens", 0)
                agent_stats[agent_name]["outputTokens"] += agent_usage.get("outputTokens", 0)
                agent_stats[agent_name]["totalTokens"] += agent_usage.get("totalTokens", 0)
                agent_stats[agent_name]["cacheReadTokens"] += agent_usage.get("cacheReadTokens", 0)
                agent_stats[agent_name]["cacheCreationTokens"] += agent_usage.get("cacheCreationTokens", 0)
                agent_stats[agent_name]["chatCount"] += 1
        
        # Calculate costs and averages
        for agent_stat in agent_stats.values():
            agent_stat["costUSD"] = self._calculate_cost(agent_stat)
            chat_count = agent_stat["chatCount"]
            if chat_count > 0:
                agent_stat["avgInputPerChat"] = agent_stat["inputTokens"] / chat_count
                agent_stat["avgOutputPerChat"] = agent_stat["outputTokens"] / chat_count
        
        return agent_stats

    async def get_top_expensive_chats(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get the most expensive chats ranked by USD cost.
        
        Args:
            user_id: The ID of the user
            limit: Max number of chats to return (default 10)
            
        Returns:
            Sorted list of chat objects with cost calculated
        """
        chats = await self.fetch_chats(user_id)
        
        # Calculate cost for each chat
        for chat in chats:
            usage = chat.get("usage", {})
            chat["costUSD"] = self._calculate_cost(usage)
        
        # Sort by cost descending and limit
        return sorted(chats, key=lambda x: x["costUSD"], reverse=True)[:limit]

    async def get_analytics_summary(self, user_id: str, period_days: int = 7) -> dict:
        """Get a comprehensive analytics summary for a period.
        
        Args:
            user_id: The ID of the user
            period_days: Number of days to aggregate (default 7)
            
        Returns:
            Dict with overall stats, daily breakdown, and agent breakdown
        """
        daily_stats = await self.get_daily_stats(user_id, period_days)
        agent_stats = await self.get_agent_stats(user_id)
        chats = await self.fetch_chats(user_id)
        
        # Calculate totals
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost = 0.0
        
        for chat in chats:
            usage = chat.get("usage", {})
            total_input += usage.get("inputTokens", 0)
            total_output += usage.get("outputTokens", 0)
            total_tokens += usage.get("totalTokens", 0)
            total_cost += self._calculate_cost(usage)
        
        return {
            "period": f"{period_days}d",
            "summary": {
                "totalChats": len(chats),
                "totalTokens": total_tokens,
                "inputTokens": total_input,
                "outputTokens": total_output,
                "totalCostUSD": round(total_cost, 4),
                "avgCostPerChat": round(total_cost / len(chats), 4) if chats else 0.0,
            },
            "byDay": daily_stats,
            "byAgent": agent_stats,
        }

    async def get_cost_summary(self, user_id: str) -> dict:
        """Get a quick cost summary for all chats.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dict with total cost breakdown
        """
        chats = await self.fetch_chats(user_id)
        
        total_cost = 0.0
        cost_breakdown = {
            "inputCost": 0.0,
            "outputCost": 0.0,
            "cacheReadCost": 0.0,
            "cacheCreationCost": 0.0,
        }
        
        for chat in chats:
            usage = chat.get("usage", {})
            cost_breakdown["inputCost"] += usage.get("inputTokens", 0) * TOKEN_PRICING["input"]
            cost_breakdown["outputCost"] += usage.get("outputTokens", 0) * TOKEN_PRICING["output"]
            cost_breakdown["cacheReadCost"] += usage.get("cacheReadTokens", 0) * TOKEN_PRICING["cache_read"]
            cost_breakdown["cacheCreationCost"] += usage.get("cacheCreationTokens", 0) * TOKEN_PRICING["cache_creation"]
        
        total_cost = sum(cost_breakdown.values())
        
        return {
            "totalCostUSD": round(total_cost, 4),
            "breakdown": {k: round(v, 4) for k, v in cost_breakdown.items()},
        }

    async def fetch_chat(self, chat_id: str, user_id: str) -> dict | None:
        """
        Fetch a specific chat thread from the database.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
        Returns:
            A chat thread record or None if not found.
        """
        config = RunnableConfig(configurable={"thread_id": chat_id, "user_id": user_id})
        checkpoint_tuple = await self.get_checkpointer().aget_tuple(config=config)

        if checkpoint_tuple:
            logging.debug(f"Found checkpoint_tuple for chat_id: {chat_id}, user_id: {user_id}")

            if self._is_empty_chat(checkpoint_tuple):
                logging.debug(f"Chat_id: {chat_id}, user_id: {user_id} is empty, skipping")
                return None

            chat_metadata = self._get_chat_metadata(checkpoint_tuple)
            chat = {
                "id": chat_id,
                "userId": user_id,
                "name": chat_metadata.get("name"),
                "createdAt": chat_metadata.get("created_at"),
                "usage": await self._aggregate_token_usage(chat_id, user_id),
            }
            return chat

        return None

    async def update_chat(self, chat_id: str, user_id: str, chat_data: dict) -> dict | None:
        """
        Update a specific chat thread for a specific user.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
            chat_data: The chat data to update.
        Returns:
            The updated chat thread record.
        """
        name = chat_data.get("name")
        
        if name and isinstance(name, str) and len(name) > 0:
            config = RunnableConfig(configurable={ "thread_id": chat_id, "user_id": user_id, "checkpoint_ns": ""})
            
            checkpoint_tuple = await self.get_checkpointer().aget_tuple(config)
            
            if checkpoint_tuple:
                current_metadata = dict(checkpoint_tuple.metadata) if checkpoint_tuple.metadata else {}

                new_metadata = {**current_metadata, "chat_name": name}

                await self.get_checkpointer().aput(
                    config=config,
                    checkpoint=checkpoint_tuple.checkpoint,
                    metadata=cast(CheckpointMetadata, new_metadata),
                    new_versions={}
                )

        chat = await self.fetch_chat(chat_id, user_id)

        return chat
    
    async def delete_chat(self, chat_id: str, user_id: str) -> None:
        """
        Delete a specific chat thread for a specific user.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
        """
        config = RunnableConfig(configurable={"thread_id": chat_id})
        checkpoint_tuple = await self.get_checkpointer().aget_tuple(config=config)

        if checkpoint_tuple and checkpoint_tuple.metadata.get("user_id") == user_id:
            await self.get_checkpointer().adelete_thread(chat_id)
            logging.debug(f"Deleted thread: {chat_id}, user_id: {user_id}")

    async def fetch_messages(self, chat_id: str, user_id: str, filters: dict = {}) -> list:
        """
        Fetch messages from the database based on filter configuration.
        Supports filtering by tags and limiting results.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
            filters: A dictionary to filter messages with keys:
                - tag_filters: list of {role: [tags]} dicts to exclude
                - limit: max number of messages to return
        Returns:
            A list of message records filtered by tag_filters and limit.
            
        Filters example:
            filters = {
                "tag_filters": [
                    { "human": ["welcome"] },
                    { "ai": ["welcome"] },
                ],
                "limit": 100
            }
        """
        limit = filters.get("limit")
        tag_filters = filters.get("tag_filters", [])

        rows = []
        
        config = RunnableConfig(configurable={"thread_id": chat_id, "user_id": user_id})
        checkpoint_tuple = await self.get_checkpointer().aget_tuple(config=config)

        if not checkpoint_tuple:
            return rows

        all_messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages_history", [])

        for message in all_messages:
            # Apply tag filtering if configured
            if tag_filters and message.type in ("human", "ai"):
                request_metadata = message.additional_kwargs.get("request_metadata", {})
                tags = request_metadata.get("tags", [])
                # Skip this message if it matches any tag filter
                if not self._filter_by_tags(tag_filters, tags, message.type):
                    continue
            
            if message.type == 'human' and message.content != "":
                request_metadata = message.additional_kwargs.get("request_metadata", {})
                tags = request_metadata.get("tags", [])

                rows.append({
                    "chatId": chat_id,
                    "role": "user",
                    "agent": request_metadata.get("agent", None),
                    "message": message.content.split(CONTEXT_PARAMETERS_SUFFIX)[0],
                    "context": request_metadata.get("context", None),
                    "labels": request_metadata.get("labels", None),
                    "tags": tags,
                    "createdAt": message.additional_kwargs.get("created_at"),
                })
            elif message.type == 'ai' and (message.content != "" or message.additional_kwargs.get("stopped")):
                request_metadata = message.additional_kwargs.get("request_metadata", {})

                # A stopped message is emitted as two rows (partial text, then an empty marker with
                # "stopped") so the UI gets the same two messages as the live stream. Written by
                # _record_stopped_in_history in routers/websocket.py. Keep both in sync.
                if message.content != "":
                    rows.append({
                        "chatId": chat_id,
                        "role": "agent",
                        "agent": request_metadata.get("agent", None),
                        "message": (message.additional_kwargs.get("mcp_response") or "") + str(message.text),
                        "tools": message.additional_kwargs.get("ui_tools", []),
                        "tags": request_metadata.get("tags", []),
                        "createdAt": message.additional_kwargs.get("created_at"),
                    })

                if message.additional_kwargs.get("stopped"):
                    rows.append({
                        "chatId": chat_id,
                        "role": "agent",
                        "agent": request_metadata.get("agent", None),
                        "message": "",
                        "tools": [],
                        "tags": request_metadata.get("tags", []),
                        "stopped": message.additional_kwargs.get("stopped", ""),
                        "createdAt": message.additional_kwargs.get("created_at"),
                    })
            elif message.type == 'tool' and message.additional_kwargs.get("interrupt_message", "") != "":
                rows.append({
                    "chatId": chat_id,
                    "role": "agent",
                    "agent": message.additional_kwargs.get("selected_agent", ""),
                    "message": message.additional_kwargs.get("interrupt_message", ""),
                    "confirmation": message.additional_kwargs.get("confirmation", None),
                    "tools": message.additional_kwargs.get("ui_tools", []) or (getattr(message, "artifact", None) or {}).get("ui_tools", []),
                    "createdAt": message.additional_kwargs.get("created_at"),
                })
            

            if limit and len(rows) >= limit:
                return rows[:limit]

        return rows

async def create_memory_manager() -> MemoryManager:
    """
    Factory function to create a MemoryManager instance.

    Returns:
        An instance of MemoryManager.
    """
    
    manager = MemoryManager()
    await manager.initialize()

    logging.info("MemoryManager created and initialized")
    return manager
