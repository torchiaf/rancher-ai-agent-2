import logging
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, JSONResponse

from ..services.auth import get_user_id_from_token

router = APIRouter(prefix="/v1/api", tags=["chats"])

@router.get("/chats")
async def get_chats(request: Request, sort: str = "createdAt:desc"):
    """
    TODO: add filtering by tags from query parameters.
    Get all threads that have at least one user message and return them as chats list.

    Returns:
        A list of chat objects.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        chats = await request.app.memory_manager.fetch_chats(
            user_id=user_id
        )

        chats = sorted(chats, key=lambda c: c.get("createdAt") or "", reverse=(sort != "createdAt:asc"))

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=chats
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
        
@router.delete("/chats")
async def delete_chats(request: Request) -> JSONResponse:
    """
    Delete all chats for the user.

    Returns:
        A success message and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        await request.app.memory_manager.delete_chats(
            user_id=user_id
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.get("/chats/{chat_id}")
async def get_chat(request: Request, chat_id: str) -> JSONResponse:
    """
    Get a chat by ID.

    Args:
        chat_id: The ID of the thread.

    Returns:
        A chat object and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")

        chat = await request.app.memory_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=chat
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
        
@router.put("/chats/{chat_id}")
async def update_chat(request: Request, chat_id: str, chat_data: dict) -> JSONResponse:
    """
    Update a chat.
    Args:
        chat_id: The ID of the thread.
        chat_data: The chat data to update.
    Returns:
        The updated chat object and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")
        
        if not chat_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_data is required")

        chat = await request.app.memory_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        updated_chat = await request.app.memory_manager.update_chat(
            chat_id=chat_id,
            user_id=user_id,
            chat_data=chat_data
        )
        
        if not updated_chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Failed to update chat, chat not found")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=updated_chat
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.delete("/chats/{chat_id}")
async def delete_chat(request: Request, chat_id: str) -> JSONResponse:
    """
    Delete a specific chat by ID.

    Args:
        chat_id: The ID of the thread.

    Returns:
        A success message and an HTTP status code.
    """

    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")

        await request.app.memory_manager.delete_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.get("/chats/{chat_id}/messages")
async def get_chat_messages(request: Request, chat_id: str) -> JSONResponse:
    """
    TODO: add filtering by tags from query parameters.
    Get messages for a specific thread.

    Args:
        chat_id: The ID of the thread.
    Returns:
        A list of message objects and an HTTP status code.
    """

    try:        
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        if not chat_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")

        chat = await request.app.memory_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        messages = await request.app.memory_manager.fetch_messages(
            chat_id=chat_id,
            user_id=user_id
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=messages
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching messages for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/stats/daily")
async def get_daily_stats(request: Request, days: int = 7) -> JSONResponse:
    """
    Get daily aggregated token usage and cost statistics.

    Args:
        days: Number of days to look back (default 7)

    Returns:
        List of daily statistics with token counts and costs.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        stats = await request.app.memory_manager.get_daily_stats(user_id, days)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"data": stats}
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting daily stats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/stats/agents")
async def get_agent_stats(request: Request) -> JSONResponse:
    """
    Get agent-level statistics including token counts, costs, and averages.

    Returns:
        Dict with per-agent statistics.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        stats = await request.app.memory_manager.get_agent_stats(user_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"data": stats}
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting agent stats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/analytics/costs")
async def get_cost_analytics(request: Request) -> JSONResponse:
    """
    Get cost breakdown by type (input, output, cache read, cache creation).

    Returns:
        Dict with total cost and cost breakdown by type.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        summary = await request.app.memory_manager.get_cost_summary(user_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting cost analytics for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/top/expensive")
async def get_expensive_chats(request: Request, limit: int = 10) -> JSONResponse:
    """
    Get the top most expensive chats ranked by USD cost.

    Args:
        limit: Maximum number of chats to return (default 10)

    Returns:
        List of chat objects with calculated costs, sorted by cost descending.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        chats = await request.app.memory_manager.get_top_expensive_chats(user_id, limit)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"data": chats}
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting expensive chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/analytics")
async def get_analytics_summary(request: Request, period: str = "7d") -> JSONResponse:
    """
    Get comprehensive analytics summary for a time period.

    Args:
        period: Time period in format like "7d", "30d" (default "7d")

    Returns:
        Dict with summary, daily breakdown, and agent breakdown.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        # Parse period string (e.g., "7d" -> 7)
        period_days = 7
        if period.endswith("d"):
            try:
                period_days = int(period[:-1])
            except ValueError:
                period_days = 7

        summary = await request.app.memory_manager.get_analytics_summary(user_id, period_days)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting analytics summary for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


@router.get("/chats/cost")
async def get_cost_summary(request: Request) -> JSONResponse:
    """
    Get quick cost summary for all chats.

    Returns:
        Dict with total USD cost and breakdown by type.
    """
    try:
        user_id = await get_user_id_from_token(request.cookies)
        
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        summary = await request.app.memory_manager.get_cost_summary(user_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting cost summary for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
