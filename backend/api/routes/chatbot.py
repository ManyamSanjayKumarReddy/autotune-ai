# api/routes/chatbot.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime

from db.models import ChatSession, ChatMessage
from agent_builder.use_case_call import get_response as simple_response

router = APIRouter()


# ============== REQUEST/RESPONSE MODELS ==============

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None



class FeedbackRequest(BaseModel):
    message_id: str
    liked: bool  # True for like, False for dislike


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    liked: Optional[bool]
    created_at: datetime


class SessionResponse(BaseModel):
    session_id: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool


# ============== CHAT ENDPOINTS ==============

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a message and get AI response.
    Creates new session if session_id not provided.
    """
    try:
        # Get or create session
        if request.session_id:
            session = await ChatSession.get_or_none(session_id=request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            # Create new session
            session_id = f"session-{uuid.uuid4().hex[:8]}"
            session = await ChatSession.create(session_id=session_id)

        # Generate unique message IDs
        user_msg_id = f"msg-{uuid.uuid4().hex[:12]}"
        ai_msg_id = f"msg-{uuid.uuid4().hex[:12]}"

        # Save user message
        user_message = await ChatMessage.create(
            message_id=user_msg_id,
            session=session,
            role="user",
            content=request.query,
            parent_message_id=None
        )

        # Get AI response
        ai_content = simple_response(request.query)
        model_name = "Qwen 1.5B"

        # Save AI message
        ai_message = await ChatMessage.create(
            message_id=ai_msg_id,
            session=session,
            role="assistant",
            content=ai_content,
            parent_message_id=user_msg_id,  # Link to user query
            model_used=model_name,
            uses_rag=False
        )

        return {
            "session_id": session.session_id,
            "user_message": {
                "message_id": user_message.message_id,
                "content": user_message.content,
                "created_at": user_message.created_at
            },
            "ai_message": {
                "message_id": ai_message.message_id,
                "content": ai_message.content,
                "model_used": model_name,
                "uses_rag": False,
                "created_at": ai_message.created_at
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def add_feedback(request: FeedbackRequest):
    """
    Like or dislike a message.
    """
    try:
        message = await ChatMessage.get_or_none(message_id=request.message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        message.liked = request.liked
        await message.save()

        return {
            "message_id": message.message_id,
            "liked": message.liked,
            "feedback": "👍 Liked" if request.liked else "👎 Disliked"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get full chat history for a session.
    """
    try:
        session = await ChatSession.get_or_none(session_id=session_id).prefetch_related('messages')
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = await ChatMessage.filter(session=session).order_by('created_at')

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "message_count": len(messages),
            "messages": [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "parent_message_id": msg.parent_message_id,
                    "liked": msg.liked,
                    "model_used": msg.model_used,
                    "created_at": msg.created_at
                }
                for msg in messages
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(limit: int = 20, active_only: bool = True):
    """
    List all chat sessions.
    """
    try:
        query = ChatSession.all()
        if active_only:
            query = query.filter(is_active=True)

        sessions = await query.order_by('-updated_at').limit(limit)

        result = []
        for session in sessions:
            message_count = await ChatMessage.filter(session=session).count()
            result.append({
                "session_id": session.session_id,
                "message_count": message_count,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "is_active": session.is_active
            })

        return {
            "total": len(result),
            "sessions": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """
    Mark session as inactive and create new session.
    Returns new session_id.
    """
    try:
        # Mark old session as inactive
        old_session = await ChatSession.get_or_none(session_id=session_id)
        if old_session:
            old_session.is_active = False
            await old_session.save()

        # Create new session
        new_session_id = f"session-{uuid.uuid4().hex[:8]}"
        new_session = await ChatSession.create(session_id=new_session_id)

        return {
            "message": "Session reset successfully",
            "old_session_id": session_id if old_session else None,
            "new_session_id": new_session.session_id,
            "created_at": new_session.created_at
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and all its messages.
    """
    try:
        session = await ChatSession.get_or_none(session_id=session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Delete all messages
        deleted_messages = await ChatMessage.filter(session=session).delete()

        # Delete session
        await session.delete()

        return {
            "message": "Session deleted successfully",
            "session_id": session_id,
            "messages_deleted": deleted_messages
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/disliked")
async def get_disliked_messages(limit: int = 50):
    """
    Get all disliked messages for training data generation.
    """
    try:
        messages = await ChatMessage.filter(
            liked=False,
            role="assistant"
        ).order_by('-created_at').limit(limit)

        result = []
        for msg in messages:
            # Get parent user message
            user_msg = await ChatMessage.get_or_none(message_id=msg.parent_message_id)

            result.append({
                "ai_message_id": msg.message_id,
                "user_query": user_msg.content if user_msg else None,
                "ai_response": msg.content,
                "model_used": msg.model_used,
                "session_id": (await msg.session).session_id,
                "created_at": msg.created_at
            })

        return {
            "total": len(result),
            "disliked_messages": result,
            "use_case": "These can be used to generate training data via AutoTune pipeline"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """
    Get chat statistics.
    """
    try:
        total_sessions = await ChatSession.all().count()
        active_sessions = await ChatSession.filter(is_active=True).count()
        total_messages = await ChatMessage.all().count()
        liked_messages = await ChatMessage.filter(liked=True).count()
        disliked_messages = await ChatMessage.filter(liked=False).count()

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "user_messages": await ChatMessage.filter(role="user").count(),
            "ai_messages": await ChatMessage.filter(role="assistant").count(),
            "feedback": {
                "liked": liked_messages,
                "disliked": disliked_messages,
                "no_feedback": total_messages - liked_messages - disliked_messages
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add these to the bottom of api/routes/chatbot.py


@router.get("/sample-data")
async def get_sample_data():
    """
    Get top 5 disliked conversations for frontend display.
    Shows user query + AI response pairs that got thumbs down.
    """
    try:
        # Get top 5 most recent disliked AI messages
        disliked_messages = await ChatMessage.filter(
            liked=False,
            role="assistant"
        ).order_by('-created_at').limit(5).prefetch_related('session')

        sample_data = []

        for ai_msg in disliked_messages:
            # Get the corresponding user message
            user_msg = await ChatMessage.get_or_none(message_id=ai_msg.parent_message_id)

            if user_msg:
                sample_data.append({
                    "id": ai_msg.message_id,
                    "session_id": ai_msg.session.session_id,
                    "user_query": user_msg.content,
                    "ai_response": ai_msg.content,
                    "model_used": ai_msg.model_used,
                    "uses_rag": ai_msg.uses_rag,
                    "timestamp": ai_msg.created_at.isoformat()
                })

        return {
            "total_samples": len(sample_data),
            "samples": sample_data,
            "message": "Top 5 disliked conversations - these need improvement!"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-data/paginated")
async def get_sample_data_paginated(
        page: int = 1,
        limit: int = 5,
        session_id: Optional[str] = None
):
    """
    Get paginated disliked conversations.
    Can filter by session_id.
    """
    try:
        offset = (page - 1) * limit

        # Build query
        query = ChatMessage.filter(liked=False, role="assistant")

        if session_id:
            session = await ChatSession.get_or_none(session_id=session_id)
            if session:
                query = query.filter(session=session)

        # Get total count
        total = await query.count()

        # Get paginated results
        disliked_messages = await query.order_by('-created_at').offset(offset).limit(limit).prefetch_related('session')

        sample_data = []
        for ai_msg in disliked_messages:
            user_msg = await ChatMessage.get_or_none(message_id=ai_msg.parent_message_id)

            if user_msg:
                sample_data.append({
                    "id": ai_msg.message_id,
                    "session_id": ai_msg.session.session_id,
                    "user_query": user_msg.content,
                    "ai_response": ai_msg.content,
                    "model_used": ai_msg.model_used,
                    "uses_rag": ai_msg.uses_rag,
                    "timestamp": ai_msg.created_at.isoformat()
                })

        return {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit,
            "samples": sample_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-training-data")
async def generate_training_data_from_dislikes():
    """
    Prepare disliked messages for AutoTune pipeline.
    Creates training dataset from failed conversations.
    """
    try:
        # Get all disliked messages
        disliked_messages = await ChatMessage.filter(
            liked=False,
            role="assistant"
        ).order_by('-created_at').prefetch_related('session')

        if not disliked_messages:
            return {
                "message": "No disliked messages found",
                "training_data_generated": 0
            }

        # Build conversation text for pipeline
        conversations = []
        for ai_msg in disliked_messages:
            user_msg = await ChatMessage.get_or_none(message_id=ai_msg.parent_message_id)
            if user_msg:
                conversations.append({
                    "user_query": user_msg.content,
                    "ai_response": ai_msg.content,
                    "session_id": ai_msg.session.session_id
                })

        # Format for pipeline
        conversations_text = "\n\n---\n\n".join([
            f"User: {c['user_query']}\nAssistant: {c['ai_response']}\nFeedback: Disliked"
            for c in conversations
        ])

        batch_id = f"disliked-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return {
            "message": "Ready to generate training data",
            "disliked_count": len(conversations),
            "batch_id": batch_id,
            "conversations_preview": conversations[:3],
            "next_step": "POST to /pipeline1/run with batch_id and conversations_text"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/feedback")
async def get_feedback_stats():
    """
    Get statistics about likes/dislikes for dashboard.
    """
    try:
        total_messages = await ChatMessage.filter(role="assistant").count()
        liked = await ChatMessage.filter(role="assistant", liked=True).count()
        disliked = await ChatMessage.filter(role="assistant", liked=False).count()
        no_feedback = await ChatMessage.filter(role="assistant", liked=None).count()

        # Get breakdown by model
        rag_messages = await ChatMessage.filter(role="assistant", uses_rag=True).count()
        rag_liked = await ChatMessage.filter(role="assistant", uses_rag=True, liked=True).count()
        rag_disliked = await ChatMessage.filter(role="assistant", uses_rag=True, liked=False).count()

        no_rag_messages = await ChatMessage.filter(role="assistant", uses_rag=False).count()
        no_rag_liked = await ChatMessage.filter(role="assistant", uses_rag=False, liked=True).count()
        no_rag_disliked = await ChatMessage.filter(role="assistant", uses_rag=False, liked=False).count()

        return {
            "overall": {
                "total": total_messages,
                "liked": liked,
                "disliked": disliked,
                "no_feedback": no_feedback,
                "like_rate": round(liked / total_messages * 100, 1) if total_messages > 0 else 0,
                "dislike_rate": round(disliked / total_messages * 100, 1) if total_messages > 0 else 0
            },
            "with_rag": {
                "total": rag_messages,
                "liked": rag_liked,
                "disliked": rag_disliked,
                "like_rate": round(rag_liked / rag_messages * 100, 1) if rag_messages > 0 else 0
            },
            "without_rag": {
                "total": no_rag_messages,
                "liked": no_rag_liked,
                "disliked": no_rag_disliked,
                "like_rate": round(no_rag_liked / no_rag_messages * 100, 1) if no_rag_messages > 0 else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))