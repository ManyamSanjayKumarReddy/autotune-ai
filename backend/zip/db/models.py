# db/models.py

from tortoise import fields
from tortoise.models import Model


class PipelineRun(Model):
    id = fields.IntField(pk=True)
    batch_id = fields.CharField(max_length=255, unique=True)
    conversations_text = fields.TextField()
    csv_path = fields.CharField(max_length=500, null=True)
    final_samples_count = fields.IntField(default=0)
    status = fields.CharField(max_length=50, default="pending")  # pending | success | failed | no_samples
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "pipeline_runs"


class ChromaIngestion(Model):
    id = fields.IntField(pk=True)
    pipeline_run = fields.ForeignKeyField(
        "models.PipelineRun",
        related_name="ingestions",
        on_delete=fields.CASCADE
    )
    batch_id = fields.CharField(max_length=255)
    collection_name = fields.CharField(max_length=255)
    documents_ingested = fields.IntField(default=0)
    status = fields.CharField(max_length=50, default="pending")  # pending | success | failed
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "chroma_ingestions"


class ChatSession(Model):
    """Chat session - represents a conversation thread"""
    id = fields.IntField(pk=True)
    session_id = fields.CharField(max_length=50, unique=True, index=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    is_active = fields.BooleanField(default=True)

    class Meta:
        table = "chat_sessions"


class ChatMessage(Model):
    """Individual chat messages"""
    id = fields.IntField(pk=True)
    message_id = fields.CharField(max_length=50, unique=True, index=True)
    session = fields.ForeignKeyField('models.ChatSession', related_name='messages')

    # Message content
    role = fields.CharField(max_length=20)  # 'user' or 'assistant'
    content = fields.TextField()

    # Relationships
    parent_message_id = fields.CharField(max_length=50, null=True)  # For linking AI response to user query

    # Feedback
    liked = fields.BooleanField(null=True, default=None)  # True=like, False=dislike, None=no feedback

    # Metadata
    model_used = fields.CharField(max_length=100, null=True)
    uses_rag = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "chat_messages"
        ordering = ["created_at"]
