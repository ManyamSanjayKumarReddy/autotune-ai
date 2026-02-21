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