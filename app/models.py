from enum import Enum
from datetime import datetime
from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    completed_at: datetime | None = None
    file_name: str
    result: dict | None = None
    error: str | None = None


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
