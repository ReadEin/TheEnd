from enum import Enum

class FinishReasonEnum(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    NONE = "none"

class RoleEnum(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
