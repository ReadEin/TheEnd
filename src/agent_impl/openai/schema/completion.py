from typing import Generic, TypeVar
from pydantic import BaseModel
from .message import Message
from .tool import Tool, ToolChoice
from .function import FunctionCall
from .enums import FinishReasonEnum

class CompletionCreate(BaseModel):
    messages: list[Message]
    usage: str|None = None ## 사용된 토큰 수 ex) "1000"
    response_format: dict|BaseModel|None = None
    tools: list[Tool]|None = None
    tool_choice: ToolChoice|str = "none"

class CompletionChoice(BaseModel):
    index: int
    message: Message|None = None
    finish_reason: FinishReasonEnum
    usage: str|None = None
    tool_calls: list[FunctionCall]|None = None 