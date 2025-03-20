import json
from typing import Callable, Generic, TypeVar
from pydantic import BaseModel, Field
from enum import Enum

class FinishReasonEnum(str,Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    NULL = None

class RoleEnum(str,Enum):
    USER = "user"
    SYSTEM = "system"

class Role(BaseModel):
    role : RoleEnum

class Message(BaseModel):
    index : int
    role : RoleEnum
    content : str

P = TypeVar("P", bound=BaseModel)
class Properties(Generic[P],BaseModel):
    type : str|list[str] = "object"
    properties : P
    
class Parameters(Generic[P],BaseModel):
    type : str = "object"
    properties : Properties[P]
    required : list[str] = []
    additionalProperties : bool = False
    
    
    
class Function(BaseModel):
    name : str
    description : str
    strict : bool = True
    parameters : Parameters
    real_function : Callable = Field(exclude=True)
    
class FunctionCall(BaseModel):
    id : str
    name : str
    arguments : str
    def get_arguments(self) -> dict:
        return json.loads(self.arguments)
    

class Tool(BaseModel):
    type : str = "function"
    
class FunctionItem(BaseModel):
    name : str
    
class ToolChoice(BaseModel):
    type : str = "function"
    function : FunctionItem
        
T = TypeVar("T", bound=BaseModel)
class CompletionCreate(Generic[T],BaseModel):
    messages : list[Message]
    usage : str|None = None
    response_format : T|None = None
    tools : list[Tool]|None = None
    tool_choice : ToolChoice|str = "none"

class CompletionChoice(BaseModel):
    index : int
    message : Message|None = None
    finish_reason : FinishReasonEnum
    usage : str|None = None
    tool_calls : list[FunctionCall]|None = None
    
    
