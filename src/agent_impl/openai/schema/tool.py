from pydantic import BaseModel
from .function import FunctionItem
from .function import Function

class Tool(BaseModel):
    type: str = "function"
    function: Function

class ToolChoice(BaseModel):
    type: str = "function"
    function: FunctionItem 