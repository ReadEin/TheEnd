import json
from typing import Callable
from pydantic import BaseModel, Field
from .parameters import Parameters

class Function(BaseModel):
    name: str
    description: str
    strict: bool = True
    parameters: Parameters
    real_function: Callable = Field(exclude=True)

class FunctionCall(BaseModel):
    id: str
    name: str
    arguments: str
    def get_arguments(self) -> dict:
        return json.loads(self.arguments)

class FunctionItem(BaseModel):
    name: str 