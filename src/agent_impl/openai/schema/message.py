from typing import Optional
from pydantic import BaseModel
from .enums import RoleEnum

class Role(BaseModel):
    role: RoleEnum

class Message(BaseModel):
    index: Optional[int] = None
    role: RoleEnum
    content: Optional[str] = None