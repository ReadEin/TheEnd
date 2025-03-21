from pydantic import BaseModel
from .enums import RoleEnum

class Role(BaseModel):
    role: RoleEnum

class Message(BaseModel):
    index: int
    role: RoleEnum
    content: str 