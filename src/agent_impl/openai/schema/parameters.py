from pydantic import BaseModel, Field

class Parameters(BaseModel):
    type: str = Field(default="object")
    properties: dict
    required: list[str]|None = None
    additional_properties: bool = Field(default=False, alias="additionalProperties") 