from typing import Generic
from pydantic import BaseModel
from typing import TypeVar

Prompt_Schema = TypeVar('Prompt_Schema', bound=BaseModel)

class PromptSentence(Generic[Prompt_Schema]):
    # template
    arg:Prompt_Schema
    def __init__(self, arg:Prompt_Schema):
        self.arg = arg
    def get_sentence(self)->str:
        return "you are a helpful assistant"
    def from_arg(self)->str:
        return self.get_sentence().format(**self.arg)