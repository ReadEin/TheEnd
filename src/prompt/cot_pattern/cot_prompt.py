from pydantic import BaseModel
from src.prompt.prompt_sentence import Prompt_Schema, PromptSentence
from textwrap import dedent

sentence = dedent("""
    지시사항 :
    {direction}
    문제 해결 절차 :
    {cot}
    배경 지식 :
    {background}
    """)

class CotSchema(BaseModel):
    direction: str
    chain_of_thought: list[str]
    background: str

class CotPtrompt(PromptSentence[CotSchema]):
    def __init__(self, cotSchema:CotSchema):
        super().__init__(cotSchema)
    def get_sentence(self)->str:
        return sentence
    def from_arg(self)->str:
        cot = "\n".join(f"{i+1}. {chain}" for i, chain in enumerate(self.arg.chain_of_thought))
        return self.get_sentence().format(
                direction=self.arg.direction,
                cot=cot,
                background=self.arg.background
            )

