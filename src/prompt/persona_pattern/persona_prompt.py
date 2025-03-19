from pydantic import BaseModel
from src.prompt.prompt_sentence import Prompt_Schema, PromptSentence
from textwrap import dedent

sentence = dedent("""
    페르소나 :
    {persona}
    
    페르소나 특성 :
    {traits}
    
    배경 지식 :
    {background}
    
    지시사항 :
    {instruction}
    """)

class PersonaSchema(BaseModel):
    persona: str
    traits: list[str]
    background: str
    instruction: str

class PersonaPrompt(PromptSentence[PersonaSchema]):
    def __init__(self, personaSchema: PersonaSchema):
        super().__init__(personaSchema)
    
    def get_sentence(self) -> str:
        return sentence
    
    def from_arg(self) -> str:
        traits = "\n".join(f"- {trait}" for trait in self.arg.traits)
        return self.get_sentence().format(
            persona=self.arg.persona,
            traits=traits,
            background=self.arg.background,
            instruction=self.arg.instruction
        )
