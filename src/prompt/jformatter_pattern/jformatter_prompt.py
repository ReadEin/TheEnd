from pydantic import BaseModel
from src.prompt.prompt_sentence import Prompt_Schema, PromptSentence
from textwrap import dedent

sentence = dedent("""
    페르소나 :
    당신은 텍스트를 구조화된 JSON 형식으로 변환하는 전문가입니다.
    
    역할 :
    주어진 내용을 분석하고 지정된 JSON 키를 사용하여 정보를 요약-정리합니다.
    
    입력 내용 :
    {content}
    
    사용할 JSON 키 :
    {json_keys}
    
    지시사항 :
    1. 위 내용을 분석하여 중요한 정보를 추출하세요
    2. 추출한 정보를 제공된 JSON 키에 맞게 정리하세요
    3. 올바른 JSON 형식으로 출력하세요
    4. 모든 키에 적절한 값을 채우세요
    """)

class JFormatterSchema(BaseModel):
    content: str
    json_keys: list[str]

class JFormatterPrompt(PromptSentence[JFormatterSchema]):
    def __init__(self, jformatterSchema: JFormatterSchema):
        super().__init__(jformatterSchema)
    
    def get_sentence(self) -> str:
        return sentence
    
    def from_arg(self) -> str:
        keys_formatted = "\n".join(f"- {key}" for key in self.arg.json_keys)
        return self.get_sentence().format(
            content=self.arg.content,
            json_keys=keys_formatted
        )
