import json
from typing import Any, Callable, Dict, Generic, List, TypeVar

from pydantic import BaseModel

from src.agent.llm_agent import LLMAgent
from src.prompt.prompt_sentence import PromptSentence, Prompt_Schema
from src.prompt.jformatter_pattern.jformatter_prompt import JFormatterPrompt, JFormatterSchema

PipeActionInputSchema = TypeVar('PipeActionInputSchema', bound=BaseModel)
PipeActionOutputSchema = TypeVar('PipeActionOutputSchema', bound=BaseModel)

class LlmAgentPipeAction(Generic[PipeActionInputSchema, PipeActionOutputSchema]):
    action:Callable[[PipeActionInputSchema],PipeActionOutputSchema]
    def __init__(self, action:Callable[[PipeActionInputSchema],PipeActionOutputSchema]):
        self.action = action
    def run(self, input:PipeActionInputSchema)->PipeActionOutputSchema:
        return self.action(input)

class LlmAgentPipe:
    """LlmAgent를 사용하여 프롬프트 -> Json 파싱 -> 액션 -> 결과 반환"""
    _llm_agent:LLMAgent
    _pipe_action:LlmAgentPipeAction[PipeActionInputSchema, PipeActionOutputSchema]
    def __init__(
        self, llm_agent: LLMAgent,
        pipe_action:LlmAgentPipeAction[PipeActionInputSchema, PipeActionOutputSchema]
        ):
        self._llm_agent = llm_agent
        self._pipe_action = pipe_action
    def run(self, initial_prompt: PromptSentence[Any]) -> PipeActionOutputSchema:
        # 1. 초기 프롬프트 실행
        initial_result = self._llm_agent.run(initial_prompt)
        # 2. JFormatter 프롬프트 생성 및 실행
        jformatter_schema = JFormatterSchema(
            content=initial_result,
            json_keys=PipeActionInputSchema.model_fields.keys()
        )
        jformatter_prompt = JFormatterPrompt(jformatter_schema)
        json_result = self._llm_agent.run(jformatter_prompt)

        # 3. JSON 문자열을 파이썬 딕셔너리로 변환
        try:
            parsed = json.loads(json_result)
        except json.JSONDecodeError as e:
            # JSON 파싱 오류 시 빈 딕셔너리 반환하거나 원하는 방식으로 처리
            raise ValueError(f"format error : {str(e)}") from e
        # 4. 파이프 액션 실행
        try:
            # 파싱된 데이터를 PipeActionInputSchema로 변환
            input_schema = PipeActionInputSchema(**parsed)
            return self._pipe_action.run(input_schema)
        except (ValueError, TypeError) as e:
            # PipeActionInputSchema 생성 실패 또는 액션 실행 중 오류 발생
            raise ValueError(f"pipe action error : {str(e)}") from e
