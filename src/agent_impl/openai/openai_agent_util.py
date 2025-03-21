from typing import Any
from openai import OpenAI
from openai.resources.chat.completions import Completions
from pydantic import BaseModel

from src.agent_impl.openai.schema.completion import CompletionChoice, CompletionCreate, Message, FinishReasonEnum, FunctionCall

class OpenAIAgentClient:
    _chat_completion_client : Completions
    _model_name : str
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._create_agent()
        
    def _get_access_token(self) -> str:
        """
        실행 인자로부터 API 토큰을 가져옵니다.
        """
        import os
        if not os.path.exists(".private.openai_token"):
            # 3. 토큰을 찾을 수 없는 경우 오류 발생
            raise ValueError("API 토큰을 찾을 수 없습니다. 환경 변수(OPENAI_API_KEY)를 설정하거나 --api-token 인자를 사용하세요.")
        with open(".private.openai_token", "r") as f:
            return f.read().strip()

    def _create_agent(self):
        try:
            access_token = self._get_access_token()
            client = OpenAI(api_key=access_token)
            self._chat_completion_client = client.chat.completions
        except Exception as e:
            print(e)
            raise e
        
    def _create_params(self, completion_create: CompletionCreate) -> dict:
        try:
            # CompletionCreate의 각 필드를 매개변수로 추출
            params = {
                "model": self._model_name,
                "messages": [
                    {
                        "role": message.role.value,
                        "content": message.content
                    } for message in completion_create.messages
                ]
            }
            
            # 선택적 매개변수 추가
            if (completion_create.response_format is not None and 
                (isinstance(completion_create.response_format, BaseModel) or 
                 isinstance(completion_create.response_format, dict))):
                params["response_format"] = completion_create.response_format
            if completion_create.tools is not None:
                params["tools"] = [tool.model_dump() for tool in completion_create.tools]
            
            if completion_create.tool_choice is not None:
                if isinstance(completion_create.tool_choice, str):
                    params["tool_choice"] = completion_create.tool_choice
                else:
                    params["tool_choice"] = completion_create.tool_choice.model_dump()
            
            return params
        except Exception as e:
            print(e)
            raise e
    
    def completion(self, completion_create: CompletionCreate) -> list[CompletionChoice]:
        try:
            params : dict = self._create_params(completion_create)
            response : Any = self._chat_completion_client.create(**params)
            
            # API 응답에서 CompletionChoice 객체 리스트로 변환
            completion_choices : list[CompletionChoice] = []
            
            for choice in response.choices:
                # Message 객체 생성
                message = None
                has_message = hasattr(choice, 'message') and choice.message is not None
                if has_message:
                    message = Message(
                        index=0,  # 기본값 설정
                        role=choice.message.role,
                        content=choice.message.content
                    )
                
                # tool_calls 변환
                tool_calls : list[FunctionCall] | None = None
                has_tool_calls = has_message and hasattr(choice.message, 'tool_calls')
                if has_tool_calls:
                    tool_calls = []
                    for tool_call in choice.message.tool_calls:
                        if hasattr(tool_call, 'function'):
                            # FunctionCall 객체 생성
                            function_call = FunctionCall(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments
                            )
                            tool_calls.append(function_call)
                
                # 종료 이유 변환
                finish_reason : FinishReasonEnum = FinishReasonEnum.NULL
                if choice.finish_reason == "stop":
                    finish_reason = FinishReasonEnum.STOP
                elif choice.finish_reason == "length":
                    finish_reason = FinishReasonEnum.LENGTH
                elif choice.finish_reason == "content_filter":
                    finish_reason = FinishReasonEnum.CONTENT_FILTER
                
                # CompletionChoice 객체 생성
                completion_choice : CompletionChoice = CompletionChoice(
                    index=choice.index,
                    message=message,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls
                )
                
                completion_choices.append(completion_choice)
            
            return completion_choices
            
        except Exception as e:
            print(e)
            raise e
    
    
