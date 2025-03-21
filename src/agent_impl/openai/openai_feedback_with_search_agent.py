from openai import OpenAI
from .openai_agent_util import OpenAIAgentClient, CompletionCreate, CompletionChoice
from src.agent_impl.openai.schema.message import Message
from src.agent_impl.openai.schema.function import Function, FunctionCall
from src.agent_impl.openai.schema.parameters import Parameters
from src.agent_impl.openai.schema.tool import Tool
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FeedbackWithSearchResponse(BaseModel):
    praise: str  # 칭찬할 만한 부분과 발췌 내용
    criticism: str  # 개선이 필요한 부분과 발췌 내용
    feedback: str  # 전반적인 피드백과 제안사항
    references: Optional[List[Dict[str, str]]]  # 검색을 통해 찾은 참고 자료

class SearchRequest(BaseModel):
    query: str  # 검색 쿼리

class FeedbackWithSearchRequest:
    _system_message: str = (
        "당신은 작가의 어시스턴트입니다. "
        "작가의 창작 과정을 돕고, "
        "스토리 구성, 캐릭터 발전, 문체 개선에 대한 건설적인 제안을 제공하며, "
        "작품의 전반적인 완성도를 높이는 데 도움을 주세요. "
        "필요한 경우 search 함수를 사용하여 관련 정보를 검색할 수 있습니다. "
        "응답은 다음과 같은 구조로 작성해주세요: "
        "1. praise: 작품에서 잘된 점과 그 근거가 되는 구절을 발췌해서 설명해주세요 "
        "2. criticism: 개선이 필요한 부분과 그 근거가 되는 구절을 발췌해서 설명해주세요 "
        "3. feedback: 전반적인 피드백과 구체적인 개선 제안사항을 작성해주세요 "
        "4. references: 검색을 통해 찾은 관련 자료나 참고 문헌을 제공해주세요(선택적)"
    )
    
    def __init__(self):
        # 검색 기능을 위한 Function 인스턴스 생성
        self._search_function = self._create_search_function()
        self._search_tool = Tool(function=self._search_function)
        
    def _create_search_function(self) -> Function:
        """검색 기능을 위한 Function 인스턴스를 생성합니다."""
        # 검색 매개변수 정의
        search_properties = {
            "query": {
                "type": "string",
                "description": "검색할 쿼리 문자열"
            }
        }
        
        # Parameters 객체 생성
        search_params = Parameters(
            type="object",
            properties=search_properties,
            required=["query"]
        )
        
        # 실제 검색 함수 정의
        def search_function(query: str) -> Dict[str, Any]:
            """외부 API를 사용한 검색 기능. 실제 구현에서는 검색 서비스 API를 호출합니다."""
            # 실제 구현에서는 여기에 검색 로직 구현
            # 예시 응답
            results = [
                {"title": f"검색 결과 1: {query}에 관한 자료", "url": "http://example.com/1", "snippet": "관련 내용 발췌..."},
                {"title": f"검색 결과 2: {query}에 관한 추가 정보", "url": "http://example.com/2", "snippet": "추가 정보..."}
            ]
            return {"results": results}
        
        # Function 객체 생성
        return Function(
            name="search",
            description="주어진 쿼리로 웹에서 관련 정보를 검색합니다.",
            parameters=search_params,
            real_function=search_function
        )
    
    def of(self, content: str) -> CompletionCreate:
        """CompletionCreate 객체를 생성합니다."""
        return CompletionCreate(
            messages=[
                Message(index=0, role="system", content=self._system_message),
                Message(index=1, role="user", content=content)
            ],
            response_format=FeedbackWithSearchResponse,
            tools=[self._search_tool]
        )

class OpenAIFeedbackWithSearchAgent:
    def __init__(self, model_name: str):
        self.openai_agent_client = OpenAIAgentClient(model_name=model_name)
        
    def _process_tool_calls(self, tool_calls: List[FunctionCall]) -> List[Dict[str, Any]]:
        """
        Tool 호출 결과를 처리합니다.
        """
        results = []
        for call in tool_calls:
            if call.name == "search":
                args = call.get_arguments()
                query = args.get("query", "")
                
                # 실제 검색 수행 (여기서는 가상의 결과)
                search_results = [
                    {"title": f"검색 결과 1: {query}에 관한 자료", "url": "http://example.com/1", "snippet": "관련 내용 발췌..."},
                    {"title": f"검색 결과 2: {query}에 관한 추가 정보", "url": "http://example.com/2", "snippet": "추가 정보..."}
                ]
                results.append({"name": call.name, "results": search_results})
        
        return results
    
    def feedback_with_search(self, content: str) -> FeedbackWithSearchResponse:
        """
        검색 기능이 포함된 피드백을 제공합니다.
        1. 초기 요청 전송
        2. 검색 함수 호출이 있으면 결과 처리
        3. 검색 결과와 함께 최종 응답 생성
        """
        # 초기 요청 생성 및 전송
        request = FeedbackWithSearchRequest()
        completion_choices = self.openai_agent_client.completion(request.of(content))
        
        # 첫 번째 응답 확인
        if not completion_choices or len(completion_choices) == 0:
            raise ValueError("응답을 받지 못했습니다.")
            
        first_choice = completion_choices[0]
        
        # Tool 호출이 있는지 확인
        if first_choice.tool_calls and len(first_choice.tool_calls) > 0:
            # Tool 호출 결과 처리
            tool_results = self._process_tool_calls(first_choice.tool_calls)
            
            # 검색 결과를 포함한 새 메시지 생성
            messages = [
                Message(index=0, role="system", content=request._system_message),
                Message(index=1, role="user", content=content),
                Message(index=2, role="assistant", content=None, tool_calls=first_choice.tool_calls),
                Message(index=3, role="user", content=f"검색 결과: {str(tool_results)}")
            ]
            
            # 최종 요청 생성 및 전송
            final_request = CompletionCreate(
                messages=messages,
                response_format=FeedbackWithSearchResponse
            )
            
            final_completion = self.openai_agent_client.completion(final_request)
            
            # 최종 응답 반환
            if final_completion and len(final_completion) > 0:
                return final_completion[0].message.content
        
        # Tool 호출이 없었다면 첫 응답 반환
        return first_choice.message.content
