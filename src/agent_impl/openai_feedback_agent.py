from typing import List

from src.agent_impl.openai.openai_agent_base import RequestBase, RequestPrepare
from .openai.openai_agent_util import OpenAIAgentClient, CompletionCreate, CompletionChoice
from src.agent_impl.openai.schema.message import Message
from src.agent_impl.openai.schema.enums import RoleEnum
from pydantic import BaseModel, Field

class FeedbackResponse(BaseModel):
    praise: str = Field(description="칭찬할 만한 부분과 발췌 내용")
    criticism: str = Field(description="개선이 필요한 부분과 발췌 내용")
    feedback: str = Field(description="전반적인 피드백과 제안사항")

class FeedbackRequestPrepare(RequestPrepare):
    """
    FeedbackResponse로 RequestPrepare를 생성한다.
    FeedbackResponse를 _response_format으로 사용한다.
    """
    def __init__(self):
        system_message = """
        당신은 글쓰기 피드백을 제공하는 전문가입니다. 
        주어진 텍스트를 분석하고 다음 세 가지 관점에서 피드백을 제공해주세요:
        
        1. 칭찬할 만한 부분 (praise): 잘 작성된 부분, 효과적인 표현, 좋은 구조 등을 지적하고 구체적인 예시와 함께 설명해주세요.
        2. 개선이 필요한 부분 (criticism): 수정이 필요한 부분, 모호한 표현, 개선할 수 있는 구조 등을 지적하고 구체적인 예시와 함께 설명해주세요.
        3. 전반적인 피드백 (feedback): 전체적인 인상과 개선을 위한 구체적인 제안사항을 제공해주세요.
        
        항상 건설적이고 객관적인 피드백을 제공하며, 가능한 구체적인 예시와 함께 설명해주세요.
        """
        super().__init__(
            system_message=system_message,
            tools=[],
            tool_choice="none",
            response_format={"type": "json_object", "schema": FeedbackResponse.model_json_schema()}
        )

class FeedbackRequest(RequestBase):
    """
    FeedbackResponse로 RequestPrepare를 생성한다.
    """
    def __init__(self, history: List[Message] = None):
        prepare = FeedbackRequestPrepare()
        super().__init__(prepare=prepare, history=history)
        
    def create_completion(self, content: str) -> CompletionCreate:
        """
        CompletionCreate 객체를 생성하여 반환합니다.
        """
        return self.of(content)

class OpenAIFeedbackAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.client = OpenAIAgentClient(model_name)
        self.system_prompt = """
        당신은 글쓰기 피드백을 제공하는 전문가입니다. 
        주어진 텍스트를 분석하고 다음 세 가지 관점에서 피드백을 제공해주세요:
        
        1. 칭찬할 만한 부분 (praise): 잘 작성된 부분, 효과적인 표현, 좋은 구조 등을 지적하고 구체적인 예시와 함께 설명해주세요.
        2. 개선이 필요한 부분 (criticism): 수정이 필요한 부분, 모호한 표현, 개선할 수 있는 구조 등을 지적하고 구체적인 예시와 함께 설명해주세요.
        3. 전반적인 피드백 (feedback): 전체적인 인상과 개선을 위한 구체적인 제안사항을 제공해주세요.
        
        항상 건설적이고 객관적인 피드백을 제공하며, 가능한 구체적인 예시와 함께 설명해주세요.
        """
    
    def extract_history(self, question: str, answer: str) -> List[Message]:
        # 질문과 답변을 포함하는 사용자 메시지 생성
        user_message = Message(
            index=0,
            role=RoleEnum.USER,
            content=f"다음 질문에 대한 답변을 평가해주세요:\n\n질문: {question}\n\n답변: {answer}"
        )
        
        history = [user_message]
        
        return history
    
    def request(self, text: str, history: List[Message] = None) -> FeedbackResponse:
        if not history:
            history = [Message(
                index=0,
                role=RoleEnum.USER,
                content=text
            )]
            text = ""
        
        request = FeedbackRequest(history)
        completion_create = request.create_completion(text)
        
        response = self.client.completion(completion_create)
        
        if not response or len(response) == 0:
            return FeedbackResponse(
                praise="응답을 받지 못했습니다.",
                criticism="API 호출에 문제가 있었습니다.",
                feedback="다시 시도해 보세요."
            )
        
        result = response[0]
        
        if not result.message or not result.message.content:
            return FeedbackResponse(
                praise="응답 내용이 비어있습니다.",
                criticism="API 응답 형식에 문제가 있었습니다.",
                feedback="다시 시도해 보세요."
            )
        
        try:
            import json
            feedback_data = json.loads(result.message.content)
            feedback_response = FeedbackResponse(**feedback_data)
            return feedback_response
        except Exception as e:
            print(f"Error parsing response: {e}")
            return FeedbackResponse(
                praise="응답 파싱 중 오류가 발생했습니다.",
                criticism=f"오류 내용: {str(e)}",
                feedback="다시 시도해 보세요."
            )
    
    def feedback_for_qa(self, question: str, answer: str) -> FeedbackResponse:
        history = self.extract_history(question, answer)
        
        return self.request("", history) 