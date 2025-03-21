from openai import OpenAI
from .openai_agent_util import OpenAIAgentClient, CompletionCreate,CompletionChoice
from src.agent_impl.openai.schema.message import Message
from pydantic import BaseModel
class FeedbackResponse(BaseModel):
    praise: str  # 칭찬할 만한 부분과 발췌 내용
    criticism: str  # 개선이 필요한 부분과 발췌 내용
    feedback: str  # 전반적인 피드백과 제안사항

class FeedbackRequest:
    _system_message : str = (
        "당신은 작가의 어시스턴트입니다. "
        "작가의 창작 과정을 돕고, "
        "스토리 구성, 캐릭터 발전, 문체 개선에 대한 건설적인 제안을 제공하며, "
        "작품의 전반적인 완성도를 높이는 데 도움을 주세요. "
        "응답은 다음과 같은 구조로 작성해주세요: "
        "1. praise: 작품에서 잘된 점과 그 근거가 되는 구절을 발췌해서 설명해주세요 "
        "2. criticism: 개선이 필요한 부분과 그 근거가 되는 구절을 발췌해서 설명해주세요 "
        "3. feedback: 전반적인 피드백과 구체적인 개선 제안사항을 작성해주세요"
    )
    def of(self, content: str) -> CompletionCreate:
        return CompletionCreate(
            messages=[
                Message(role="system", content=self._system_message),
                Message(role="user", content=content)
            ],
            response_format=FeedbackResponse
        )

class OpenAIFeedbackAgent:
    def __init__(self, model_name: str):
        self.openai_agent_client = OpenAIAgentClient(model_name=model_name)
        
    def feedback(self, content: str) -> FeedbackResponse:
        completion = self.openai_agent_client.completion(FeedbackRequest().of(content))
        return completion[0].message.content

