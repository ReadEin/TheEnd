import textwrap
from src.agent_impl.openai.openai_agent_base import OpenAIAgentBase, RequestBase, RequestPrepare
from src.agent_impl.openai.openai_agent_util import OpenAIAgentClient
from src.agent_impl.openai.schema.message import Message
from src.agent_impl.openai.schema.function import Function
from src.agent_impl.openai.schema.tool import Tool
from pydantic import BaseModel, Field

class ChatResponse(BaseModel):
    content: str = Field(description="챗봇의 응답 내용")
    has_summary: bool = Field(description="요약이 생성되었는지 여부", default=False)

class SummaryResponse(BaseModel):
    summary: str = Field(description="대화 내용의 요약")

class ChatRequestPrepare(RequestPrepare):
    _history: list[Message] = []
    def __init__(self, with_summary_tool: bool = True):
        system_message = textwrap.dedent("""
        당신은 사용자와 자연스러운 대화를 나누는 AI 어시스턴트입니다.
        사용자의 질문에 친절하고 도움이 되는 방식으로 응답해 주세요.
        필요할 때는 'summary' 도구를 사용하여 대화 내용을 요약할 수 있습니다.
        """)
        
        tools = []
        
        if with_summary_tool:
            summary_function = Function(
                name="summarize_conversation",
                description="지금까지의 대화 내용에 대한 요약본을 저장합니다.",
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "요약된 대화 내용"
                        }
                    },
                    "required": ["summary"]
                },
                real_function=lambda args: self.save_summary(args["summary"])
            )
            
            tools.append(Tool(function=summary_function))
        
        super().__init__(
            system_message=system_message,
            tools=tools,
            tool_choice="required"
        )
        
    def save_summary(self, summary: str):
        self._history.clear()
        self._history.append(Message(role="assistant", content=summary))
    
    def get_history(self) -> list[Message]:
        return self._history


class ChatRequest(RequestBase):
    def __init__(self, with_summary_tool: bool = True):
        prepare = ChatRequestPrepare(with_summary_tool=with_summary_tool)
        super().__init__(prepare=prepare)
        self.set_history(prepare.get_history())
        self.set_functions(prepare.get_functions())

class OpenAIChatAgent(OpenAIAgentBase):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name)
        self.set_client(OpenAIAgentClient(model_name))
        
    def request(self, request: ChatRequest, content: str) -> ChatResponse:
        response_str = super().request(request, content)
        try:
            return ChatResponse.model_validate_json(response_str)
        except Exception as e:
            print(f"OpenAIChatAgent - 응답 파싱 중 오류가 발생했습니다: {e}")
            raise e
