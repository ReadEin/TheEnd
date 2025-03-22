import textwrap
import unittest

from src.agent_impl.openai.openai_agent_base import OpenAIAgentBase, RequestBase, RequestPrepare, ToolChoiceType
from src.agent_impl.openai.schema.function import Function
from src.agent_impl.openai.schema.message import Message
from src.agent_impl.openai.schema.parameters import Parameters
from src.agent_impl.openai.schema.tool import Tool


class TestOpenAIAgentBase(unittest.TestCase):
    def test_openai_agent_base(self):
        tools = [
            Tool(
                name="summarize_conversation",
                description="지금까지의 대화 내용에 대한 요약본을 저장합니다.",
                function=Function(
                    name="summarize_conversation", 
                    description="지금까지의 대화 내용에 대한 요약본을 저장합니다.",
                    parameters=Parameters(
                        type="object",
                        properties={
                            "summary": {"type": "string", "description": "요약된 대화 내용"}
                        },
                        required=["summary"],
                        additionalProperties=False
                    ),
                    real_function=lambda x: x,
                    response_format=None
                )
            )
        ]
        requestPrepare = RequestPrepare(
            system_message=textwrap.dedent("""
            당신은 사용자와 자연스러운 대화를 나누는 AI 어시스턴트입니다.
            """),
            tools=tools,
            tool_choice=ToolChoiceType.REQUIRED,
            response_format=None
        )
        history = [
            Message(role="user", content="안녕하세요."),
            Message(role="assistant", content="안녕하세요. 반갑습니다."),
            Message(role="user", content="오늘 날씨가 좋네요."),
            Message(role="assistant", content="오늘은 맑은 날씨입니다."),
        ]
        request = RequestBase(requestPrepare, history)
        request.of("여태 우리가 나눈 대화를 요약해줘")
        print(f"request: {request}")
        agent = OpenAIAgentBase(model_name="gpt-3.5-turbo")
        response = agent.request(request, "여태 우리가 나눈 대화를 요약해줘")
        print(f"response: {response}")
        pass

if __name__ == "__main__":
    unittest.main()

