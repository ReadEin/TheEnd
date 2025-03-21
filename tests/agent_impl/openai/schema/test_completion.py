from src.agent_impl.openai.schema.completion import CompletionCreate, CompletionChoice
from src.agent_impl.openai.schema.message import Message
from src.agent_impl.openai.schema.tool import Tool, ToolChoice
from src.agent_impl.openai.schema.function import Function, FunctionCall, FunctionItem
from src.agent_impl.openai.schema.parameters import Parameters
from src.agent_impl.openai.schema.enums import RoleEnum, FinishReasonEnum
import unittest

class TestCompletionCreate(unittest.TestCase):
    def test_completion_create_basic(self):
        """
        CompletionCreate 클래스가 기본 설정으로 예상대로 생성되는지 테스트합니다.
        """
        messages = [
            Message(index=0, role=RoleEnum.USER, content="Hello, AI!")
        ]
        
        completion = CompletionCreate(messages=messages)
        
        self.assertEqual(len(completion.messages), 1)
        self.assertEqual(completion.messages[0].content, "Hello, AI!")
        self.assertIsNone(completion.usage)
        self.assertIsNone(completion.response_format)
        self.assertIsNone(completion.tools)
        self.assertEqual(completion.tool_choice, "none")
        
    def test_completion_create_with_tools(self):
        """
        CompletionCreate 클래스가 도구와 함께 예상대로 생성되는지 테스트합니다.
        """
        messages = [
            Message(index=0, role=RoleEnum.SYSTEM, content="You are a helpful assistant."),
            Message(index=1, role=RoleEnum.USER, content="What's the weather in Seoul?")
        ]
        
        # 도구 생성
        location_properties = {
                "type": "string",
                "description": "City name"
            }
        
        params = Parameters(properties=location_properties)
        
        def get_weather(location: str):
            return f"Weather in {location}: Sunny, 25°C"
            
        function = Function(
            name="get_weather",
            description="Get weather information for a location",
            parameters=params,
            real_function=get_weather
        )
        
        tool = Tool(function=function)
        function_item = FunctionItem(name="get_weather")
        tool_choice = ToolChoice(function=function_item)
        
        completion = CompletionCreate(
            messages=messages,
            tools=[tool],
            tool_choice=tool_choice
        )
        
        self.assertEqual(len(completion.messages), 2)
        self.assertEqual(len(completion.tools), 1)
        self.assertEqual(completion.tools[0].function.name, "get_weather")
        self.assertEqual(completion.tool_choice.function.name, "get_weather")


class TestCompletionChoice(unittest.TestCase):
    def test_completion_choice_with_message(self):
        """
        CompletionChoice 클래스가 메시지와 함께 예상대로 생성되는지 테스트합니다.
        """
        message = Message(
            index=0,
            role=RoleEnum.SYSTEM,
            content="I'm an AI assistant."
        )
        
        choice = CompletionChoice(
            index=0,
            message=message,
            finish_reason=FinishReasonEnum.STOP
        )
        
        self.assertEqual(choice.index, 0)
        self.assertEqual(choice.message.content, "I'm an AI assistant.")
        self.assertEqual(choice.finish_reason, FinishReasonEnum.STOP)
        self.assertIsNone(choice.usage)
        self.assertIsNone(choice.tool_calls)
        
    def test_completion_choice_with_tool_calls(self):
        """
        CompletionChoice 클래스가 도구 호출과 함께 예상대로 생성되는지 테스트합니다.
        """
        function_call = FunctionCall(
            id="call_123",
            name="get_weather",
            arguments='{"location": "Seoul"}'
        )
        
        choice = CompletionChoice(
            index=0,
            finish_reason=FinishReasonEnum.STOP,
            tool_calls=[function_call]
        )
        
        self.assertEqual(choice.index, 0)
        self.assertEqual(choice.finish_reason, FinishReasonEnum.STOP)
        self.assertEqual(len(choice.tool_calls), 1)
        self.assertEqual(choice.tool_calls[0].name, "get_weather")
        
        # get_arguments 메서드 테스트
        args = choice.tool_calls[0].get_arguments()
        self.assertEqual(args["location"], "Seoul")


if __name__ == '__main__':
    unittest.main()
