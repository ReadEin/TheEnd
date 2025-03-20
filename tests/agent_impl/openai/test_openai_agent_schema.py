import json
import unittest
from src.agent_impl.openai.openai_agent_schema import (
    FinishReasonEnum,
    RoleEnum,
    Role,
    Message,
    Properties,
    Parameters,
    Function,
    FunctionCall,
    Tool,
    FunctionItem,
    ToolChoice,
    CompletionCreate,
    CompletionChoice
)

class TestFinishReasonEnum(unittest.TestCase):
    def test_finish_reason_values(self):
        self.assertEqual(FinishReasonEnum.STOP, "stop")
        self.assertEqual(FinishReasonEnum.LENGTH, "length")
        self.assertEqual(FinishReasonEnum.CONTENT_FILTER, "content_filter")
        self.assertEqual(FinishReasonEnum.NULL, 'None')

class TestRoleEnum(unittest.TestCase):
    def test_role_values(self):
        self.assertEqual(RoleEnum.USER, "user")
        self.assertEqual(RoleEnum.SYSTEM, "system")

class TestRole(unittest.TestCase):
    def test_role_creation(self):
        role = Role(role=RoleEnum.USER)
        self.assertEqual(role.role, RoleEnum.USER)
        
        role = Role(role=RoleEnum.SYSTEM)
        self.assertEqual(role.role, RoleEnum.SYSTEM)

class TestMessage(unittest.TestCase):
    def test_message_creation(self):
        message = Message(index=0, role=RoleEnum.USER, content="Hello")
        self.assertEqual(message.index, 0)
        self.assertEqual(message.role, RoleEnum.USER)
        self.assertEqual(message.content, "Hello")

class TestFunctionCall(unittest.TestCase):
    def test_function_call_creation(self):
        function_call = FunctionCall(
            id="call_123",
            name="test_function",
            arguments='{"arg1": "value1", "arg2": 42}'
        )
        self.assertEqual(function_call.id, "call_123")
        self.assertEqual(function_call.name, "test_function")
        self.assertEqual(function_call.arguments, '{"arg1": "value1", "arg2": 42}')
        
    def test_get_arguments(self):
        function_call = FunctionCall(
            id="call_123",
            name="test_function",
            arguments='{"arg1": "value1", "arg2": 42}'
        )
        args = function_call.get_arguments()
        self.assertEqual(args, {"arg1": "value1", "arg2": 42})
        self.assertIsInstance(args, dict)

class TestToolChoice(unittest.TestCase):
    def test_tool_choice_creation(self):
        function_item = FunctionItem(name="test_function")
        tool_choice = ToolChoice(function=function_item)
        
        self.assertEqual(tool_choice.type, "function")
        self.assertEqual(tool_choice.function.name, "test_function")

class TestCompletionCreate(unittest.TestCase):
    def test_completion_create_minimal(self):
        message = Message(index=0, role=RoleEnum.USER, content="Hello")
        completion_create = CompletionCreate(messages=[message])
        
        self.assertEqual(len(completion_create.messages), 1)
        self.assertEqual(completion_create.messages[0].content, "Hello")
        self.assertIsNone(completion_create.usage)
        self.assertIsNone(completion_create.response_format)
        self.assertIsNone(completion_create.tools)
        self.assertEqual(completion_create.tool_choice, "none")
    
    def test_completion_create_with_tools(self):
        message = Message(index=0, role=RoleEnum.USER, content="Hello")
        
        props_dict = {"name": "test"}
        param_props = Properties(type="object", properties=props_dict)
        params = Parameters(properties=param_props)
        
        def dummy_func():
            pass
        
        function = Function(
            name="test_function",
            description="A test function",
            parameters=params,
            real_function=dummy_func
        )
        
        tool = Tool(type="function")
        
        function_item = FunctionItem(name="test_function")
        tool_choice = ToolChoice(function=function_item)
        
        completion_create = CompletionCreate(
            messages=[message],
            tools=[tool],
            tool_choice=tool_choice
        )
        
        self.assertEqual(len(completion_create.messages), 1)
        self.assertIsNotNone(completion_create.tools)
        self.assertEqual(len(completion_create.tools), 1)
        self.assertEqual(completion_create.tool_choice.function.name, "test_function")

class TestCompletionChoice(unittest.TestCase):
    def test_completion_choice_creation(self):
        message = Message(index=0, role=RoleEnum.SYSTEM, content="Response")
        
        completion_choice = CompletionChoice(
            index=0,
            message=message,
            finish_reason=FinishReasonEnum.STOP
        )
        
        self.assertEqual(completion_choice.index, 0)
        self.assertEqual(completion_choice.message.content, "Response")
        self.assertEqual(completion_choice.finish_reason, FinishReasonEnum.STOP)
        self.assertIsNone(completion_choice.usage)
        self.assertIsNone(completion_choice.tool_calls)
    
    def test_completion_choice_with_tool_calls(self):
        message = Message(index=0, role=RoleEnum.SYSTEM, content="Response")
        
        function_call = FunctionCall(
            id="call_123",
            name="test_function",
            arguments='{"arg1": "value1"}'
        )
        
        completion_choice = CompletionChoice(
            index=0,
            message=message,
            finish_reason=FinishReasonEnum.STOP,
            tool_calls=[function_call]
        )
        
        self.assertIsNotNone(completion_choice.tool_calls)
        self.assertEqual(len(completion_choice.tool_calls), 1)
        self.assertEqual(completion_choice.tool_calls[0].id, "call_123")
        self.assertEqual(completion_choice.tool_calls[0].get_arguments()["arg1"], "value1")

if __name__ == '__main__':
    unittest.main() 