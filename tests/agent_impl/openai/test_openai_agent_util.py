import unittest
from unittest.mock import patch, MagicMock, mock_open

from src.agent_impl.openai.openai_agent_util import OpenAIAgentClient
from src.agent_impl.openai.openai_agent_schema import (
    Message, RoleEnum, CompletionCreate, FinishReasonEnum, FunctionCall, Tool, ToolChoice, FunctionItem
)

class TestOpenAIAgentClient(unittest.TestCase):
    
    @patch('src.agent_impl.openai.openai_agent_util.OpenAI')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="test-api-token")
    def test_init_and_create_agent(self, mock_file, mock_exists, mock_openai):
        # 파일 존재 여부 모킹
        mock_exists.return_value = True
        
        # OpenAI 클라이언트 모킹
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # 클라이언트 초기화
        client = OpenAIAgentClient(model_name="gpt-4")
        
        # 테스트 검증
        mock_exists.assert_called_once_with(".private.openai_token")
        mock_file.assert_called_once_with(".private.openai_token", "r")
        mock_openai.assert_called_once_with(api_key="test-api-token")
        self.assertEqual(client._model_name, "gpt-4")
        self.assertEqual(client._chat_completion_client, mock_client.chat.completions)
    
    @patch.object(OpenAIAgentClient, '_get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAI')
    def test_create_agent_exception(self, mock_openai, mock_get_token):
        # 예외 발생 시뮬레이션
        mock_get_token.side_effect = ValueError("Token error")
        
        # 예외 발생 검증
        with self.assertRaises(ValueError) as context:
            OpenAIAgentClient(model_name="gpt-4")
        self.assertIn("Token error", str(context.exception))
    
    @patch.object(OpenAIAgentClient, '_get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAI')
    def test_create_params_basic(self, mock_openai, mock_get_token):
        # 기본 설정
        mock_get_token.return_value = "test-token"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        client = OpenAIAgentClient(model_name="gpt-4")
        
        # 테스트 데이터
        message = Message(index=0, role=RoleEnum.USER, content="Hello")
        completion_create = CompletionCreate(messages=[message])
        
        # 함수 호출
        params = client._create_params(completion_create)
        
        # 검증
        self.assertEqual(params["model"], "gpt-4")
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Hello")
    
    @patch.object(OpenAIAgentClient, '_get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAI')
    def test_create_params_with_tools(self, mock_openai, mock_get_token):
        # 기본 설정
        mock_get_token.return_value = "test-token"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        client = OpenAIAgentClient(model_name="gpt-4")
        
        # 테스트 데이터
        message = Message(index=0, role=RoleEnum.USER, content="Hello")
        tool = Tool(type="function")
        function_item = FunctionItem(name="test_function")
        tool_choice = ToolChoice(function=function_item)
        
        completion_create = CompletionCreate(
            messages=[message],
            tools=[tool],
            tool_choice=tool_choice
        )
        
        # 함수 호출
        params = client._create_params(completion_create)
        
        # 검증
        self.assertEqual(params["model"], "gpt-4")
        self.assertEqual(len(params["tools"]), 1)
        self.assertEqual(params["tools"][0]["type"], "function")
        self.assertEqual(params["tool_choice"]["type"], "function")
        self.assertEqual(params["tool_choice"]["function"]["name"], "test_function")
    
    @patch.object(OpenAIAgentClient, '_get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAI')
    def test_completion(self, mock_openai, mock_get_token):
        # 기본 설정
        mock_get_token.return_value = "test-token"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # API 응답 모킹
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # RoleEnum에 'assistant'가 없으므로 SYSTEM으로 변경
        mock_message.role = RoleEnum.SYSTEM
        mock_message.content = "Hello, I am an AI."
        mock_choice.message = mock_message
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIAgentClient(model_name="gpt-4")
        
        # 테스트 데이터
        message = Message(index=0, role=RoleEnum.USER, content="Who are you?")
        completion_create = CompletionCreate(messages=[message])
        
        # 함수 호출
        result = client.completion(completion_create)
        
        # 검증
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].index, 0)
        self.assertEqual(result[0].message.content, "Hello, I am an AI.")
        self.assertEqual(result[0].finish_reason, FinishReasonEnum.STOP)
    
    @patch.object(OpenAIAgentClient, '_get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAI')
    def test_completion_with_tool_calls(self, mock_openai, mock_get_token):
        # 기본 설정
        mock_get_token.return_value = "test-token"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # API 응답 모킹
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()
        mock_function = MagicMock()
        
        mock_function.name = "test_function"
        mock_function.arguments = '{"arg1": "value1"}'
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function
        mock_message.role = RoleEnum.SYSTEM  # assistant -> SYSTEM으로 변경
        mock_message.content = "I'll help you with that."
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice.message = mock_message
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIAgentClient(model_name="gpt-4")
        
        # 테스트 데이터
        message = Message(index=0, role=RoleEnum.USER, content="Call a function")
        tool = Tool(type="function")
        completion_create = CompletionCreate(messages=[message], tools=[tool])
        
        # 함수 호출
        result = client.completion(completion_create)
        
        # 검증
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].tool_calls)
        self.assertEqual(len(result[0].tool_calls), 1)
        self.assertEqual(result[0].tool_calls[0].id, "call_123")
        self.assertEqual(result[0].tool_calls[0].name, "test_function")
        self.assertEqual(result[0].tool_calls[0].arguments, '{"arg1": "value1"}')
    
    @patch.object(OpenAIAgentClient, '_create_params')
    @patch.object(OpenAIAgentClient, '_get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAI')
    def test_completion_exception(self, mock_openai, mock_get_token, mock_create_params):
        # 기본 설정
        mock_get_token.return_value = "test-token"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # 예외 발생 시뮬레이션
        mock_create_params.side_effect = ValueError("Invalid parameters")
        
        client = OpenAIAgentClient(model_name="gpt-4")
        
        # 테스트 데이터
        message = Message(index=0, role=RoleEnum.USER, content="Hello")
        completion_create = CompletionCreate(messages=[message])
        
        # 예외 발생 검증
        with self.assertRaises(ValueError) as context:
            client.completion(completion_create)
        self.assertIn("Invalid parameters", str(context.exception))

if __name__ == '__main__':
    unittest.main() 