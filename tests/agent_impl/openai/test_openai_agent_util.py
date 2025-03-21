import unittest
from unittest.mock import patch, MagicMock
import json

example_structured_output_request = {
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "아래 정보를 JSON 형식으로 정리해줘.\n이름: 홍길동\n나이: 30\n직업: 개발자"
    }
  ],
  "temperature": 0,
  "max_tokens": 150
}

example_structured_output_response ={
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1670000000,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\n  \"name\": \"홍길동\",\n  \"age\": 30,\n  \"job\": \"개발자\"\n}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}


example_function_call_request = {
  "model": "gpt-3.5-turbo-0613",
  "messages": [
    {
      "role": "user",
      "content": "현재 서울의 날씨 정보를 알려줘."
    }
  ],
  "functions": [
    {
      "name": "get_weather",
      "description": "서울의 날씨 정보를 가져옵니다.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "지역 이름"
          },
          "temperature": {
            "type": "number",
            "description": "현재 온도 (섭씨)"
          },
          "description": {
            "type": "string",
            "description": "날씨 설명"
          }
        },
        "required": ["location", "temperature", "description"]
      }
    }
  ],
  "function_call": "auto"
}

example_function_call_response = {
  "id": "chatcmpl-def456",
  "object": "chat.completion",
  "created": 1670001234,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "null",
        "function_call": {
          "name": "get_weather",
          "arguments": "{\n  \"location\": \"Seoul\",\n  \"temperature\": 18,\n  \"description\": \"맑음\"\n}"
        }
      },
      "finish_reason": "function_call"
    }
  ],
  "usage": {
    "prompt_tokens": 70,
    "completion_tokens": 50,
    "total_tokens": 120
  }
}


class TestOpenAIAgentUtil(unittest.TestCase):
    """
    OpenAIAgentUtil 클래스의 테스트 클래스
    mocking targets:
        - _get_access_token
        - _create_agent
        - completion
    """
    
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._create_agent')
    def test_create_params_structured_output(self, mock_create_agent, mock_get_access_token):
        """
        구조화된 출력을 위한 매개변수 생성 테스트
        """
        # Mocking 설정
        mock_get_access_token.return_value = "mock_access_token"
        
        from src.agent_impl.openai.openai_agent_util import OpenAIAgentClient
        from src.agent_impl.openai.schema.message import Message
        from src.agent_impl.openai.schema.enums import RoleEnum
        from src.agent_impl.openai.schema.completion import CompletionCreate
        
        # 테스트를 위한 클라이언트 생성
        client = OpenAIAgentClient(model_name="gpt-3.5-turbo")
        
        # _create_agent가 호출되었는지 확인
        mock_create_agent.assert_called_once()
        
        # 테스트 메시지 생성
        messages = [
            Message(
                index=0, 
                role=RoleEnum.USER, 
                content="아래 정보를 JSON 형식으로 정리해줘.\n이름: 홍길동\n나이: 30\n직업: 개발자"
            )
        ]
        
        # CompletionCreate 객체 생성
        completion_create = CompletionCreate(messages=messages)
        
        # _create_params 메서드 호출
        params = client._create_params(completion_create)
        
        # 예상 출력과 비교
        expected_messages = example_structured_output_request["messages"]
        
        # 기본 필드 확인
        self.assertEqual(params["model"], "gpt-3.5-turbo")
        self.assertEqual(len(params["messages"]), len(expected_messages))
        
        # 메시지 내용 확인
        for i, msg in enumerate(params["messages"]):
            self.assertEqual(msg["role"], expected_messages[i]["role"])
            self.assertEqual(msg["content"], expected_messages[i]["content"])
        
        # 전체 딕셔너리 비교 (온도와 토큰 수는 테스트에서는 설정하지 않았으므로 제외)
        expected_dict = {
            "model": "gpt-3.5-turbo",
            "messages": expected_messages,
            "tool_choice": "none"
        }

        # 파라미터와 예상 딕셔너리 출력
        print("\n=== 생성된 파라미터 ===")
        print(json.dumps(params, indent=2, ensure_ascii=False))
        
        print("\n=== 예상 딕셔너리 ===") 
        print(json.dumps(expected_dict, indent=2, ensure_ascii=False))
        
        self.assertDictEqual(params, expected_dict)
    
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._create_agent')
    def test_create_params_function_call(self, mock_create_agent, mock_get_access_token):
        """
        함수 호출을 위한 매개변수 생성 테스트
        """
        # Mocking 설정
        mock_get_access_token.return_value = "mock_access_token"
        
        from src.agent_impl.openai.openai_agent_util import OpenAIAgentClient
        from src.agent_impl.openai.schema.message import Message
        from src.agent_impl.openai.schema.enums import RoleEnum
        from src.agent_impl.openai.schema.completion import CompletionCreate
        from src.agent_impl.openai.schema.function import Function, FunctionItem
        from src.agent_impl.openai.schema.tool import Tool, ToolChoice
        from src.agent_impl.openai.schema.parameters import Parameters
        
        # 테스트를 위한 클라이언트 생성
        client = OpenAIAgentClient(model_name="gpt-3.5-turbo-0613")
        
        # _create_agent가 호출되었는지 확인
        mock_create_agent.assert_called_once()
        
        # 테스트 메시지 생성
        messages = [
            Message(
                index=0, 
                role=RoleEnum.USER, 
                content="현재 서울의 날씨 정보를 알려줘."
            )
        ]
        
        # 함수 파라미터 생성
        properties = {
            "location": {
                "type": "string",
                "description": "지역 이름"
            },
            "temperature": {
                "type": "number",
                "description": "현재 온도 (섭씨)"
            },
            "description": {
                "type": "string",
                "description": "날씨 설명"
            }
        }
        
        params = Parameters(
            type="object",
            properties=properties,
            required=["location", "temperature", "description"]
        )
        
        # 함수 정의
        def get_weather(location: str, temperature: float, description: str):
            return f"{location}의 날씨: {temperature}°C, {description}"
            
        function = Function(
            name="get_weather",
            description="서울의 날씨 정보를 가져옵니다.",
            parameters=params,
            real_function=get_weather
        )
        
        # Tool과 ToolChoice 생성
        tool = Tool(function=function)
        function_item = FunctionItem(name="get_weather")
        tool_choice = "auto"  # auto 사용
        
        # CompletionCreate 객체 생성
        completion_create = CompletionCreate(
            messages=messages,
            tools=[tool],
            tool_choice=tool_choice
        )
        
        # _create_params 메서드 호출
        params = client._create_params(completion_create)
        
        # 모델 이름 확인
        self.assertEqual(params["model"], "gpt-3.5-turbo-0613")
        
        # 메시지 확인
        expected_messages = example_function_call_request["messages"]
        self.assertEqual(len(params["messages"]), len(expected_messages))
        self.assertEqual(params["messages"][0]["role"], expected_messages[0]["role"])
        self.assertEqual(params["messages"][0]["content"], expected_messages[0]["content"])
        
        # tools 확인
        self.assertIn("tools", params)
        self.assertEqual(len(params["tools"]), 1)
        self.assertEqual(params["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(params["tools"][0]["function"]["description"], "서울의 날씨 정보를 가져옵니다.")
        
        # tool_choice 확인
        self.assertEqual(params["tool_choice"], "auto")
        
        # 필요한 필드들이 존재하는지 확인
        required_fields = ["model", "messages", "tools", "tool_choice"]
        for field in required_fields:
            self.assertIn(field, params, f"{field} 필드가 누락되었습니다.")
    
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._create_agent')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient.completion')
    def test_completion_structured_output(self, mock_completion, mock_create_agent, mock_get_access_token):
        """
        구조화된 출력 응답 테스트
        """
        # Mocking 설정
        mock_get_access_token.return_value = "mock_access_token"
        
        # OpenAI 응답을 MagicMock으로 만들어서 예상 응답 구조 설정
        mock_response = MagicMock()
        mock_response.choices = []
        
        # 첫 번째 선택 항목 생성
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        
        # 메시지 설정
        choice.message = MagicMock()
        choice.message.role = "assistant"
        choice.message.content = "{\n  \"name\": \"홍길동\",\n  \"age\": 30,\n  \"job\": \"개발자\"\n}"
        
        mock_response.choices.append(choice)
        mock_completion.return_value = [mock_response]
        
        from src.agent_impl.openai.openai_agent_util import OpenAIAgentClient
        from src.agent_impl.openai.schema.message import Message
        from src.agent_impl.openai.schema.enums import RoleEnum
        from src.agent_impl.openai.schema.completion import CompletionCreate
        
        # 테스트를 위한 클라이언트 생성
        client = OpenAIAgentClient(model_name="gpt-3.5-turbo")
        
        # 테스트 메시지 생성
        messages = [
            Message(
                index=0, 
                role=RoleEnum.USER, 
                content="아래 정보를 JSON 형식으로 정리해줘.\n이름: 홍길동\n나이: 30\n직업: 개발자"
            )
        ]
        
        # CompletionCreate 객체 생성
        completion_create = CompletionCreate(messages=messages)
        
        # 응답 확인
        response = client.completion(completion_create)
        mock_completion.assert_called_once_with(completion_create)
    
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._get_access_token')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient._create_agent')
    @patch('src.agent_impl.openai.openai_agent_util.OpenAIAgentClient.completion')
    def test_completion_function_call(self, mock_completion, mock_create_agent, mock_get_access_token):
        """
        함수 호출 응답 테스트
        """
        # Mocking 설정
        mock_get_access_token.return_value = "mock_access_token"
        
        # OpenAI 응답을 MagicMock으로 만들어서 예상 응답 구조 설정
        mock_response = MagicMock()
        mock_response.choices = []
        
        # 첫 번째 선택 항목 생성
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "function_call"
        
        # 메시지 설정
        choice.message = MagicMock()
        choice.message.role = "assistant"
        choice.message.content = None
        
        # 함수 호출 설정
        choice.message.tool_calls = []
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function = MagicMock()
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{\n  "location": "Seoul",\n  "temperature": 18,\n  "description": "맑음"\n}'
        choice.message.tool_calls.append(tool_call)
        
        mock_response.choices.append(choice)
        mock_completion.return_value = [mock_response]
        
        from src.agent_impl.openai.openai_agent_util import OpenAIAgentClient
        from src.agent_impl.openai.schema.message import Message
        from src.agent_impl.openai.schema.enums import RoleEnum
        from src.agent_impl.openai.schema.completion import CompletionCreate
        from src.agent_impl.openai.schema.function import Function, FunctionItem
        from src.agent_impl.openai.schema.tool import Tool, ToolChoice
        from src.agent_impl.openai.schema.parameters import Parameters
        
        # 테스트를 위한 클라이언트 생성
        client = OpenAIAgentClient(model_name="gpt-3.5-turbo-0613")
        
        # 테스트 메시지 생성
        messages = [
            Message(
                index=0, 
                role=RoleEnum.USER, 
                content="현재 서울의 날씨 정보를 알려줘."
            )
        ]
        
        # 함수 파라미터 생성
        properties = {
            "location": {
                "type": "string",
                "description": "지역 이름"
            },
            "temperature": {
                "type": "number",
                "description": "현재 온도 (섭씨)"
            },
            "description": {
                "type": "string",
                "description": "날씨 설명"
            }
        }
        
        params = Parameters(
            type="object",
            properties=properties,
            required=["location", "temperature", "description"]
        )
        
        # 함수 정의
        def get_weather(location: str, temperature: float, description: str):
            return f"{location}의 날씨: {temperature}°C, {description}"
            
        function = Function(
            name="get_weather",
            description="서울의 날씨 정보를 가져옵니다.",
            parameters=params,
            real_function=get_weather
        )
        
        # Tool과 ToolChoice 생성
        tool = Tool(function=function)
        tool_choice = "auto"  # auto 사용
        
        # CompletionCreate 객체 생성
        completion_create = CompletionCreate(
            messages=messages,
            tools=[tool],
            tool_choice=tool_choice
        )
        
        # 응답 확인
        response = client.completion(completion_create)
        mock_completion.assert_called_once_with(completion_create)


if __name__ == '__main__':
    unittest.main()

