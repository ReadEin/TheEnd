import unittest
from unittest.mock import patch, MagicMock

class TestOpenAIAgent(unittest.TestCase):
    
    @patch('openai.OpenAI')
    def test_openai_agent_example(self, mock_openai):
        """
        openai_agent.py 예제 스크립트가 오류 없이 실행되는지 테스트합니다.
        
        실제로 OpenAI API를 호출하지 않고 모킹을 통해 테스트합니다.
        """
        # OpenAI 클라이언트 모킹
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # 응답 모킹
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # 메시지 내용 설정
        mock_message.content = "Once upon a time, a magical unicorn flew over the rainbow and fell asleep under a glittering star."
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        
        # 클라이언트 응답 설정
        mock_client.chat.completions.create.return_value = mock_completion
        
        # 다른 모듈의 함수를 패치하고 테스트
        with patch('builtins.print') as mock_print:
            # 스크립트 실행을 시뮬레이션
            import src.agent_impl.openai.openai_agent
            
            # print 함수 호출 확인
            mock_print.assert_called_once_with(
                "Once upon a time, a magical unicorn flew over the rainbow and fell asleep under a glittering star."
            )
            
            # API 호출 확인
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args[1]
            
            # API 호출 인자 확인
            self.assertEqual(call_args["model"], "gpt-4o")
            self.assertEqual(len(call_args["messages"]), 1)
            self.assertEqual(call_args["messages"][0]["role"], "user")
            self.assertTrue("bedtime story about a unicorn" in call_args["messages"][0]["content"])

if __name__ == '__main__':
    unittest.main() 