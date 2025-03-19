from textwrap import dedent
import unittest
from src.prompt.jformatter_pattern.jformatter_prompt import JFormatterSchema, JFormatterPrompt

class JFormatterPromptTest(unittest.TestCase):
    def setUp(self):
        # 테스트에 사용할 기본 데이터
        pass

    def test_basic_formatting(self):
        # 기본적인 포맷팅 테스트
        test_schema = JFormatterSchema(
            content="테스트 내용입니다. 이것은 JSON으로 변환될 예정입니다.",
            json_keys=["key1", "key2", "key3"]
        )
        prompt = JFormatterPrompt(test_schema)
        result = prompt.from_arg()
        
        expected = dedent("""
            페르소나 :
            당신은 텍스트를 구조화된 JSON 형식으로 변환하는 전문가입니다.
            
            역할 :
            주어진 내용을 분석하고 지정된 JSON 키를 사용하여 정보를 요약-정리합니다.
            
            입력 내용 :
            테스트 내용입니다. 이것은 JSON으로 변환될 예정입니다.
            
            사용할 JSON 키 :
            - key1
            - key2
            - key3
            
            지시사항 :
            1. 위 내용을 분석하여 중요한 정보를 추출하세요
            2. 추출한 정보를 제공된 JSON 키에 맞게 정리하세요
            3. 올바른 JSON 형식으로 출력하세요
            4. 모든 키에 적절한 값을 채우세요
            """)
        self.assertEqual(result.strip(), expected.strip())

    def test_empty_json_keys(self):
        # 빈 json_keys 테스트
        test_schema = JFormatterSchema(
            content="테스트 내용입니다. 이것은 JSON으로 변환될 예정입니다.",
            json_keys=[]
        )
        prompt = JFormatterPrompt(test_schema)
        result = prompt.from_arg()
        
        expected = dedent("""
            페르소나 :
            당신은 텍스트를 구조화된 JSON 형식으로 변환하는 전문가입니다.
            
            역할 :
            주어진 내용을 분석하고 지정된 JSON 키를 사용하여 정보를 요약-정리합니다.
            
            입력 내용 :
            테스트 내용입니다. 이것은 JSON으로 변환될 예정입니다.
            
            사용할 JSON 키 :
            
            
            지시사항 :
            1. 위 내용을 분석하여 중요한 정보를 추출하세요
            2. 추출한 정보를 제공된 JSON 키에 맞게 정리하세요
            3. 올바른 JSON 형식으로 출력하세요
            4. 모든 키에 적절한 값을 채우세요
            """)
        self.assertEqual(result.strip(), expected.strip())

if __name__ == '__main__':
    unittest.main() 