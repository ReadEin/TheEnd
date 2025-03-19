from textwrap import dedent
import unittest
from src.prompt.cot_pattern.cot_prompt import CotSchema, CotPtrompt

class CotPromptTest(unittest.TestCase):
    def setUp(self):
        # 테스트에 사용할 기본 데이터
        pass

    def test_basic_formatting(self):  # sourcery skip: class-extract-method
        # 기본적인 포맷팅 테스트
        test_schema = CotSchema(
            direction="테스트 방향",
            chain_of_thought=["단계1", "단계2", "단계3"],
            background="테스트 배경 지식"
        )
        prompt = CotPtrompt(test_schema)
        result = prompt.from_arg()
        
        expected = dedent("""
            지시사항 :
            테스트 방향
            문제 해결 절차 :
            1. 단계1
            2. 단계2
            3. 단계3
            배경 지식 :
            테스트 배경 지식
            """)
        self.assertEqual(result.strip(), expected.strip())

    def test_empty_chain_of_thought(self):
        # 빈 chain_of_thought 테스트
        test_schema = CotSchema(
            direction="테스트 방향",
            chain_of_thought=[],
            background="테스트 배경 지식"
        )
        prompt = CotPtrompt(test_schema)
        result = prompt.from_arg()
        
        expected = dedent("""
            지시사항 :
            테스트 방향
            문제 해결 절차 :
            
            배경 지식 :
            테스트 배경 지식
            """)
        print(result)
        print(expected)
        self.assertEqual(result.strip(), expected.strip())
if __name__ == '__main__':
    unittest.main()