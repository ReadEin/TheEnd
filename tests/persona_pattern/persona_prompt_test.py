from textwrap import dedent
import unittest
from src.prompt.persona_pattern.persona_prompt import PersonaSchema, PersonaPrompt

class PersonaPromptTest(unittest.TestCase):
    def setUp(self):
        # 테스트에 사용할 기본 데이터
        pass

    def test_basic_formatting(self):  # sourcery skip: class-extract-method
        # 기본적인 포맷팅 테스트
        test_schema = PersonaSchema(
            persona="테스트 페르소나",
            traits=["특성1", "특성2", "특성3"],
            background="테스트 배경 지식",
            instruction="테스트 지시사항"
        )
        prompt = PersonaPrompt(test_schema)
        result = prompt.from_arg()
        
        expected = dedent("""
            페르소나 :
            테스트 페르소나
            
            페르소나 특성 :
            - 특성1
            - 특성2
            - 특성3
            
            배경 지식 :
            테스트 배경 지식
            
            지시사항 :
            테스트 지시사항
            """)
        self.assertEqual(result.strip(), expected.strip())

    def test_empty_traits(self):
        # 빈 traits 테스트
        test_schema = PersonaSchema(
            persona="테스트 페르소나",
            traits=[],
            background="테스트 배경 지식",
            instruction="테스트 지시사항"
        )
        prompt = PersonaPrompt(test_schema)
        result = prompt.from_arg()
        
        expected = dedent("""
            페르소나 :
            테스트 페르소나
            
            페르소나 특성 :
            
            
            배경 지식 :
            테스트 배경 지식
            
            지시사항 :
            테스트 지시사항
            """)
        self.assertEqual(result.strip(), expected.strip())

if __name__ == '__main__':
    unittest.main() 