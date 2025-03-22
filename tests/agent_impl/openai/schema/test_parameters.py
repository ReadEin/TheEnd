from src.agent_impl.openai.schema.parameters import Parameters
import unittest

expected_output:dict = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "weather": {
                "type": "string",
                "description": "Weather for a given location."
            }
        },
        "additionalProperties": False,
        "required": ["location"]
    }

class TestParameters(unittest.TestCase):
    def test_parameters_schema(self):
        """
        Parameters 클래스의 스키마가 예상대로 생성되는지 테스트합니다.
        """
        # 속성 정의
        location = {
            "type": "string",
            "description": "City and country e.g. Bogotá, Colombia"
        }

        weather = {
            "type": "string",
            "description": "Weather for a given location."
        }
        
        properties = {
            "location": location,
            "weather": weather
        }
        
        # Parameters 인스턴스 생성
        params = Parameters(
            type="object",
            properties=properties,
            required=["location"],
            additionalProperties=False
        )
        
        # model_dump()를 사용하여 딕셔너리 변환 (Pydantic v2)
        import json
        
        # Parameters 모델을 JSON으로 변환 (alias 사용)
        params_model_dump_result = params.model_dump(by_alias=True)
        
        # 출력된 스키마가 예상 출력과 일치하는지 확인
        print(f"watch params_model_dump_result: {params_model_dump_result}")
        print(f"watch expected_output: {expected_output}")

        self.assertDictEqual(params_model_dump_result, expected_output, "두 파라미터 딕셔너리가 일치하지 않습니다.")


if __name__ == '__main__':
    unittest.main() 