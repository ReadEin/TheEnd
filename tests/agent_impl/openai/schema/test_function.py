from src.agent_impl.openai.schema.function import Function
from src.agent_impl.openai.schema.parameters import Parameters
import unittest

expected_output:dict = {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
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
    }

class TestFunction(unittest.TestCase):
    def test_function_schema(self):
        """
        Function 클래스의 스키마가 예상대로 생성되는지 테스트합니다.
        """
        # 위치 패러미터를 정의
        location = {
            "type": "string",
            "description": "City and country e.g. Bogotá, Colombia"
        }

        weather = {
            "type" : "string",
            "description" : "Weather for a given location."
        }
        
        properties = {
            "location" : location,
            "weather" : weather
        }
        # Function 인스턴스 생성
        params = Parameters(
            type = "object",
            properties = properties,
            required = ["location"],
            additional_properties = False
        )
        
        def mock_function():
            pass
            
        function = Function(
            name="get_weather",
            description="Get current temperature for a given location.",
            parameters=params,
            real_function=mock_function
        )
        
        # model_dump()를 사용하여 딕셔너리 변환 (Pydantic v2)
        import json
        
        # Function 모델을 JSON으로 변환
        function_model_dump_result = function.model_dump(exclude={"real_function"}, by_alias=True)
        
        # 출력된 스키마가 예상 출력과 일치하는지 확인
        self.assertEqual(function_model_dump_result["name"], expected_output["name"])
        self.assertEqual(function_model_dump_result["description"], expected_output["description"])

        print(f"watch function_model_dump_result: {function_model_dump_result['parameters']}")
        print(f"watch expected_output: {expected_output['parameters']}")

        self.assertDictEqual(function_model_dump_result["parameters"], expected_output["parameters"], "두 파라미터 딕셔너리가 일치하지 않습니다.")


if __name__ == '__main__':
    unittest.main()
