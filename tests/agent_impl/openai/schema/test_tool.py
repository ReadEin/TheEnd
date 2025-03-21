from src.agent_impl.openai.schema.tool import Tool, ToolChoice
from src.agent_impl.openai.schema.function import Function, FunctionItem
from src.agent_impl.openai.schema.parameters import Parameters
import unittest

expected_tool_output:dict = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": None,
            "additionalProperties": False
        }
    }
}

expected_tool_choice_output:dict = {
    "type": "function",
    "function": {
        "name": "get_weather"
    }
}

class TestTool(unittest.TestCase):
    def test_tool_schema(self):
        """
        Tool 클래스의 스키마가 예상대로 생성되는지 테스트합니다.
        """
        # 위치 패러미터 정의
        location = {
            "type": "string",
            "description": "City and country e.g. Bogotá, Colombia"
        }
        
        properties = {
            "location": location
        }
        
        # 파라미터 생성
        params = Parameters(
            type="object",
            properties=properties,
            additional_properties=False
        )
        
        # Function 생성
        def mock_function():
            pass
            
        function = Function(
            name="get_weather",
            description="Get current temperature for a given location.",
            parameters=params,
            real_function=mock_function
        )
        
        # Tool 생성
        tool = Tool(function=function)
        
        # model_dump() 사용
        tool_model_dump_result = tool.model_dump(exclude={"function": {"real_function"}}, by_alias=True)
        
        # 출력된 스키마가 예상 출력과 일치하는지 확인
        print(f"watch tool_model_dump_result: {tool_model_dump_result}")
        print(f"watch expected_tool_output: {expected_tool_output}")
        
        self.assertEqual(tool_model_dump_result["type"], expected_tool_output["type"])
        self.assertEqual(tool_model_dump_result["function"]["name"], expected_tool_output["function"]["name"])
        self.assertEqual(tool_model_dump_result["function"]["description"], expected_tool_output["function"]["description"])

        import json
        print(f"watch tool_model_dump_result: {json.dumps(tool_model_dump_result, indent=2)}")
        print(f"watch expected_tool_output: {json.dumps(expected_tool_output, indent=2)}")
        self.assertDictEqual(tool_model_dump_result, expected_tool_output, "두 Tool 딕셔너리가 일치하지 않습니다.")
        

class TestToolChoice(unittest.TestCase):
    def test_tool_choice_schema(self):
        """
        ToolChoice 클래스의 스키마가 예상대로 생성되는지 테스트합니다.
        """
        function_item = FunctionItem(name="get_weather")
        tool_choice = ToolChoice(function=function_item)
        
        # model_dump() 사용
        tool_choice_model_dump_result = tool_choice.model_dump(by_alias=True)
        
        # 출력된 스키마가 예상 출력과 일치하는지 확인
        print(f"watch tool_choice_model_dump_result: {tool_choice_model_dump_result}")
        print(f"watch expected_tool_choice_output: {expected_tool_choice_output}")
        
        self.assertDictEqual(tool_choice_model_dump_result, expected_tool_choice_output, "두 ToolChoice 딕셔너리가 일치하지 않습니다.")


if __name__ == '__main__':
    unittest.main()
