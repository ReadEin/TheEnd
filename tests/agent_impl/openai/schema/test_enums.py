from src.agent_impl.openai.schema.enums import FinishReasonEnum, RoleEnum
import unittest
from openai import OpenAI 

class TestFinishReasonEnum(unittest.TestCase):
    def test_finish_reason_enum_values(self):
        """
        FinishReasonEnum의 값이 예상대로 설정되어 있는지 테스트합니다.
        """
        self.assertEqual(FinishReasonEnum.STOP, "stop")
        self.assertEqual(FinishReasonEnum.LENGTH, "length")
        self.assertEqual(FinishReasonEnum.CONTENT_FILTER, "content_filter")
        
    def test_finish_reason_enum_comparison(self):
        """
        FinishReasonEnum의 비교 연산이 정상적으로 동작하는지 테스트합니다.
        """
        self.assertEqual(FinishReasonEnum.STOP, FinishReasonEnum.STOP)
        self.assertNotEqual(FinishReasonEnum.STOP, FinishReasonEnum.LENGTH)
        self.assertEqual(FinishReasonEnum.STOP, "stop")
        

class TestRoleEnum(unittest.TestCase):
    def test_role_enum_values(self):
        """
        RoleEnum의 값이 예상대로 설정되어 있는지 테스트합니다.
        """
        self.assertEqual(RoleEnum.USER, "user")
        self.assertEqual(RoleEnum.SYSTEM, "system")
        
    def test_role_enum_comparison(self):
        """
        RoleEnum의 비교 연산이 정상적으로 동작하는지 테스트합니다.
        """
        self.assertEqual(RoleEnum.USER, RoleEnum.USER)
        self.assertNotEqual(RoleEnum.USER, RoleEnum.SYSTEM)
        self.assertEqual(RoleEnum.SYSTEM, "system")


if __name__ == '__main__':
    unittest.main() 