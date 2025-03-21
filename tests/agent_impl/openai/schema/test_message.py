from src.agent_impl.openai.schema.message import Message, Role
from src.agent_impl.openai.schema.enums import RoleEnum
import unittest

class TestRole(unittest.TestCase):
    def test_role_creation(self):
        """
        Role 클래스가 예상대로 생성되는지 테스트합니다.
        """
        role = Role(role=RoleEnum.USER)
        self.assertEqual(role.role, RoleEnum.USER)
        
        role_system = Role(role=RoleEnum.SYSTEM)
        self.assertEqual(role_system.role, RoleEnum.SYSTEM)
        
    def test_role_validation(self):
        """
        Role 클래스의 유효성 검사가 정상적으로 동작하는지 테스트합니다.
        """
        # 유효한 역할은 정상적으로 생성됨
        role = Role(role=RoleEnum.USER)
        self.assertEqual(role.role, RoleEnum.USER)
        
        # 문자열 직접 사용 (Pydantic이 자동 변환)
        role = Role(role="user")
        self.assertEqual(role.role, RoleEnum.USER)
        

class TestMessage(unittest.TestCase):
    def test_message_creation(self):
        """
        Message 클래스가 예상대로 생성되는지 테스트합니다.
        """
        message = Message(
            index=0,
            role=RoleEnum.USER,
            content="Hello, world!"
        )
        
        self.assertEqual(message.index, 0)
        self.assertEqual(message.role, RoleEnum.USER)
        self.assertEqual(message.content, "Hello, world!")
        
    def test_message_serialization(self):
        """
        Message 클래스의 직렬화가 정상적으로 동작하는지 테스트합니다.
        """
        message = Message(
            index=1,
            role=RoleEnum.SYSTEM,
            content="System message"
        )
        
        message_dict = message.model_dump()
        expected = {
            "index": 1,
            "role": "system",
            "content": "System message"
        }
        
        self.assertEqual(message_dict["index"], expected["index"])
        self.assertEqual(message_dict["role"], expected["role"])
        self.assertEqual(message_dict["content"], expected["content"])


if __name__ == '__main__':
    unittest.main() 