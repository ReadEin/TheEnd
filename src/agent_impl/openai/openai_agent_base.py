from enum import Enum
from pydantic import BaseModel
from src.agent_impl.openai.openai_agent_util import OpenAIAgentClient
from src.agent_impl.openai.schema.completion import CompletionChoice, CompletionCreate
from src.agent_impl.openai.schema.function import Function
from src.agent_impl.openai.schema.message import Message
from src.agent_impl.openai.schema.tool import Tool, ToolChoice

class ToolChoiceType(Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"

class RequestPrepare():
    _system_message: str
    _tools : list[Tool]
    _tool_choice : ToolChoice = "none"
    _response_format : type[BaseModel] = None
    
    def __init__(
        self,
        system_message: str,
        tools: list[Tool],
        tool_choice: ToolChoice = "none",
        response_format: type[BaseModel] = None
    ):
        self._system_message = system_message
        self._tools = tools
        self._tool_choice = tool_choice
        self._response_format = response_format
        
    def get_functions(self) -> dict[str, Function]:
        return {tool.function.name: tool.function for tool in self._tools}
    
    def set_history(self, history: list[Message]) -> 'RequestBase':
        return RequestBase(self, history)
    
    def get_response_format(self) -> type[BaseModel]:
        return self._response_format

class RequestBase():
    _prepare : RequestPrepare
    _history : list[Message]
    _functions : dict[str, Function]
    
    def __init__(self, prepare: RequestPrepare, history: list[Message] = None):
        if history is None:
            history = []
        self._prepare = prepare
        self._history = history
        self._functions = prepare.get_functions()
        
    def of(self, content: str) -> CompletionCreate:
        completion_create = CompletionCreate(
            messages=self._history + [Message(role="user", content=content)],
            tools=self._prepare._tools,
            tool_choice=self._prepare._tool_choice,
            response_format=self._prepare._response_format
        )
        print(f"completion_create: {completion_create}")
        return completion_create
    
    def get_functions(self) -> dict[str, Function]:
        return self._functions
    def set_functions(self, functions: dict[str, Function]) -> 'RequestBase':
        self._functions = functions
        return self
    def get_history(self) -> list[Message]:
        return self._history
    def set_history(self, history: list[Message]) -> 'RequestBase':
        self._history = history
        return self
    def get_prepare(self) -> RequestPrepare:
        return self._prepare
    def set_prepare(self, prepare: RequestPrepare) -> 'RequestBase':
        self._prepare = prepare
        return self

class OpenAIAgentBase:
    _model_name: str
    _client:OpenAIAgentClient
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._client = OpenAIAgentClient(model_name)
    
    def request(self, request: RequestBase, content: str) -> str :
        firstChoice = self._client.completion(request.of(content))[0]
        
        if firstChoice.tool_calls and len(firstChoice.tool_calls) > 0:
            tool_calls = firstChoice.tool_calls
            request_functions = request.get_functions()
            for tool_call in tool_calls:
                function = request_functions[tool_call.name]
                function.real_function(tool_call.arguments)
        return firstChoice.message.content ## parse with request.get_response_format()

    def get_client(self) -> OpenAIAgentClient:
        return self._client
    def set_client(self, client: OpenAIAgentClient) -> 'OpenAIAgentBase':
        self._client = client
        return self
    
    
