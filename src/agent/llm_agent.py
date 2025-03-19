from typing import Any, TypeVar
from langchain_core.language_models import BaseLanguageModel
from src.prompt.prompt_sentence import Prompt_Schema, PromptSentence

class LLMAgent:
    """언어 모델을 사용하여 프롬프트를 처리하고 결과를 반환하는 에이전트"""
    
    def __init__(self, language_model: BaseLanguageModel):
        """
        LLMAgent 초기화
        
        Args:
            language_model: LangChain의 BaseLanguageModel 인스턴스
        """
        self.language_model = language_model
    
    def run(self, prompt_sentence: PromptSentence[Prompt_Schema]) -> str:
        """
        프롬프트 문장을 언어 모델에 전달하고 결과를 반환
        
        Args:
            prompt_sentence: 언어 모델에 전달할 프롬프트
            
        Returns:
            언어 모델이 생성한 응답 텍스트
        """
        # 프롬프트 문장에서 형식화된 프롬프트 텍스트 생성
        prompt_text = prompt_sentence.from_arg()
        
        # LangChain의 언어 모델에 프롬프트 전달하고 응답 받기
        response = self.language_model.invoke(prompt_text)
        
        return response
