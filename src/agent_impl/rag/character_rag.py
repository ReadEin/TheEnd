from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from openai import OpenAI
import json
import os

class Character:
    """캐릭터 정보를 저장하는 클래스"""
    def __init__(self, id: str, name: str, description: str, personality: str, backstory: str):
        self.id = id
        self.name = name
        self.description = description  # 외모, 특징 등
        self.personality = personality  # 성격
        self.backstory = backstory      # 배경 이야기
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name, 
            "description": self.description,
            "personality": self.personality,
            "backstory": self.backstory
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Character':
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            personality=data.get("personality", ""),
            backstory=data.get("backstory", "")
        )
    
    def __str__(self) -> str:
        return f"""캐릭터: {self.name}
설명: {self.description}
성격: {self.personality}
배경: {self.backstory}"""


class CharacterRAG:
    """캐릭터 정보에 대한 RAG(Retrieval-Augmented Generation) 시스템"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        RAG 시스템 초기화
        
        Args:
            model_name: 임베딩 생성에 사용할 모델 이름
        """
        self.characters: Dict[str, Character] = {}  # id -> Character
        self.embeddings: Dict[str, List[float]] = {}  # id -> 임베딩 벡터
        self.model_name = model_name
        self._init_client()
    
    def _init_client(self):
        """OpenAI API 클라이언트 초기화"""
        try:
            # 토큰 불러오기
            if os.path.exists(".private.openai_token"):
                with open(".private.openai_token", "r") as f:
                    api_key = f.read().strip()
                self.client = OpenAI(api_key=api_key)
            else:
                raise ValueError("API 토큰을 찾을 수 없습니다.")
        except Exception as e:
            print(f"OpenAI 클라이언트 초기화 오류: {e}")
            raise e
            
    def save_to_file(self, filepath: str):
        """캐릭터 데이터와 임베딩을 파일에 저장"""
        data = {
            "characters": {char_id: char.to_dict() for char_id, char in self.characters.items()},
            "embeddings": self.embeddings
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_from_file(self, filepath: str):
        """파일에서 캐릭터 데이터와 임베딩을 불러옴"""
        if not os.path.exists(filepath):
            print(f"파일이 존재하지 않습니다: {filepath}")
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.characters = {
            char_id: Character.from_dict(char_data) 
            for char_id, char_data in data.get("characters", {}).items()
        }
        self.embeddings = data.get("embeddings", {})
        
    def add_character(self, character: Character, generate_embedding: bool = True):
        """
        캐릭터를 추가하고 선택적으로 임베딩을 생성
        
        Args:
            character: 추가할 캐릭터 객체
            generate_embedding: 임베딩을 생성할지 여부
        """
        self.characters[character.id] = character
        
        if generate_embedding:
            self._generate_embedding(character.id)
            
    def _generate_embedding(self, character_id: str):
        """
        특정 캐릭터에 대한 임베딩 생성
        
        Args:
            character_id: 임베딩을 생성할 캐릭터 ID
        """
        if character_id not in self.characters:
            raise ValueError(f"존재하지 않는 캐릭터 ID: {character_id}")
            
        character = self.characters[character_id]
        
        # 임베딩을 생성할 텍스트 (캐릭터의 모든 정보 통합)
        text = f"캐릭터 이름: {character.name}\n"
        text += f"설명: {character.description}\n"
        text += f"성격: {character.personality}\n"
        text += f"배경 이야기: {character.backstory}"
        
        try:
            # OpenAI API를 사용하여 임베딩 생성
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            # 임베딩 저장
            self.embeddings[character_id] = response.data[0].embedding
            
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            raise e
            
    def generate_all_embeddings(self):
        """모든 캐릭터에 대한 임베딩 생성"""
        for character_id in self.characters:
            self._generate_embedding(character_id)
            
    def _get_embedding(self, text: str) -> List[float]:
        """
        주어진 텍스트의 임베딩 벡터를 가져옴
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            raise e
            
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        두 벡터 간의 코사인 유사도 계산
        
        Args:
            vec1: 첫 번째 벡터
            vec2: 두 번째 벡터
            
        Returns:
            코사인 유사도 (0~1 사이 값)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # 0으로 나누기 방지
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
            
        return dot_product / (norm_vec1 * norm_vec2)
        
    def search(self, query: str, top_k: int = 3, threshold: float = 0.7) -> List[Tuple[Character, float]]:
        """
        쿼리와 가장 유사한 캐릭터를 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            threshold: 최소 유사도 임계값
            
        Returns:
            (캐릭터, 유사도) 튜플의 리스트, 유사도 내림차순 정렬
        """
        if not self.characters or not self.embeddings:
            return []
            
        # 쿼리 임베딩 계산
        query_embedding = self._get_embedding(query)
        
        # 각 캐릭터와의 유사도 계산
        similarities = []
        for char_id, char_embedding in self.embeddings.items():
            if char_id in self.characters:
                similarity = self._calculate_similarity(query_embedding, char_embedding)
                if similarity >= threshold:
                    similarities.append((self.characters[char_id], similarity))
        
        # 유사도에 따라 정렬, 가장 유사한 것부터
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # top_k 결과 반환
        return similarities[:top_k]
        
    def get_relevant_characters_context(self, query: str, top_k: int = 2) -> str:
        """
        쿼리와 관련 있는 캐릭터 정보를 문맥으로 가져옴
        
        Args:
            query: 검색 쿼리
            top_k: 가져올 캐릭터 수
            
        Returns:
            관련 캐릭터 정보를 포함한 문맥 텍스트
        """
        search_results = self.search(query, top_k=top_k)
        
        if not search_results:
            return ""
            
        context = "관련 캐릭터 정보:\n\n"
        for character, similarity in search_results:
            context += f"{str(character)}\n\n"
            
        return context.strip()


# 예시 캐릭터 데이터 (필요시 사용)
def create_sample_characters() -> CharacterRAG:
    """샘플 캐릭터 데이터를 생성하여 CharacterRAG 인스턴스 반환"""
    rag = CharacterRAG()
    
    # 샘플 캐릭터들
    characters = [
        Character(
            id="char1",
            name="김주인공",
            description="키 180cm, 검은 머리, 항상 청바지와 흰 티셔츠를 입음. 오른쪽 눈썹에 작은 흉터가 있음.",
            personality="내성적이지만 친구들에게는 매우 충실함. 정의감이 강하고 옳다고 생각하는 일은 끝까지 밀고 나감.",
            backstory="어린 시절 부모님을 사고로 잃고 할아버지 밑에서 자랐음. 우연한 기회에 특별한 능력을 얻게 됨."
        ),
        Character(
            id="char2",
            name="이조력자",
            description="키 165cm, 보라색 머리, 항상 세련된 옷차림. 둥근 안경을 씀.",
            personality="밝고 활발함. 항상 긍정적으로 생각하며 주인공을 응원함. 재치 있는 농담을 자주 함.",
            backstory="어린 시절부터 뛰어난 두뇌를 가짐. 일찍이 주인공의 능력을 알아보고 파트너가 됨."
        ),
        Character(
            id="char3",
            name="박악당",
            description="키 190cm, 백발, 항상 검은 정장을 입음. 왼쪽 뺨에 큰 흉터가 있음.",
            personality="냉정하고 계산적임. 목표를 위해서라면 수단과 방법을 가리지 않음. 지적이며 전략적임.",
            backstory="명문 가문 출신이지만 가문이 몰락한 후 복수를 다짐함. 비밀 조직의 리더로 세계 지배를 꿈꿈."
        ),
        Character(
            id="char4",
            name="최멘토",
            description="키 175cm, 회색 머리, 수수한 차림. 항상 지팡이를 짚고 다님.",
            personality="지혜롭고 인내심이 많음. 주인공에게 많은 가르침을 줌. 가끔 신비로운 말투를 사용함.",
            backstory="오래전 주인공과 같은 능력을 가졌으나 현재는 능력을 잃음. 주인공을 발견하고 가르치기로 결심함."
        )
    ]
    
    # 캐릭터들을 RAG에 추가
    for character in characters:
        rag.add_character(character, generate_embedding=False)
    
    return rag
