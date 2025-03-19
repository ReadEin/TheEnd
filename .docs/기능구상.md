### 프롬프트 기법
#### Chain of thougth
##### 플롯 구성, 소재 탐색
#### Persona
##### 가정법으로 이야기 전개, 인물 인터뷰
#### Json Formatter(jprompt)
##### 대화 내용에 대해 특정 정보들을 추출하고 Json Format을 채운다.

### 프롬프트와 function을 활용하여 어플리케이션에 적용
#### 파이프를 구상하여야 한다.
#### 프롬프트 : prompt, jprompt
#### function : action, parser

### 파이프
#### prompt -> jprompt -> parser -> action
#### 예시 : 
```prompt
당신은 심리학자 입니다.
아래 정보를 바탕으로 인물의 추측되는 심리적 특성과 인물의 삶의 배경을 추측하세요.
{man_history}
```

```jprompt
당신은 아래 주어진 문장을 요약하여 하나의 json 형식으로 답변하세요.
json 답변의 key 는 오직 {json_key_list} 입니다.
```

parser.parse(jprompt).get_action().call();