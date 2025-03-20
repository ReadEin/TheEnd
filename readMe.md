
##### openAI API 와 LangChain/LangGraph 에 대한 예시 프로젝트입니다.

##### 파이썬 버전
3.11.x

##### vscode 에서 개발. 파이썬 플러그인 외, 필요한 플러그인 목록
```
Batch Runner
Black Formatter
autoDocstring
isort
```

##### 프로젝트 구조

각 디렉토리와 파일의 역할:

- `src/`: 소스 코드 디렉토리
- `tests/`: 테스트 코드 디렉토리
- `requirements.txt`: 프로젝트에 필요한 파이썬 패키지 목록
- `.gitignore`: Git 버전 관리에서 제외할 파일 목록
- `README.md`: 프로젝트 설명 문서
- `.private` : api access key 또는 환경 변수 설정
- `.bat` : 배치 파일 설정(Batch Runner 플러그인 활용)

##### .bat
```Batch Runnere 플러그인 설치 시, bat 확장자 파일에서 실행 버튼이 생김.```
###### project_initialize.bat
```프로젝트 venv 세팅, 빌드```

##### .vscode
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "TEST_MODE": "true"
            }
        },
        {
            "name": "Python Debugger: All Test File",
            "type": "debugpy",
            "request": "launch",
            "module": "unittest",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "TEST_MODE": "true"
            }
        }
    ]
}
```