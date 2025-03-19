chcp 65001

cd..
if not exist venv (
    python -m venv venv
)
call venv/Scripts/deactivate
call venv/Scripts/activate
pip install -r requirements.txt