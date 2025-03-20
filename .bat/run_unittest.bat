chcp 65001
echo "run all unittest"
cd ..
set PYTHONPATH=%CD%
python -m unittest discover -s tests
