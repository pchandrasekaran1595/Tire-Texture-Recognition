title Prepare Virtual Environment

py -3.9 -m pip install virtualenv

py -3.9 -m virtualenv venv

start /MAX cmd /c "title Install Dependencies && cls && cd venv/Scripts && activate && cd .. && cd .. && pip install -r requirements.txt && timeout /t 5 /nobreak"

