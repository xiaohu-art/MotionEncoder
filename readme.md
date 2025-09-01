```bash
uv venv --python 3.10 .venv-py310
source .venv-py310/bin/activate

uv pip install -r requirements.txt

uv run train.py
uv run infer.py
```