```bash
uv venv --python 3.9 py39
source py39/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```