"""
Run the gptmed agentic API locally.

IMPORTANT: always use the project virtualenv, not system Python.

  # From the repo root:
  source env/bin/activate
  python -m gptmed.api.run

  # Or directly with the venv interpreter (no activation needed):
  ./env/bin/python -m gptmed.api.run

  # If fastapi / uvicorn are missing:
  pip install "fastapi>=0.100.0" "uvicorn[standard]>=0.22.0"

Once running, open:
    http://localhost:8000          - root / health check
    http://localhost:8000/docs     - Swagger UI
    http://localhost:8000/redoc    - ReDoc
"""

import sys

try:
    import uvicorn
except ModuleNotFoundError:
    print(
        "\n[ERROR] uvicorn is not installed in the active Python environment.\n"
        f"        Active interpreter: {sys.executable}\n\n"
        "        Install it with:\n"
        "            pip install 'fastapi>=0.100.0' 'uvicorn[standard]>=0.22.0'\n\n"
        "        Make sure you are using the project virtualenv, e.g.:\n"
        "            source env/bin/activate          # then re-run\n"
        "            ./env/bin/python -m gptmed.api.run   # or use venv Python directly\n"
    )
    sys.exit(1)

if __name__ == "__main__":
    uvicorn.run(
        "gptmed.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["gptmed"],
    )
