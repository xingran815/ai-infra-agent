import json
from pathlib import Path

current_fixture = None


def load_scenario(path: str):
    global current_fixture 
    current_fixture= json.loads(
        Path(path).read_text()
    )

def load_fixture():
    if current_fixture is None:
        raise RuntimeError("Load a scenario before calling tools.")

    return current_fixture
