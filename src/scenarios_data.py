import json
from pathlib import Path

current_fixture = None


def load_scenario(path: str) -> bool:
    global current_fixture 
    try:
        current_fixture= json.loads(
            Path(path).read_text()
        )
        return True
    except Exception as e:
        print(f"Error loading scenario from {path}: {e}")
        return False

def load_fixture():
    if current_fixture is None:
        raise RuntimeError("Load a scenario before calling tools.")

    return current_fixture
