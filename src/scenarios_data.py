import json
from pathlib import Path

def load_scenario(path: str):
    fixture = json.loads(
        Path(path).read_text()
    )

    return fixture
