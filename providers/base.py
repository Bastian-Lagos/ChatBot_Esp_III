from __future__ import annotations
from typing import Any, Dict, List

class Provider:
    name: str

    def __init__(self, name: str):
        self.name = name

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
        returns: {"text": "...", "usage": {...}, "raw": {...}}
        """
        raise NotImplementedError
