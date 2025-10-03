import os
import requests
from typing import List, Dict, Any
from .base import Provider

class DeepSeekProvider(Provider):
    def __init__(self, model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")):
        super().__init__("deepseek")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        assert self.api_key, "DEEPSEEK_API_KEY required"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.0),
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "usage": data.get("usage", {}), "raw": data}
