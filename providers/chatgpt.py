import os
import openai
from typing import List, Dict, Any
from .base import Provider
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatGPTProvider(Provider):
    def __init__(self, model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")):
        super().__init__("chatgpt")
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.0),
        )
        text = resp.choices[0].message.content
        return {"text": text, "usage": resp.usage if hasattr(resp, "usage") else {}, "raw": resp}
