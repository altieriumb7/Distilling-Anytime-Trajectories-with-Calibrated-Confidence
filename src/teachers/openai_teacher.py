from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

from openai import OpenAI

@dataclass
class OpenAITeacher:
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    store: bool = False  # keep false if you don’t want responses stored

    def __post_init__(self):
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY (env) or api_key argument.")
        self.client = OpenAI(api_key=key)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        # Responses API: max_output_tokens controls output length. :contentReference[oaicite:3]{index=3}
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            store=self.store,
        )

        # Cookbook shows how to read output text. :contentReference[oaicite:4]{index=4}
        text = resp.output[0].content[0].text

        if stop_strings:
            cut = len(text)
            for s in stop_strings:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]

        return text.strip()
