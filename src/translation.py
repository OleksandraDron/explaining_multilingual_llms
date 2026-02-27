import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


class Translation:
    """
    Handles English to German translation using LLM.
    Supports OpenAI models and Academic Cloud models (e.g., Qwen2.5-72B).
    """

    def __init__(
        self,
        model_name="gpt-4.1-mini",
        system_prompt: Optional[str] = None,
        base_url=None,
        api_key=None,
    ):
        load_dotenv(override=True)

        self.model_name = model_name
        if base_url:
            api_key = api_key or os.getenv("ACADEMIC_API_KEY") or os.getenv("QWEN_API_KEY")
            if not api_key:
                raise ValueError("ACADEMIC_API_KEY (or QWEN_API_KEY) not found in environment variables.")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            self.client = OpenAI(api_key=api_key)

        self.system_prompt = system_prompt or (
            "You are a professional English-German translator. "
            "Translate each field into fluent, grammatically correct German "
            "while preserving meaning, logic, and tone. "
            "Keep numbers, entities, and structure unchanged."
        )

    def translate(self, text):
        """
        Translate a text.
        """
        if not text or pd.isna(text):
            return ""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
        )

        return response.choices[0].message.content.strip()
