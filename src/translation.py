import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional


class Translation:
    """
    Handles English to German translation using LLM.
    Supports OpenAI models and Academic Cloud models (e.g., Qwen2.5-72B).
    """

    def __init__(
            self,
            model_name: str = "gpt-4o",
            system_prompt: Optional[str] = None,
    ):
        load_dotenv(override=True)

        self.model_name = model_name

        # Option 1: OpenAI (default)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.client = OpenAI(api_key=api_key)

        # Option 2: Academic Cloud (Qwen 2.5 72B)
        # Uncomment to use Qwen via Academic API
        #
        # ACADEMIC_API_KEY = "xxxxxxxx"
        # BASE_URL = "https://chat-ai.academiccloud.de/v1"
        #
        # self.client = OpenAI(
        #     api_key=ACADEMIC_API_KEY,
        #     base_url=BASE_URL
        # )
        # self.model_name = "qwen2.5-72b-instruct"

        self.system_prompt = system_prompt or (
            "You are a professional English-German translator. "
            "Translate each field into fluent, grammatically correct German "
            "while preserving meaning, logic, and tone. "
            "Keep numbers, entities, and structure unchanged."
        )

    def translate(self, text: str) -> str:
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