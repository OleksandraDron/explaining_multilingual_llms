import os
from openai import OpenAI
from dotenv import load_dotenv


class ExplanationGenerationDE:
    """
    Generates German explanations for NLI tasks using OpenAI models.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.3,
        max_tokens: int = 150,
    ):
        load_dotenv(override=True)

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prompt
    def build_prompt(
        self,
        premise: str,
        hypothesis: str,
        label: str
    ) -> str:
        return (
            "Gegeben ist eine Aufgabe zur natürlichen Sprachinferenz (NLI).\n"
            "Die Beziehung zwischen den beiden Sätzen ist bereits festgelegt.\n\n"
            f'Premise: "{premise}"\n'
            f'Hypothesis: "{hypothesis}"\n'
            f"Label: {label}\n\n"
            "Schreibe eine sehr kurze Erklärung auf Deutsch (1–2 kurze Sätze), "
            "die genau die konkreten Fakten nennt, die zu dieser Beziehung passen.\n\n"
            "Wichtige Vorgaben:\n"
            "- Die Erklärung MUSS inhaltlich zum gegebenen Label passen.\n"
            "- Erkläre die Situation, nicht das Label selbst.\n"
            "- Erwähne weder Premise, Hypothese, Label noch Begriffe wie "
            "\"Folgerung\", \"Widerspruch\" oder \"neutral\".\n"
            "- Verwende konkrete Tatsachenbehauptungen.\n"
            "- Keine Bewertung, ob das Label richtig oder falsch ist."
        )

    # Single Generation
    def generate(
        self,
        premise: str,
        hypothesis: str,
        label: str
    ) -> str:
        prompt = self.build_prompt(premise, hypothesis, label)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein linguistisch präziser Assistent für "
                        "Natural Language Inference. Die Erklärung muss "
                        "kurz, sachlich und logisch korrekt sein."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content.strip()