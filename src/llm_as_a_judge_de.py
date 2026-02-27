import json
import os
import argparse

import pandas as pd


SYSTEM_PROMPT = """Du bist ein Assistent, der Erklärungen für Prämisse-Hypothese-Paare bewertet.
Deine Aufgabe ist es, jede Erklärung unabhängig und objektiv anhand von fünf fest definierten Kriterien zu bewerten.
Die Bewertung erfolgt binär: 0 = nein, 1 = ja.
"""

USER_TEMPLATE = """Gegeben sind die folgenden Felder:
Gold-Label: "{gold_label}"
Prämisse: "{premise}"
Hypothese: "{hypothesis}"
Erklärung: "{explanation}"

Das Gold-Label gibt die korrekte Beziehung zwischen Prämisse und Hypothese an und dient ausschließlich als Kontext für die Bewertung der Erklärung.

Bewerte die Erklärung anhand der folgenden Kriterien:

1. SprachlicheQualität
– Die Erklärung ist klar, verständlich und sprachlich korrekt formuliert.

2. Relevanz
– Die Erklärung bezieht sich direkt auf den logischen Zusammenhang zwischen Prämisse und Hypothese.

3. FaktischeKorrektheit
– Die Erklärung basiert ausschließlich auf den gegebenen Sätzen oder auf allgemein gültigen Definitionen.

4. NeueInformation
– Die Erklärung führt zusätzliche Informationen oder Schlussfolgerungen ein, die nicht explizit aus den gegebenen Sätzen oder Definitionen folgen.

5. UnnötigeInformation
– Die Erklärung enthält Details, die für die Begründung der Beziehung zwischen Prämisse und Hypothese irrelevant sind.

Die folgenden Beispiele zeigen typische Bewertungssituationen und dienen als Referenz für die Anwendung der Kriterien:

{Beispiele}

Nun bewerte die obige Erklärung anhand der definierten Kriterien.

Gib ausschließlich ein JSON-Objekt im folgenden Format zurück, ohne zusätzlichen Text oder Begründung:

{{
  "SprachlicheQualität": 0|1,
  "Relevanz": 0|1,
  "FaktischeKorrektheit": 0|1,
  "NeueInformation": 0|1,
  "UnnötigeInformation": 0|1
}}
"""


def render_examples(examples):
    """Format few-shot examples into one text block for the prompt."""
    blocks = []
    for i, ex in enumerate(examples, start=1):
        block = (
            f"Beispiel {i}:\n"
            f'Prämisse: "{ex["Prämisse"]}"\n'
            f'Hypothese: "{ex["Hypothese"]}"\n'
            f'Erklärung: "{ex["Erklärung"]}"\n'
            f'Bewertung: {json.dumps(ex["Bewertung"], ensure_ascii=False)}\n'
        )
        blocks.append(block)
    return "\n".join(blocks).strip()


def load_examples(path):
    """Load few-shot JSON file and return formatted example text."""
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    return render_examples(examples)


def extract_json_only(text):
    """Extract the first JSON object from model output text."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Kein JSON gefunden. Output:\n{text}")
    return json.loads(text[start : end + 1])


class BaseJudgeDE:
    def __init__(
        self,
        fewshot_json,
        col_gold="gold_label",
        col_premise="Sentence1_de",
        col_hyp="Sentence2_de",
        col_expl="Explanation_de_generated",
    ):
        self.col_gold = col_gold
        self.col_premise = col_premise
        self.col_hyp = col_hyp
        self.col_expl = col_expl

        self.out_cols = {
            "SprachlicheQualität": "Well-Written",
            "Relevanz": "Related",
            "FaktischeKorrektheit": "Factual",
            "NeueInformation": "New Information",
            "UnnötigeInformation": "Unnecessary Information",
        }

        self.beispiele_text = load_examples(fewshot_json)

    def build_user_prompt(self, row):
        """Build one user prompt from a dataframe row."""
        return USER_TEMPLATE.format(
            gold_label=str(row[self.col_gold]),
            premise=str(row[self.col_premise]),
            hypothesis=str(row[self.col_hyp]),
            explanation=str(row[self.col_expl]),
            Beispiele=self.beispiele_text,
        )

    def judge_one(self, user_prompt, max_new_tokens=180):
        """Subclass must return (scores_dict, raw_output_text)."""
        raise NotImplementedError

    def run_excel(self, excel_in, excel_out, save_every=5, max_new_tokens=180):
        """Run judging for all rows and save progress to Excel."""
        df = pd.read_excel(excel_in)

        for out_col in self.out_cols.values():
            if out_col not in df.columns:
                df[out_col] = pd.NA
        if "judge_error" not in df.columns:
            df["judge_error"] = pd.NA
        if "judge_raw_output" not in df.columns:
            df["judge_raw_output"] = pd.NA

        for i, row in df.iterrows():
            user_prompt = self.build_user_prompt(row)
            try:
                scores, raw = self.judge_one(user_prompt, max_new_tokens=max_new_tokens)
                for de_key, out_col in self.out_cols.items():
                    if de_key not in scores:
                        raise ValueError(f"Key fehlt: {de_key}. JSON: {scores}")
                    df.at[i, out_col] = int(scores[de_key])
                df.at[i, "judge_error"] = pd.NA
                df.at[i, "judge_raw_output"] = raw
            except Exception as e:
                df.at[i, "judge_error"] = str(e)

            if (i + 1) % save_every == 0:
                df.to_excel(excel_out, index=False)
                print(f"Saved progress at row {i + 1}")

        df.to_excel(excel_out, index=False)
        print("Done:", excel_out)
        return df


class HuggingFaceJudgeDE(BaseJudgeDE):
    def __init__(self, fewshot_json, model_id="prometheus-eval/prometheus-7b-v2.0", **kwargs):
        super().__init__(fewshot_json, **kwargs)
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load tokenizer and model once, then reuse for all rows."""
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

    def judge_one(self, user_prompt, max_new_tokens=180):
        """Generate one judgment with the local Prometheus model."""
        import torch

        self.load_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )

        gen = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        return extract_json_only(gen), gen


class QwenApiJudgeDE(BaseJudgeDE):
    def __init__(
        self,
        fewshot_json,
        api_key=None,
        base_url="https://chat-ai.academiccloud.de/v1",
        model="qwen2.5-72b-instruct",
        temperature=0.0,
        max_tokens=220,
        **kwargs,
    ):
        super().__init__(fewshot_json, **kwargs)
        from openai import OpenAI

        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("QWEN_API_KEY fehlt. Bitte in .env setzen oder beim Init übergeben.")

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def judge_one(self, user_prompt, max_new_tokens=180):
        """Generate one judgment by calling the Qwen API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, max_new_tokens + 40),
        )

        raw = (response.choices[0].message.content or "").strip()
        return extract_json_only(raw), raw


def run_hf_judge(
    input_excel="data/esnli_de_generated_ger_explanations_gpt41mini.xlsx",
    output_excel="data/results_llm-as-a-judge/result_de_prometheus.xlsx",
    fewshot_json="data/few-shot_examples_de.json",
    model_id="prometheus-eval/prometheus-7b-v2.0",
    save_every=5,
    max_new_tokens=180,
):
    judge = HuggingFaceJudgeDE(
        fewshot_json=fewshot_json,
        model_id=model_id,
    )
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    return judge.run_excel(
        excel_in=input_excel,
        excel_out=output_excel,
        save_every=save_every,
        max_new_tokens=max_new_tokens,
    )


def run_api_judge(
    input_excel="data/esnli_de_generated_ger_explanations_gpt41mini.xlsx",
    output_excel="data/results_llm-as-a-judge/result_de_qwen.xlsx",
    fewshot_json="data/few-shot_examples_de.json",
    api_key=None,
    base_url="https://chat-ai.academiccloud.de/v1",
    model="qwen2.5-72b-instruct",
    save_every=5,
    max_new_tokens=180,
):
    judge = QwenApiJudgeDE(
        fewshot_json=fewshot_json,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    return judge.run_excel(
        excel_in=input_excel,
        excel_out=output_excel,
        save_every=save_every,
        max_new_tokens=max_new_tokens,
    )


def main():
    parser = argparse.ArgumentParser(description="Run German LLM-as-a-judge.")
    parser.add_argument("--backend", choices=["hf", "api"], default="hf")
    parser.add_argument("--input", default="data/esnli_de_generated_ger_explanations_gpt41mini.xlsx")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fewshot", default="data/few-shot_examples_de.json")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--model", default=None, help="HF model_id or API model name")
    parser.add_argument("--base-url", default="https://chat-ai.academiccloud.de/v1")
    args = parser.parse_args()

    if args.backend == "hf":
        output = args.output or "data/results_llm-as-a-judge/result_de_prometheus.xlsx"
        model_id = args.model or "prometheus-eval/prometheus-7b-v2.0"
        run_hf_judge(
            input_excel=args.input,
            output_excel=output,
            fewshot_json=args.fewshot,
            model_id=model_id,
            save_every=args.save_every,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        output = args.output or "data/results_llm-as-a-judge/result_de_qwen.xlsx"
        model = args.model or "qwen2.5-72b-instruct"
        run_api_judge(
            input_excel=args.input,
            output_excel=output,
            fewshot_json=args.fewshot,
            base_url=args.base_url,
            model=model,
            save_every=args.save_every,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
