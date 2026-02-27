#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


@dataclass
class ExplanationGenerationENConfig:
    input_path: str = "data/esnli_selected.xlsx"
    sheet_name: str = "English"
    output_dir: str = "outputs"
    final_output: str = "esnli_generated_deepseek_en.xlsx"
    model_name: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 80
    temperature: float = 0.7
    top_p: float = 0.9
    checkpoint_every: int = 100
    max_retries: int = 5
    resume_from_existing: bool = True
    api_key_env: str = "DEEPSEEK_API_KEY"


class ExplanationGenerationEN:
    """
    Batch-generate English explanations for e-SNLI subset using DeepSeek API (OpenAI-compatible).
    """

    def __init__(self, config: ExplanationGenerationENConfig):
        self.cfg = config
        self.checkpoint_dir = os.path.join(self.cfg.output_dir, "checkpoints")
        self.client: Optional[OpenAI] = None
        self.df: Optional[pd.DataFrame] = None

    # ---------- Setup ----------
    def setup_client(self) -> OpenAI:
        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            raise SystemExit(
                f"Missing env var {self.cfg.api_key_env}. Set it via: export {self.cfg.api_key_env}=sk-xxxx"
            )
        self.client = OpenAI(api_key=api_key, base_url=self.cfg.base_url)
        return self.client

    def ensure_dirs(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ---------- Data ----------
    def load_dataframe(self) -> pd.DataFrame:
        if not os.path.exists(self.cfg.input_path):
            raise FileNotFoundError(f"Input file not found: {self.cfg.input_path}")

        df = pd.read_excel(self.cfg.input_path, sheet_name=self.cfg.sheet_name)

        expected = {"idx", "gold_label", "Sentence1", "Sentence2", "Explanation_1"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in sheet '{self.cfg.sheet_name}': {missing}")

        for col in ["Generated_Explanation_EN", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"]:
            if col not in df.columns:
                df[col] = None

        self.df = df
        return df

    def maybe_resume(self) -> None:
        """
        If resume enabled and final output exists, merge the generated columns back into current df via idx.
        """
        assert self.df is not None, "Dataframe not loaded."

        if not (self.cfg.resume_from_existing and os.path.exists(self.cfg.final_output)):
            return

        print(f"[Resume] Loading existing output: {self.cfg.final_output}")
        df_existing = pd.read_excel(self.cfg.final_output)

        cols = ["Generated_Explanation_EN", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"]
        if "idx" in self.df.columns and "idx" in df_existing.columns:
            # Remove possibly stale columns then merge
            for c in cols:
                if c in self.df.columns:
                    self.df.drop(columns=[c], inplace=True)

            keep_cols = ["idx"] + [c for c in cols if c in df_existing.columns]
            self.df = self.df.merge(df_existing[keep_cols], on="idx", how="left")
        else:
            # Fallback: align row-wise if idx missing (less safe)
            for c in cols:
                if c not in self.df.columns:
                    self.df[c] = None
            for c in cols:
                if c in df_existing.columns:
                    mask = self.df[c].isna()
                    self.df.loc[mask, c] = df_existing.loc[mask, c]

    # ---------- Prompting / API ----------
    @staticmethod
    def build_prompt(premise: str, hypothesis: str, label: str) -> str:
        return (
            "Given the following natural language inference task:\n"
            f'Premise: "{premise}"\n'
            f'Hypothesis: "{hypothesis}"\n'
            f"Label: {label}.\n\n"
            "Write a short (1–3 sentences), clear, and logically consistent English explanation "
            "that justifies why the label is correct. "
            "Avoid introducing new facts not supported by the premise."
        )

    def call_api(self, messages: List[Dict[str, str]]) -> Any:
        """
        OpenAI v1-style /chat/completions call with retry.
        Returns the raw response object.
        """
        assert self.client is not None, "Client not initialized."

        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=messages,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    stream=False,
                )
                return resp
            except Exception as e:
                wait = (2**attempt) + random.uniform(0, 0.5)
                print(f"[Warn] API error on attempt {attempt + 1}/{self.cfg.max_retries}: {e} -> retry in {wait:.1f}s")
                time.sleep(wait)

        raise RuntimeError("Failed after maximum retries.")

    # ---------- Persistence ----------
    def save_checkpoint(self, upto_index: int) -> None:
        assert self.df is not None, "Dataframe not loaded."
        ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_{upto_index:04d}.xlsx")
        self.df.to_excel(ckpt_path, index=False)
        print(f"[Checkpoint] Saved: {ckpt_path}")

    def save_final(self) -> None:
        assert self.df is not None, "Dataframe not loaded."
        self.df.to_excel(self.cfg.final_output, index=False)
        print(f"[Done] Saved final output to: {self.cfg.final_output}")

    

    # ---------- Main run ----------
    def run(self) -> None:
        self.ensure_dirs()
        self.setup_client()
        self.load_dataframe()
        self.maybe_resume()

        assert self.df is not None
        total_rows = len(self.df)
        print(f"[Info] Total rows: {total_rows}")

        pbar = tqdm(range(total_rows), desc="Generating EN explanations")

        for i in pbar:
            existing = self.df.at[i, "Generated_Explanation_EN"]
            if pd.notna(existing) and str(existing).strip():
                continue

            premise = str(self.df.at[i, "Sentence1"])
            hypothesis = str(self.df.at[i, "Sentence2"])
            label = str(self.df.at[i, "gold_label"])

            prompt = self.build_prompt(premise, hypothesis, label)
            messages = [
                {"role": "system", "content": "You are an expert in natural language inference explanations."},
                {"role": "user", "content": prompt},
            ]

            try:
                resp = self.call_api(messages)

                content = resp.choices[0].message.content.strip() if getattr(resp, "choices", None) else ""
                usage = getattr(resp, "usage", None)
                prompt_toks = getattr(usage, "prompt_tokens", None) if usage else None
                completion_toks = getattr(usage, "completion_tokens", None) if usage else None
                total_toks = getattr(usage, "total_tokens", None) if usage else None

                self.df.at[i, "Generated_Explanation_EN"] = content
                self.df.at[i, "Prompt_Tokens"] = prompt_toks
                self.df.at[i, "Completion_Tokens"] = completion_toks
                self.df.at[i, "Total_Tokens"] = total_toks

            except Exception as e:
                self.df.at[i, "Generated_Explanation_EN"] = f"[Error: {e}]"

            if self.cfg.checkpoint_every > 0 and (i + 1) % self.cfg.checkpoint_every == 0:
                self.save_checkpoint(i + 1)

        self.save_final()
        self.print_cost_estimate()