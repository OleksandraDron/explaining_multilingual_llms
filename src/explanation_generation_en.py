#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_explanations_en.py 
Batch-generate English explanations for e-SNLI subset using DeepSeek API.

Requirements:
  pip install -U openai pandas tqdm openpyxl

"""

import os
import time
import random
from typing import Dict, Any

import pandas as pd
from tqdm import tqdm
from openai import OpenAI 

# --- Config ---
INPUT_PATH = "data/esnli_selected.xlsx"
SHEET_NAME = "English"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "esnli_generated_deepseek_en.xlsx")

MODEL_NAME = "deepseek-chat"  
BASE_URL = "https://api.deepseek.com"  
MAX_TOKENS = 80
TEMPERATURE = 0.7
TOP_P = 0.9
CHECKPOINT_EVERY = 100
MAX_RETRIES = 5
RESUME_FROM_EXISTING = True

def setup_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing env var DEEPSEEK_API_KEY. Set it via: export DEEPSEEK_API_KEY=sk-xxxx")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return client

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_dataframe() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    expected = {"idx", "gold_label", "Sentence1", "Sentence2", "Explanation_1"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sheet '{SHEET_NAME}': {missing}")
    for col in ["Generated_Explanation_EN", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"]:
        if col not in df.columns:
            df[col] = None
    return df

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

def call_deepseek(client: OpenAI, messages, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """OpenAI v1-style /chat/completions call with retry."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stream=False,
            )
            return resp
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[Warn] API error on attempt {attempt + 1}/{max_retries}: {e} -> retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Failed after maximum retries.")

def save_checkpoint(df: pd.DataFrame, upto_index: int):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_{upto_index:04d}.xlsx")
    df.to_excel(ckpt_path, index=False)
    print(f"[Checkpoint] Saved: {ckpt_path}")

def main():
    client = setup_client()
    ensure_dirs()
    df = load_dataframe()

    if RESUME_FROM_EXISTING and os.path.exists(FINAL_OUTPUT):
        print(f"[Resume] Loading existing output: {FINAL_OUTPUT}")
        df_existing = pd.read_excel(FINAL_OUTPUT)
        if "idx" in df.columns and "idx" in df_existing.columns:
            cols = ["Generated_Explanation_EN", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"]
            for c in cols:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)
            df = df.merge(df_existing[["idx"] + cols], on="idx", how="left")
        else:
            for col in ["Generated_Explanation_EN", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"]:
                if col not in df.columns:
                    df[col] = None
            for col in ["Generated_Explanation_EN", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"]:
                if col in df_existing.columns:
                    df.loc[df[col].isna(), col] = df_existing.loc[df[col].isna(), col]

    total_rows = len(df)
    print(f"[Info] Total rows: {total_rows}")

    pbar = tqdm(range(total_rows), desc="Generating EN explanations")

    for i in pbar:
        if pd.notna(df.at[i, "Generated_Explanation_EN"]) and str(df.at[i, "Generated_Explanation_EN"]).strip():
            continue

        premise = str(df.at[i, "Sentence1"])
        hypothesis = str(df.at[i, "Sentence2"])
        label = str(df.at[i, "gold_label"])

        prompt = build_prompt(premise, hypothesis, label)
        messages = [
            {"role": "system", "content": "You are an expert in natural language inference explanations."},
            {"role": "user", "content": prompt},
        ]

        try:
            resp = call_deepseek(client, messages)
            content = resp.choices[0].message.content.strip() if resp.choices else ""

            usage = getattr(resp, "usage", None)
            prompt_toks = getattr(usage, "prompt_tokens", None) if usage else None
            completion_toks = getattr(usage, "completion_tokens", None) if usage else None
            total_toks = getattr(usage, "total_tokens", None) if usage else None

            df.at[i, "Generated_Explanation_EN"] = content
            df.at[i, "Prompt_Tokens"] = prompt_toks
            df.at[i, "Completion_Tokens"] = completion_toks
            df.at[i, "Total_Tokens"] = total_toks

        except Exception as e:
            df.at[i, "Generated_Explanation_EN"] = f"[Error: {e}]"

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(df, i + 1)

    df.to_excel(FINAL_OUTPUT, index=False)
    print(f"[Done] Saved final output to: {FINAL_OUTPUT}")

    # Rough cost estimate if usage available (upper bound assumes input cache-miss)
    if "Total_Tokens" in df.columns and df["Total_Tokens"].notna().any():
        out_tokens = pd.to_numeric(df["Completion_Tokens"], errors="coerce").fillna(0).sum()
        in_tokens = pd.to_numeric(df["Prompt_Tokens"], errors="coerce").fillna(0).sum()
        cost_in = (in_tokens / 1_000_000.0) * 0.28
        cost_out = (out_tokens / 1_000_000.0) * 0.42
        print(f"[Usage] Prompt tokens: {int(in_tokens):,} | Completion tokens: {int(out_tokens):,}")
        print(f"[Cost ] Est. input ${cost_in:.4f} + output ${cost_out:.4f} = total ${cost_in + cost_out:.4f} (upper bound)")
    else:
        print("[Usage] Token usage not returned by API; cost estimate skipped.")

if __name__ == "__main__":
    main()
