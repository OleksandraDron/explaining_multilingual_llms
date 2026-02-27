import argparse
import os
from pathlib import Path

import pandas as pd
import sacrebleu


class TranslationEvaluation:
    def __init__(
        self,
        data_dir=None,
        datasets=None,
        comet_model_name="Unbabel/wmt22-cometkiwi-da",
    ):
        project_root = Path(__file__).resolve().parents[1]
        self.data_dir = data_dir or (project_root / "data" / "translated_dataset")
        self.datasets = datasets or {
            "qwen_backtranslation": "esnli_selected_translated_qwen_backtranslated.csv",
            "gpt41mini_backtranslation": "esnli_selected_translated_gpt41mini_backtranslated.csv",
            "llama3_backtranslation": "esnli_selected_translated_llama3_backtranslated.csv",
        }
        self.column_pairs = [
            ("Sentence1", "Back_Sentence1"),
            ("Sentence2", "Back_Sentence2"),
        ]
        self.comet_model_name = comet_model_name
        self.comet_model = None

    def load_df(self, dataset_key):
        if dataset_key not in self.datasets:
            raise KeyError(f"Unknown dataset: {dataset_key}")
        file_path = self.data_dir / self.datasets[dataset_key]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        return pd.read_csv(file_path)

    def build_pairs(self, df, source_col, target_col):
        part = df[[source_col, target_col]].dropna()
        source = part[source_col].astype(str).tolist()
        target = part[target_col].astype(str).tolist()
        return source, target

    def evaluate_bleu_chrf(self, df):
        missing_cols = [c for p in self.column_pairs for c in p if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing BLEU/chrF columns: {sorted(set(missing_cols))}")

        results = []
        all_source = []
        all_target = []

        for source_col, target_col in self.column_pairs:
            source, target = self.build_pairs(df, source_col, target_col)
            bleu = sacrebleu.corpus_bleu(target, [source]).score
            chrf = sacrebleu.corpus_chrf(target, [source]).score
            results.append({"text_type": source_col, "bleu": bleu, "chrf": chrf})
            all_source += source
            all_target += target

        bleu_all = sacrebleu.corpus_bleu(all_target, [all_source]).score
        chrf_all = sacrebleu.corpus_chrf(all_target, [all_source]).score
        results.append({"text_type": "ALL combined", "bleu": bleu_all, "chrf": chrf_all})
        return pd.DataFrame(results)

    def _get_comet_model(self):
        if self.comet_model is not None:
            return self.comet_model
        os.environ["HF_HUB_USE_SYMLINKS_DEFAULT"] = "0"
        from comet import download_model, load_from_checkpoint  # type: ignore

        ckpt = download_model(self.comet_model_name)
        self.comet_model = load_from_checkpoint(ckpt)
        return self.comet_model

    def parse_comet_output(self, output):
        if isinstance(output, dict):
            system_score = output.get("system_score")
            seg_scores = output.get("segments_scores") or output.get("scores") or []
            return system_score, seg_scores
        if isinstance(output, list):
            seg_scores = output
            system_score = sum(seg_scores) / len(seg_scores) if seg_scores else None
            return system_score, seg_scores
        return None, []

    def evaluate_comet_qe(self, df):
        parts = []
        if {"Sentence1", "Translated_Sentence1"}.issubset(df.columns):
            parts.append(pd.DataFrame({"field": "Sentence1", "src": df["Sentence1"], "mt": df["Translated_Sentence1"]}))
        if {"Sentence2", "Translated_Sentence2"}.issubset(df.columns):
            parts.append(pd.DataFrame({"field": "Sentence2", "src": df["Sentence2"], "mt": df["Translated_Sentence2"]}))

        if not parts:
            raise ValueError("Missing COMET columns. Need Sentence{1,2} and Translated_Sentence{1,2}.")

        df_long = pd.concat(parts, ignore_index=True).fillna("")
        data = [{"src": s, "mt": m} for s, m in zip(df_long["src"], df_long["mt"])]

        model = self._get_comet_model()
        output = model.predict(data, batch_size=32, gpus=0, num_workers=1)
        system_score, seg_scores = self.parse_comet_output(output)

        df_long = df_long.copy()
        df_long["comet_qe"] = seg_scores

        rows = []
        if system_score is not None:
            rows.append({"field": "ALL combined", "comet_qe": system_score})
        for name, group in df_long.groupby("field"):
            rows.append({"field": name, "comet_qe": group["comet_qe"].mean()})
        return pd.DataFrame(rows)

    def evaluate_one(self, dataset_key, include_comet=False):
        df = self.load_df(dataset_key)
        result = {"bleu_chrf": self.evaluate_bleu_chrf(df)}
        if include_comet:
            result["comet_qe"] = self.evaluate_comet_qe(df)
        return result

    def evaluate_all(self, include_comet=False):
        all_results = {}
        for dataset_key in self.datasets:
            try:
                all_results[dataset_key] = self.evaluate_one(dataset_key, include_comet=include_comet)
            except FileNotFoundError as e:
                print(f"[Skip] {e}")
                continue
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Translation evaluation with BLEU/chrF and optional COMET-QE.")
    parser.add_argument("--dataset", type=str, default=None, help="Example: qwen_backtranslation")
    parser.add_argument("--with-comet", action="store_true", help="Enable COMET-QE scoring.")
    args = parser.parse_args()

    evaluator = TranslationEvaluation()
    if args.dataset:
        results = {args.dataset: evaluator.evaluate_one(args.dataset, include_comet=args.with_comet)}
    else:
        results = evaluator.evaluate_all(include_comet=args.with_comet)

    for dataset_key, outputs in results.items():
        print(f"\n===== {dataset_key} =====")
        print("\n[BLEU/chrF]")
        print(outputs["bleu_chrf"].to_string(index=False))
        if "comet_qe" in outputs:
            print("\n[COMET-QE]")
            print(outputs["comet_qe"].to_string(index=False))


if __name__ == "__main__":
    main()
