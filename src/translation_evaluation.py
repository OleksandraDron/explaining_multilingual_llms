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
        self.data_dir = data_dir or (project_root / "data" / "translated_and_generated_dataset")
        self.datasets = datasets or {
            "qwen_backtranslation": "esnli_selected_translated_qwen_backtranslated.csv",
            # "gpt41mini_backtranslation": "esnli_selected_translated_gpt41mini_backtranslated.csv",
            # "llama3_backtranslation": "esnli_selected_translated_llama3_backtranslated.csv",
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

class HumanEvalICC:
    def __init__(self, data_dir=None, filename="esnli_translation_de_human_eval.xlsx"):
        project_root = Path(__file__).resolve().parents[1]
        self.data_dir = data_dir or (project_root / "data" / "translated_and_generated_dataset")
        self.file_path = self.data_dir / filename

    def load_df(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Human eval file not found: {self.file_path}")
        return pd.read_excel(self.file_path)

    def compute_icc_2_1(self, df, rater_cols):
        part = df[list(rater_cols)].apply(pd.to_numeric, errors="coerce").dropna()
        if part.empty:
            return {"icc_2_1": None, "n_items": 0, "n_raters": len(rater_cols)}

        ratings = part.to_numpy(dtype=float)
        n, k = ratings.shape

        grand_mean = ratings.mean()
        row_means = ratings.mean(axis=1)
        col_means = ratings.mean(axis=0)

        ss_rows = k * ((row_means - grand_mean) ** 2).sum()
        ss_cols = n * ((col_means - grand_mean) ** 2).sum()
        ss_total = ((ratings - grand_mean) ** 2).sum()
        ss_error = ss_total - ss_rows - ss_cols

        ms_rows = ss_rows / (n - 1) if n > 1 else 0.0
        ms_cols = ss_cols / (k - 1) if k > 1 else 0.0
        ms_error = ss_error / ((n - 1) * (k - 1)) if n > 1 and k > 1 else 0.0

        denominator = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
        icc = (ms_rows - ms_error) / denominator if denominator != 0 else None

        return {"icc_2_1": icc, "n_items": n, "n_raters": k}

    def evaluate(self):
        df = self.load_df()
        sentence1 = self.compute_icc_2_1(df, ("Sen1_Human1", "Sen1_Human2"))
        sentence2 = self.compute_icc_2_1(df, ("Sen2_Human1", "Sen2_Human2"))

        return pd.DataFrame(
            [
                {
                    "field": "Sentence1",
                    "rater_cols": "Sen1_Human1,Sen1_Human2",
                    **sentence1,
                },
                {
                    "field": "Sentence2",
                    "rater_cols": "Sen2_Human1,Sen2_Human2",
                    **sentence2,
                },
            ]
        )
