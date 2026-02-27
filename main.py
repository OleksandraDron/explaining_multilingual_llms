import argparse

import pandas as pd
from src.explanation_generation_de import ExplanationGenerationDE
from src.imbalanced_agreement import ImbalancedAgreementCalculator
from src.llm_as_a_judge_de import run_api_judge, run_hf_judge
from src.translation import Translation
from src.translation_evaluation import HumanEvalICC, TranslationEvaluation


BASE_URL_ACADEMIC = "https://chat-ai.academiccloud.de/v1"
MODEL_CONFIGS = [
    ("gpt", "gpt-4.1-mini", None, "gpt41mini"),
    ("qwen", "qwen2.5-72b-instruct", BASE_URL_ACADEMIC, "qwen25_72b"),
    ("llama", "llama-3.3-70b-instruct", BASE_URL_ACADEMIC, "llama33_70b"),
]


def run_translation(args):
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    model_keys = args.models.split(",") if args.models != "all" else ["gpt", "qwen", "llama"]
    config_map = {k: (m, b, t) for k, m, b, t in MODEL_CONFIGS}

    for key in model_keys:
        key = key.strip()
        if key not in config_map:
            print(f"[Skip] unknown model key: {key}")
            continue

        model_name, base_url, output_tag = config_map[key]
        out_path = f"{args.output_dir}/esnli_selected_translated_{output_tag}.xlsx"
        print(f"\n[Translation] {model_name}")

        translator = Translation(model_name=model_name, base_url=base_url)
        out_df = df.copy()
        for col in ["Translated_Sentence1", "Translated_Sentence2", "Translated_Explanation_1"]:
            if col not in out_df.columns:
                out_df[col] = pd.NA

        for i in range(len(out_df)):
            if pd.isna(out_df.at[i, "Translated_Sentence1"]) or not str(out_df.at[i, "Translated_Sentence1"]).strip():
                out_df.at[i, "Translated_Sentence1"] = translator.translate(out_df.at[i, "Sentence1"])
            if pd.isna(out_df.at[i, "Translated_Sentence2"]) or not str(out_df.at[i, "Translated_Sentence2"]).strip():
                out_df.at[i, "Translated_Sentence2"] = translator.translate(out_df.at[i, "Sentence2"])
            if pd.isna(out_df.at[i, "Translated_Explanation_1"]) or not str(out_df.at[i, "Translated_Explanation_1"]).strip():
                out_df.at[i, "Translated_Explanation_1"] = translator.translate(out_df.at[i, "Explanation_1"])

            if args.save_every > 0 and (i + 1) % args.save_every == 0:
                out_df.to_excel(out_path, index=False)
                print(f"Saved {i + 1}/{len(out_df)}")

        out_df.to_excel(out_path, index=False)
        print(f"[Done] {out_path}")


def run_translation_eval(args):
    evaluator = TranslationEvaluation(
        datasets={
            "qwen_backtranslation": "esnli_selected_translated_qwen_backtranslated.csv",
        }
    )
    results = evaluator.evaluate_one("qwen_backtranslation", include_comet=args.with_comet)
    print("\n[BLEU/chrF - qwen_backtranslation]")
    print(results["bleu_chrf"].to_string(index=False))
    if "comet_qe" in results:
        print("\n[COMET-QE]")
        print(results["comet_qe"].to_string(index=False))

    icc_df = HumanEvalICC(filename="esnli_translation_de_human_eval.xlsx").evaluate()
    print("\n[Human Eval ICC(2,1)]")
    print(icc_df.to_string(index=False))


def run_explanation_de(args):
    path = args.de_file
    df = pd.read_excel(path)
    if "Explanation_de_generated" not in df.columns:
        df["Explanation_de_generated"] = pd.NA

    generator = ExplanationGenerationDE(model_name=args.de_model)

    for i in range(len(df)):
        value = df.at[i, "Explanation_de_generated"]
        if pd.notna(value) and str(value).strip():
            continue

        df.at[i, "Explanation_de_generated"] = generator.generate(
            premise=str(df.at[i, "Sentence1_de"]),
            hypothesis=str(df.at[i, "Sentence2_de"]),
            label=str(df.at[i, "gold_label"]),
        )

        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            df.to_excel(path, index=False)
            print(f"Saved {i + 1}/{len(df)}")

    df.to_excel(path, index=False)
    print(f"[Done] {path}")


def run_agreement(args):
    file_map = {
        "translated": "data/translated_and_generated_dataset/esnli_translated_explanations_de_human_eval.xlsx",
        "generated": "data/translated_and_generated_dataset/esnli_generated_explanations_de_human_eval.xlsx",
    }
    dimensions = ["Well-Written", "Related", "Factual", "New Information", "Unnecessary Information"]
    targets = ["translated", "generated"] if args.agreement_target == "all" else [args.agreement_target]

    for key in targets:
        path = file_map[key]
        df = pd.read_excel(path)
        calc = ImbalancedAgreementCalculator(df)
        results = calc.compute_multiple_dimensions(dimensions, suffix_1="_1", suffix_2="_2")

        rows = []
        for dim in dimensions:
            r = results[dim]
            rows.append(
                {
                    "dimension": dim,
                    "n": r["n"],
                    "observed_agreement": r["observed_agreement"],
                    "cohen_kappa": r["cohen_kappa"],
                    "pabak": r["pabak"],
                    "gwet_ac1": r["gwet_ac1"],
                    "positive_agreement": r["positive_agreement"],
                    "negative_agreement": r["negative_agreement"],
                }
            )

        print(f"\n[Agreement: {key}] {path}")
        print("Raters: *_1 vs *_2 (within this file only)")
        print(pd.DataFrame(rows).to_string(index=False))


def run_llm_judge(args):
    if args.judge_backend == "hf":
        output = args.judge_output or "data/results_llm-as-a-judge/result_de_prometheus.xlsx"
        model = args.judge_model or "prometheus-eval/prometheus-7b-v2.0"
        run_hf_judge(
            input_excel=args.judge_input,
            output_excel=output,
            fewshot_json=args.judge_fewshot,
            model_id=model,
            save_every=args.save_every,
            max_new_tokens=args.judge_max_new_tokens,
        )
    else:
        output = args.judge_output or "data/results_llm-as-a-judge/result_de_qwen.xlsx"
        model = args.judge_model or "qwen2.5-72b-instruct"
        run_api_judge(
            input_excel=args.judge_input,
            output_excel=output,
            fewshot_json=args.judge_fewshot,
            base_url=args.judge_base_url,
            model=model,
            save_every=args.save_every,
            max_new_tokens=args.judge_max_new_tokens,
        )


def main():
    parser = argparse.ArgumentParser(description="Project main entrypoint.")
    parser.add_argument("--module", default="translation")
    parser.add_argument("--input", default="data/esnli_selected.xlsx")
    parser.add_argument("--sheet", default="English")
    parser.add_argument("--output-dir", default="data/translated_and_generated_dataset")
    parser.add_argument("--models", default="all")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--with-comet", action="store_true")
    parser.add_argument("--agreement-target", choices=["all", "translated", "generated"], default="all")
    parser.add_argument("--judge-backend", choices=["hf", "api"], default="api")
    parser.add_argument(
        "--judge-input",
        default="data/translated_and_generated_dataset/esnli_de_generated_ger_explanations_gpt41mini.xlsx",
    )
    parser.add_argument("--judge-output", default=None)
    parser.add_argument("--judge-fewshot", default="data/few-shot_examples_de.json")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default="https://chat-ai.academiccloud.de/v1")
    parser.add_argument("--judge-max-new-tokens", type=int, default=180)
    parser.add_argument(
        "--de-file",
        default="data/translated_and_generated_dataset/esnli_de_generated_ger_explanations_gpt41mini.xlsx",
    )
    parser.add_argument("--de-model", default="gpt-4.1-mini")
    args = parser.parse_args()

    if args.module == "translation":
        run_translation(args)
    elif args.module == "translation_eval":
        run_translation_eval(args)
    elif args.module == "explanation_de":
        run_explanation_de(args)
    elif args.module == "agreement":
        run_agreement(args)
    elif args.module == "llm_judge":
        run_llm_judge(args)
    else:
        print(f"Module not implemented yet: {args.module}")


if __name__ == "__main__":
    main()
