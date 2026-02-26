import os

from src.llm_as_a_judge_de import HuggingFaceJudgeDE, QwenApiJudgeDE

def run_hf_judge(
    input_excel="data/esnli_de_generated_ger_explanations_gpt41mini.xlsx",
    output_excel="data/results_llm-as-a-judge/result_de_prometheus.xlsx",
    fewshot_json="data/few-shot_examples_de.json",
):
    """Run German LLM-as-a-judge with local Hugging Face model."""
    judge = HuggingFaceJudgeDE(
        fewshot_json=fewshot_json,
    )
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    judge.run_excel(
        excel_in=input_excel,
        excel_out=output_excel,
        save_every=5,
        max_new_tokens=180,
    )


def run_api_judge(
    input_excel="data/esnli_de_generated_ger_explanations_gpt41mini.xlsx",
    output_excel="data/results_llm-as-a-judge/result_de_prometheus.xlsx",
    fewshot_json="data/few-shot_examples_de.json",
):
    """Run German LLM-as-a-judge with Qwen API backend."""
    judge = QwenApiJudgeDE(
        fewshot_json=fewshot_json,
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://chat-ai.academiccloud.de/v1",
        model="qwen2.5-72b-instruct",
    )
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    judge.run_excel(
        excel_in=input_excel,
        excel_out=output_excel,
        save_every=5,
        max_new_tokens=180,
    )


def main():
    """Select which task to run from one simple entry point."""
    task = "judge_hf"
    if task == "judge_hf":
        run_hf_judge()
    elif task == "judge_api":
        run_api_judge()
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
