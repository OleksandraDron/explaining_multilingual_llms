# Multilingual NLI Explanation Project

This project explores multilingual explanation generation and evaluation for Natural Language Inference (NLI).

## Project Scope
- English to German machine translation
- English and German explanation generation
- Automatic evaluation
- LLM-as-a-Judge experiments

## Project Structure
- `src/translation.py`
- `src/explanation_generation_en.py`
- `src/explanation_generation_de.py`
- `src/translation_evaluation.py`
- `src/llm_as_a_judge_de.py`
- `src/imbalanced_agreement.py`
- `src/faithfulness.py` - re-engineered CCSHAP

## Requirements
- Python 3.10 or higher
- Install dependencies via `requirements.txt`

## Execution
German group pipeline entrypoint: `main.py`

Set API keys first:
- `OPENAI_API_KEY` (for GPT models)
- `ACADEMIC_API_KEY` (for Academic Cloud endpoint)

Run modules in this order:

1. Translation (EN -> DE, 3 models)
```bash
python main.py --module translation --models all
```

2. Translation quality evaluation (qwen backtranslation as example + ICC of gpt 4.1 mini translated results)
```bash
python main.py --module translation_eval
```

3. Generate German explanations from corrected translations
```bash
python main.py --module explanation_de
```

4. Human agreement on evaluation of generated and translated German explanations 
```bash
python main.py --module agreement --agreement-target all
```

5. LLM-as-a-Judge on generated German explanations (2 models)
```bash
# API backend: Qwen 2.5 72B
python main.py --module llm_judge --judge-backend api --judge-model qwen2.5-72b-instruct

# HF backend: Prometheus 7B
python main.py --module llm_judge --judge-backend hf --judge-model prometheus-eval/prometheus-7b-v2.0
```

## Datasets

### `data/` (root)
- `esnli_selected.xlsx`:
  Main e-SNLI subset used as pipeline input (`idx`, `gold_label`, `Sentence1`, `Sentence2`, `Explanation_1`).
- `few-shot_examples_de.json`:
  Few-shot examples used by German LLM-as-a-Judge prompts.
- `English_dataset_extension.xlsx`:
  Extended English dataset used in English-side experiments.
- `English_human_evaluation.xlsx`:
  Human annotation/evaluation file for English-side results.

### `data/translated_and_generated_dataset`
- `esnli_de_generated_ger_explanations_gpt41mini.xlsx`:
  Contains original fields, final corrected German translations from GPT-4.1-mini (`Sentence1_de`, `Sentence2_de`, `Explanation_1_de`), and generated German explanations (`Explanation_de_generated`).
- `esnli_selected_translated_qwen_backtranslated.csv`:
  Qwen translation + backtranslation file used for BLEU/chrF evaluation.
- `esnli_translation_de_human_eval.xlsx`:
  Human translation evaluation/correction file for premise-hypothesis pairs; includes `Sen1_Human1/2` and `Sen2_Human1/2` for ICC.
- `esnli_translated_explanations_de_human_eval.xlsx`:
  Human ratings for translated explanations (`Well-Written_1/2`, `Related_1/2`, etc.).
- `esnli_generated_explanations_de_human_eval.xlsx`:
  Human ratings for generated explanations (`Well-Written_1/2`, `Related_1/2`, etc.).

### `data/results_llm-as-a-judge`
- `result_de_qwen.xlsx`:
  LLM-as-a-Judge results produced by API backend (`qwen2.5-72b-instruct`).
- `result_de_prometheus.xlsx`:
  LLM-as-a-Judge results produced by HF backend (`prometheus-eval/prometheus-7b-v2.0`).
