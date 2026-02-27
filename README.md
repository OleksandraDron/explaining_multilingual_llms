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
- Run `main.py` to execute the pipeline
- The system translates the dataset, generates explanations, and computes evaluation scores
