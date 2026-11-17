# Comparing Prompting Techniques for Improving LLM Accuracy on Math Word Problems

Code, data and results for ICISAS 2026 paper #442.

Authors: Dr. Benita Veronica, Dania Ali Khan, Dr. Geetha Mohan, Kekweditse Mogoba, Blessed Dendere
School of Electrical Engineering, Computing and Mathematical Sciences, Curtin University Dubai

## What this study does

Four prompting techniques are compared on the same 200 GSM8K test problems using **Qwen/Qwen2.5-3B-Instruct**:

| Method | Decoding | Max new tokens | Correct / 200 | Accuracy |
|---|---|---|---|---|
| Direct Prompting | greedy (`do_sample=False`) | 100 | 24 | 12.00% |
| Zero-shot CoT | greedy | 512 | 169 | 84.50% |
| Few-shot CoT (4 examples, GSM8K train indices 0–3) | greedy | 768 | 169 | 84.50% |
| Self-Consistency (5 samples, majority vote) | `temperature=0.7`, `top_p=0.95` | 768 | 181 | 90.50% |

## Data

- Source: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) on Hugging Face, `main` configuration.
- The 200 test problems are `load_dataset("openai/gsm8k", "main")["test"].shuffle(seed=42).select(range(200))`.
- `data/gsm8k_200_questions.csv` – the 200 problems (`id` 1–200 as used in the paper, question, GSM8K solution, numeric answer).
- `data/gsm8k_200_question_ids.csv` – maps each paper `id` to its original index in the 1,319-problem GSM8K test split.

## Results files (`results/`)

| File | Contents |
|---|---|
| `direct_prompting_results.csv` | raw response, extracted answer, correct flag |
| `zero_shot_cot_results.csv` | raw response, extracted answer (after extraction audit), original extracted answer, correct flag |
| `few_shot_cot_results.csv` | raw response, extracted answer, correct flag, extraction-failure flag |
| `self_consistency_results.csv` | all 5 raw responses and extracted answers per question, voted answer, tie flag, extraction failures |
| `self_consistency_checkpoint_192.csv` | checkpoint saved after 192 questions; the run was resumed with the same settings for questions 193–200 |

## Code

`GSM8K_Prompting_Experiment.ipynb` contains the prompts, generation settings, answer extraction, voting and evaluation. It was run on Google Colab with an NVIDIA Tesla T4 GPU (16 GB), using `transformers`, `accelerate` and `datasets` installed with pip in September 2026. Exact library versions were not recorded, and the Self-Consistency samples were not seeded.

## Statistics

- 95% confidence intervals: Wilson score interval.
- Pairwise comparisons: exact two-sided McNemar tests, with Holm correction for the six comparisons.
