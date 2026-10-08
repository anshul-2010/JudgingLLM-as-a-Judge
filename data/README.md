# Data

## `healthbench_probe/`

Rubric-only probe datasets built from HealthBench (Arora et al., 2025; MIT licence).
One CSV per candidate response model × HealthBench split:

| File | Response model | Split |
|---|---|---|
| `gemma_{eval,hard}_judge.csv` | google/gemma-7b-it | Eval / Hard |
| `llama_{eval,hard}_judge.csv` | meta-llama/Llama-3.1-8B-Instruct | Eval / Hard |
| `medgemma_{eval,hard}_judge.csv` | google/medgemma-1.5-4b-it | Eval / Hard |
| `medllama_{eval,hard}_judge.csv` | MMed-Llama-3-8B | Eval / Hard |

Each row is one rubric criterion. Labels come from the Qwen2.5-7B-Instruct judge.

| Column | Description |
|---|---|
| `rubric_text` | Rubric criterion text (the only probe input) |
| `binary_label` | Judge verdict given conversation + candidate response + rubric (0 = not met, 1 = met). Probe target. |
| `no_context_binary_label` | Judge verdict given the rubric alone (`src/judge/judge_rubric_only.py`) |
| `char_len`, `word_len` | Rubric length in characters / words |
| `digit_count`, `question_count` | Number of digits / question marks in the rubric |
| `bert_label` | Probe prediction (Gemma files only) |

## `healthbench_probe/healthbench/`

Raw HealthBench releases. Each line is one conversation with fields `prompt`, `rubrics` (criterion + points), `example_tags`, `ideal_completions_data`, `prompt_id` and `canary`.

| File | Split | Conversations |
|---|---|---|
| `2025-05-07-06-14-12_oss_eval.jsonl` | Eval | 5,000 |
| `hard_2025-05-08-21-00-10.jsonl` | Hard | 1,000 |
| `consensus_2025-05-09-20-00-46.jsonl` | Consensus | 3,671 |

## `researchrubrics_probe/`

ResearchRubrics (Sharma et al., 2026): 101 research prompts with instance-specific rubrics. Metadata columns: `sample_id`, `domain`, `conceptual_breadth`, `logical_nesting`, `exploration`.

| File | Rows | Contents |
|---|---|---|
| `research_rubrics.csv` | 101 | One row per prompt, all `rubrics` |
| `research_rubrics_{gemma,llama,medgemma}.csv` | 101 | Same, plus the candidate model's answer (`{model}_answer`) |
| `research_df.csv` | 2,593 | One row per rubric criterion (`rubrics_parsed`), linked to its prompt by `conversation_idx` |

Counterfactual perturbation data is stored next to its scripts in `src/counterfactual_perturbations/`.
