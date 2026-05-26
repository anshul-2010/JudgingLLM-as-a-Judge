# JudgingLLM-as-a-Judge
## Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation

Purpose
-------
This repository contains code and utilities used for the experiments in the research project "Judging LLMs as a Judge". It provides scripts to probe BERT-based models and evaluate classification probes, utilities for topic modelling and interpretability, and helper functions used to run and reproduce experiments.

Dependencies
------------
- Python 3.1+ recommended
- Commonly used packages (install via pip):

```
pip install torch transformers scikit-learn pandas numpy bertopic umap-learn tqdm matplotlib seaborn captum
```

Repository structure
--------------------
- [Bert_balanced_cla_eval_probe.py](Bert_balanced_cla_eval_probe.py) — Balanced classification evaluation probe scripts.
- [Bert_balanced_cla_hard_probe.py](Bert_balanced_cla_hard_probe.py) — Hard (difficult) balanced classification probes.
- [Bert_classification_eval_probe.py](Bert_classification_eval_probe.py) — Standard classification evaluation probes.
- [Bert_classification_hard_probe.py](Bert_classification_hard_probe.py) — Hard classification probe variants.
- [cross_dataset_classification.py](cross_dataset_classification.py) — Utilities and scripts for cross-dataset experiments.
- [llm_as_judge_only_rubric.py](llm_as_judge_only_rubric.py) — LLM-as-judge evaluation using only rubric text.
- [llm_as_judge_response_and_rubric.py](llm_as_judge_response_and_rubric.py) — LLM-as-judge using both model responses and rubric.
- [utilities/bertopic.py](utilities/bertopic.py) — BERTopic helper wrappers.
- [utilities/integrated_gradients.py](utilities/integrated_gradients.py) — Integrated Gradients utilities for interpretability.
- [utilities/llm_as_judge_features.py](utilities/llm_as_judge_features.py) — Feature extraction helpers for LLM-as-judge experiments.
- [utilities/umap.py](utilities/umap.py) — UMAP-related utilities and wrappers.
