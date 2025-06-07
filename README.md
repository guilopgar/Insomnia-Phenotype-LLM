# Insomnia-Phenotype-LLM
This repository contains the code and data used in our study: **"Automated Insomnia Phenotyping from Electronic Health Records: Leveraging Large Language Models to Decode Clinical Narratives"**.


## Overview
We present a fully automated framework for identifying patients likely experiencing insomnia using unstructured clinical notes. Our framework integrates:

* **Generative large language models (LLMs)** like Llama-3-70B and Llama-3-405B, and
* **BERT-based classification models**, including both general-domain models and domain-adapted variants such as Clinical-Longformer and Clinical-BigBird.

Our LLMs leverage **few-shot prompting**, **chain-of-thought reasoning**, and **rule-based criteria** grounded in clinical guidelines to extract insomnia-related signals from electronic health records. Applied to both ICU and outpatient notes from two distinct institutions, our LLM-based approach achieved state-of-the-art performance on both the MIMIC and KUMC corpora, approaching human-level annotation agreement.


## 📊 Results Summary

| Model               | MIMIC F1 | KUMC F1  |
| ------------------- | -------- | -------- |
| **Llama-3-70B**     | **93.0** | 84.1     |
| **Llama-3-405B**    | 91.6     | **85.7** |
| Clinical-BigBird    | 81.0     | —        |
| Clinical-Longformer | 75.9     | —        |
| BigBird             | 80.0     | —        |
| Longformer          | 71.8     | —        |

> Human inter-annotator agreement (IAA): **94.6** F1 on MIMIC and **88.9** F1 on KUMC.


## 🏥 MIMIC Corpus
We provide a corpus of ICU clinical notes from MIMIC-III v1.4 annotated for detecting patients with insomnia. Each note in the corpus is enriched with structured demographic and medication data.

Our gold standard annotations are located in [`data/mimic_corpus`](data/mimic_corpus). To reconstruct the full corpus, including note text and metadata, use the script [`text_mimic_notes.py`](data/text_mimic_notes.py). Access to [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) via PhysioNet is required.

### Usage

```bash
python data/text_mimic_notes.py \
  --annotations_path /data/mimic_corpus/val.csv \
  --mimic_path data/mimic-iii/1.4 \
  --output_path /data/mimic_corpus/val_with_text.csv
```

This command will:

* Read annotations from `val.csv`
* Retrieve the corresponding note content and associated structured fields from MIMIC-III
* Generate a new file `val_with_text.csv` with the appended `text` column


## 🧠 LLM Inference
Run LLM-based predictions using [`code/llm.py`](code/llm.py):

```bash
python code/llm.py \
  --data_dir data/mimic_corpus/test.csv \
  --prompt_path code/prompts/mimic_prompt.json \
  --model_name meta-llama/Llama-3.1-70B-Instruct \
  --temperature 0.3 \
  --top_p 0.4 \
  --out_path results/llm_preds.pkl \
  --cuda_gpu_id 0,1
```

This command reproduces the results for **Llama-3-70B**. To reproduce **Llama-3-405B**, use the default configuration already provided in the script:

* `--model_name neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16`
* `--temperature 0.6`, `--top_p 0.9`

LLM hyperparameters (`--temperature` and `--top_p`) were empirically selected to maximize validation F1 performance.


## 🤖 BERT-based Classifier

Train and evaluate BERT-based models using [`code/bert.py`](code/bert.py):

```bash
python code/bert.py \
  --train_data_dir data/mimic_corpus/train.csv \
  --val_data_dir data/mimic_corpus/val.csv \
  --test_data_dir data/mimic_corpus/test.csv \
  --model_name yikuan8/Clinical-BigBird \
  --cuda_gpu_id 0,1 \
  --out_path results/bert_model \
  --seq_len 4096 \
  --epochs 20 \
  --lr 2e-5 \
  --bs 4
```

### Configurations used in the paper
Each model was tuned by selecting the learning rate (LR) and batch size (BS) that yielded the highest F1 score on the validation set:

| Model                | `--model_name`                 | LR   | BS |
| -------------------- | ------------------------------ | ---- | -- |
| Clinical-BigBird     | `yikuan8/Clinical-BigBird`     | 2e-5 | 4  |
| Clinical-Longformer  | `yikuan8/Clinical-Longformer` | 2e-5 | 4  |
| BigBird              | `google/bigbird-roberta-base`  | 3e-5 | 3  |
| Longformer           | `allenai/longformer-base-4096` | 3e-5 | 3  |


## 📚 Citation

If you use our code or data, please cite our work:

```bibtex
@article{Lopez-Garcia2025.06.02.25328701,
   author = {Lopez-Garcia, Guillermo and Weissenbacher, Davy and Stadler, Matthew and O{\textquoteright}Connor, Karen and Xu, Dongfang and Gryboski, Lauren and Heavens, Jared and Abu-el-Rub, Noor and Mazzotti, Diego R. and Chakravorty, Subhajit and Gonzalez-Hernandez, Graciela},
   title = {Automated Insomnia Phenotyping from Electronic Health Records: Leveraging Large Language Models to Decode Clinical Narratives},
   year = {2025},
   doi = {10.1101/2025.06.02.25328701},
   publisher = {Cold Spring Harbor Laboratory Press},
   URL = {https://www.medrxiv.org/content/early/2025/06/03/2025.06.02.25328701},
   journal = {medRxiv}
}
```

