import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import json
from enum import Enum
from pydantic import BaseModel
import torch


# Insomnia-related labels
answer_value = 'answer'
pos_value, neg_value = 'yes', 'no'
label_conv = {pos_value: 1, neg_value: 0}
arr_label = ['definition_1', 'definition_2', 'rule_a', 'rule_b', 'rule_c', 'insomnia_status']
arr_rule_label = arr_label[:-1]
note_label = arr_label[-1]


# Structured output JSON format
class AnswerValue(str, Enum):
    pos_val = pos_value
    neg_val = neg_value
    
class RuleItem(BaseModel):
    explanation: str
    answer: AnswerValue

class InsomniaFormat(BaseModel):
    definition_1: RuleItem
    definition_2: RuleItem
    rule_a: RuleItem
    rule_b: RuleItem
    rule_c: RuleItem
    insomnia_status: RuleItem


# LLM generation
def func_format_user(template, row):
    return template.format(
        text=row['text']
    )

def create_prompt(
        df_eval,
        func_format_user,
        messages,
        user_template
):
    # Add evaluation texts
    return [
        [
            *messages,
            {"role": "user", "content": func_format_user(user_template, row)}
        ]
        for _, row in df_eval.iterrows()
    ]


def eval_prompt(arr_input_prompt, tokenizer, model, sampling_params):
    # Model inference
    arr_tok_prompt = [
        tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        for prompt in arr_input_prompt
    ]
    arr_outputs = model.generate(
        prompts=arr_tok_prompt,
        sampling_params=sampling_params
    )

    arr_response = [output.outputs[0].text for output in arr_outputs]

    return arr_response


def extract_label_pred(dict_note_pred, arr_label, label_conv, pos_value, neg_value, answer_value):
    dict_res = {}
    arr_bad_format = []
    for label in arr_label:
        assert label in dict_note_pred
        dict_label = dict_note_pred[label]
        assert answer_value in dict_label
        answer = dict_label[answer_value].strip().lower()
        dict_res[label] = label_conv[pos_value] if pos_value in answer else label_conv[neg_value]
        if answer not in (pos_value, neg_value):
            arr_bad_format.append(label)
    return dict_res, arr_bad_format


def format_preds(arr_preds, df_eval, arr_label, label_conv, pos_value, neg_value, answer_value):
    dict_pred = {}
    arr_bad_format = []
    for i in range(len(arr_preds)):
        note_id = df_eval.iloc[i]['note_id']
        dict_pred_note, arr_bad_format_note = extract_label_pred(
            dict_note_pred=json.loads(arr_preds[i]),
            arr_label=arr_label,
            label_conv=label_conv,
            pos_value=pos_value,
            neg_value=neg_value,
            answer_value=answer_value
        )
        dict_pred[note_id] = dict_pred_note
        if len(arr_bad_format_note):
            arr_bad_format.append((i, arr_bad_format_note))
        
    return pd.DataFrame.from_dict(dict_pred, orient='index'), arr_bad_format


# BERT-based models
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)



# Evaluation metrics
def compute_metrics(df_eval, df_pred, note_label, arr_rule_label):
    print("Note-level evaluation")
    y_note_gs = df_eval[note_label].values
    y_note_pred = df_pred.loc[df_eval.index, note_label].values
    f1 = f1_score(y_note_gs, y_note_pred)
    precision = precision_score(y_note_gs, y_note_pred)
    recall = recall_score(y_note_gs, y_note_pred)
    print(f"Precision: {round(precision * 100, 1)}, Recall: {round(recall * 100, 1)}, F1: {round(f1 * 100, 1)}")
    print(f"Pred: {np.count_nonzero(y_note_pred == 1)}")
    print(f"Support: {np.count_nonzero(y_note_gs == 1)}")
    print()
    print("Rule-level evaluation")
    print(
        classification_report(
            y_true=df_eval[arr_rule_label].values,
            y_pred=df_pred.loc[df_eval.index][arr_rule_label].values,
            digits=3,
            zero_division=0.0,
            target_names=arr_rule_label
        )
    )


def compute_metrics_text_class(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {
        'precision': round(precision * 100, 1),
        'recall': round(recall * 100, 1),
        'f1': round(f1 * 100, 1),
        'pred': np.count_nonzero(preds == 1),
        'support': np.count_nonzero(labels == 1),
    }
    

