#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import pickle
import json
import time
from datetime import timedelta
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import utils


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM inference for insomnia detection")
    parser.add_argument(
        '--data_dir',
        type=str,
        default="../data/mimic_corpus/train.csv",
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16",
        help='Model name or path'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=.6,
        help='Sampling temperature for the model'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=.9,
        help='Top-p sampling value for the model'
    )
    parser.add_argument(
        '--prompt_path',
        type=str,
        default="./prompts/mimic_prompt.json",
        help='Path to the prompt JSON file'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default="./llm_insomnia_preds.pkl",
        help='Path to save predictions (Pickle)'
    )
    parser.add_argument(
        '--cuda_gpu_id',
        type=str,
        default="0,1,2,3",
        help='CUDA GPU IDs to use'
    )
    return parser.parse_args()


def setup_environment(cuda_gpu_id):
    """Set up CUDA environment and verify GPU availability."""
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    if cuda_gpu_id != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_gpu_id
    _ = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()
    print("Number of GPUs available:", torch.cuda.device_count())
    return torch.cuda.device_count()


def load_data(data_dir):
    """Load and preprocess the clinical notes dataset."""
    df = pd.read_csv(data_dir)
    df.rename(columns={
        'Insomnia': utils.note_label,
        'Definition 1': utils.arr_label[0],
        'Definition 2': utils.arr_label[1],
        'Rule A': utils.arr_label[2],
        'Rule B': utils.arr_label[3],
        'Rule C': utils.arr_label[4],
    }, inplace=True)

    for label in utils.arr_label:
        df[label] = df[label].map(lambda x: utils.label_conv[x]).values

    df['note_id'] = df['note_id'].astype(str).values
    assert not df['note_id'].duplicated().any(), "Duplicate note_ids found"
    df.index = df['note_id'].values
    return df


def load_prompt(prompt_path):
    """Load prompt configuration from JSON file."""
    with open(prompt_path, 'r') as f:
        prompt_json = json.load(f)
    prompt_json['arr_prompt_examples'] = [str(x) for x in prompt_json['arr_prompt_examples']]
    return prompt_json


def load_model(model_name, max_input_len, num_gpus):
    print("\n\nLoading model:", model_name)
    start_time = time.time()
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        dtype=torch.bfloat16,
        gpu_memory_utilization=.9,
        max_model_len=max_input_len
    )
    print("\n\nModel loaded in", timedelta(seconds=time.time() - start_time))
    return model


def main():
    args = parse_arguments()
    num_gpus = setup_environment(args.cuda_gpu_id)
    max_input_len, max_output_len = 90000, 8192

    df_data = load_data(args.data_dir)
    prompt_json = load_prompt(args.prompt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    arr_prompt_examples = prompt_json['arr_prompt_examples']
    user_msg_template = prompt_json['user_msg_template']
    arr_prompt = prompt_json['prompt']

    model = load_model(args.model_name, max_input_len, num_gpus)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_output_len,
        seed=0,
        detokenize=True,
        guided_decoding=GuidedDecodingParams(
            json=utils.InsomniaFormat.model_json_schema()
        )
    )

    df_eval = df_data[~df_data['note_id'].isin(arr_prompt_examples)].copy()
    arr_input_prompt = utils.create_prompt(
        df_eval=df_eval,
        func_format_user=utils.func_format_user,
        messages=arr_prompt,
        user_template=user_msg_template
    )
    print("\n\nNumber of texts to predict:", len(arr_input_prompt))
    arr_tok_prompt = [
        tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True
        )
        for prompt in arr_input_prompt
    ]
    print(f"\n\nLongest seq len: {max(pd.Series(arr_tok_prompt).apply(len))}")
    
    start_time = time.time()
    arr_text_pred = utils.eval_prompt(
        arr_input_prompt=arr_input_prompt,
        tokenizer=tokenizer,
        model=model,
        sampling_params=sampling_params
    )
    print("\n\nModel inference time:", timedelta(seconds=time.time() - start_time))

    df_pred, arr_bad_format = utils.format_preds(
        arr_preds=arr_text_pred,
        df_eval=df_eval,
        arr_label=utils.arr_label,
        label_conv=utils.label_conv,
        pos_value=utils.pos_value,
        neg_value=utils.neg_value,
        answer_value=utils.answer_value
    )
    print(f"\n\nBadly formatted examples: {len(arr_bad_format)}")
    if len(arr_bad_format):
        for bad_format in arr_bad_format:
            print(f"i: {bad_format[0]}, labels: {bad_format[1]}")
            print(arr_text_pred[bad_format[0]])
            print()

    utils.compute_metrics(
        df_eval=df_eval,
        df_pred=df_pred,
        note_label=utils.note_label,
        arr_rule_label=utils.arr_rule_label
    )

    with open(args.out_path, 'wb') as file:
        pickle.dump(arr_text_pred, file)

if __name__ == "__main__":
    main()
