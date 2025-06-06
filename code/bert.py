import argparse
import os
import time
from datetime import timedelta

import pandas as pd
import torch
import torchinfo
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

import utils


def load_data(data_path):
    df = pd.read_csv(data_path)
    df['class_label'] = df['Insomnia'].map(lambda x: utils.label_conv[x]).values
    return df


def tokenize_data(texts, tokenizer, max_length):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )


def create_dataset(encodings, labels):
    return utils.CustomDataset(
        encodings=encodings,
        labels=torch.tensor(labels)
    )


def main(args):
    if args.cuda_gpu_id != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpu_id
    _ = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()
    print("\n\nNumber of GPUs available:", torch.cuda.device_count())

    set_seed(0)

    # Load data
    df_train = load_data(args.train_data_dir)
    df_val = load_data(args.val_data_dir)
    df_test = load_data(args.test_data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_encodings = tokenize_data(df_train['text'].astype(str).tolist(), tokenizer, args.seq_len)
    val_encodings = tokenize_data(df_val['text'].astype(str).tolist(), tokenizer, args.seq_len)
    test_encodings = tokenize_data(df_test['text'].astype(str).tolist(), tokenizer, args.seq_len)

    train_dataset = create_dataset(train_encodings, df_train['class_label'].astype(int).tolist())
    val_dataset = create_dataset(val_encodings, df_val['class_label'].astype(int).tolist())
    test_dataset = create_dataset(test_encodings, df_test['class_label'].astype(int).tolist())

    print("\n\nTrain data length:", len(train_dataset))
    print("Val data length:", len(val_dataset))
    print("Test data length:", len(test_dataset))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    torchinfo.summary(model)

    training_args = TrainingArguments(
        tf32=True,
        dataloader_num_workers=4,
        output_dir=args.out_path,
        disable_tqdm=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_steps=0,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_safetensors=False,
        seed=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=utils.compute_metrics_text_class
    )

    print("\n\nStarting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print("\n\nTotal training time:", str(timedelta(seconds=end_time - start_time)))

    print("\n\nEvaluating on test set...")
    test_preds = trainer.predict(test_dataset)
    print("\n\nPerformance on test set:", utils.compute_metrics_text_class(test_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a BERT-based model for insomnia classification")

    parser.add_argument("--seq_len", type=int, default=4096, help="Maximum sequence length for tokenization")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--cuda_gpu_id", type=str, default="0,1", help="CUDA GPU IDs to use")
    parser.add_argument("--model_name", type=str, default="yikuan8/Clinical-BigBird", help="Model name or path")
    parser.add_argument("--train_data_dir", type=str, default="../data/mimic_corpus/train.csv", help="Path to training dataset")
    parser.add_argument("--val_data_dir", type=str, default="../data/mimic_corpus/val.csv", help="Path to validation dataset")
    parser.add_argument("--test_data_dir", type=str, default="../data/mimic_corpus/test.csv", help="Path to test dataset")
    parser.add_argument("--out_path", type=str, default="./model_weights", help="Output directory for model checkpoints")

    args = parser.parse_args()
    main(args)
