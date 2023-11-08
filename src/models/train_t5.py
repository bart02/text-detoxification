import click
import torch
import transformers
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

transformers.enable_full_determinism(420)
PREFFIX = "detoxify: "


def compute_metrics(metric, tokenizer, eval_pred):
    """Metric function for the Trainer, computes BLEU and average generation length."""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    print(result)
    result = {"bleu": result["bleu"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def tokenize(tokenizer, examples):
    inputs = [PREFFIX + e for e in examples["reference"]]
    targets = examples["translation"]

    tokenized_inputs = tokenizer(inputs, max_length=128, truncation=True)
    tokenized_targets = tokenizer(targets, max_length=128, truncation=True)

    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs


@click.command(context_settings={'show_default': True, 'help_option_names': ['-h', '--help'], 'max_content_width': 120})
@click.option('--base_model_name_or_path', default="humarin/chatgpt_paraphraser_on_T5_base",
              help='Base model for training')
@click.option('--dataset_path', default="../../data/interim/splitted_dataset", help='Path to dataset',
              type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option('--output_dir', default="../../models/t5_detox", help='Output directory', type=click.Path())
@click.option('--num_train_epochs', default=4, help='Total number of training epochs to perform')
@click.option('--batch_size', default=32)
@click.option('--save_total_limit', default=3, help='Number of checkpoints to save')
def train(base_model_name_or_path: str, dataset_path: str, output_dir: str, num_train_epochs: int, batch_size: int,
          save_total_limit: int):
    if not torch.cuda.is_available():
        click.echo(click.style("CUDA is not available, training on CPU/MPS. This will be slow.",
                               bold=True, fg="red"))

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_or_path)

    print("Loading dataset")
    dataset = DatasetDict.load_from_disk(dataset_path)
    dataset.pop("test")

    print("Tokenizing dataset")
    tokenized_dataset = dataset.map(lambda x: tokenize(tokenizer, x), batched=True,
                                    remove_columns=['reference', 'translation'],
                                    # cache_file_names={"train": ".train_tokenized.bin",
                                    #                   "validation": ".validation_tokenized.bin"}
                                    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Generation config for T5, setted to 128 tokens and repetition penalty to 1.2 by empirical testing
    gconf = model.generation_config
    gconf.max_new_tokens = 128
    gconf.repetition_penalty = 1.2

    metric = evaluate.load("bleu")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        logging_strategy="steps",
        logging_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        generation_config=gconf,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(metric, tokenizer, x),
    )
    print("Training")
    trainer.train()


if __name__ == "__main__":
    train()
