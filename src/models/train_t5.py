from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

metric = evaluate.load("bleu")
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

def compute_metrics(eval_pred):
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

def tokenize(examples):
    inputs = [PREFFIX + e for e in examples["reference"]]
    targets = examples["translation"]
    
    tokenized_inputs = tokenizer(inputs, max_length=128, truncation=True)
    tokenized_targets = tokenizer(targets, max_length=128, truncation=True)

    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

PREFFIX = "detoxify: "
dataset = DatasetDict.load_from_disk("../../data/interim/splitted_dataset")
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=['reference', 'translation'])

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", device_map="auto")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

gconf = model.generation_config
gconf.max_new_tokens = 128
gconf.repetition_penalty = 1.2

training_args = Seq2SeqTrainingArguments(
    output_dir="t5_detox",
    evaluation_strategy="steps",
    eval_steps=2000,
    logging_strategy="steps",
    logging_steps=200,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    generation_config=gconf,
    report_to='wandb',
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
