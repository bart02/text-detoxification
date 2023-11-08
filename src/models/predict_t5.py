import click
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextStreamer

transformers.enable_full_determinism(420)
PREFFIX = "detoxify: "


@click.command(context_settings={'show_default': True, 'help_option_names': ['-h', '--help'], 'max_content_width': 120})
@click.option('--model_name_or_path', default="batalovme/t5-detox",
              help='Model for prediction')
def predict(model_name_or_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    print("Model loaded, device is", model.device)
    print()
    print("Enter your message:")
    while True:
        mes = input("> ")
        tokenized = tokenizer(PREFFIX + mes, return_tensors="pt")
        tokenized.to(device)
        streamer = TextStreamer(tokenizer, skip_special_tokens=True)
        print("<", end=" ")
        _ = model.generate(**tokenized, streamer=streamer, max_new_tokens=128)


if __name__ == "__main__":
    predict()