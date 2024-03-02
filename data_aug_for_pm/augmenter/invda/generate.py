import torch
import argparse
import spacy

from transformers import T5ForConditionalGeneration, AutoTokenizer

from data_aug_for_pm.utils.load_config import ExperimentConfiguration

try:
    nlp = spacy.load('en_core_web_sm')
except:  # untested.
    from spacy.cli.download import download
    download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')


def generate(model, tokenizer, device, sentence, experiment_configuration: ExperimentConfiguration):
    """Generate using a T5 seq2seq model

    Args:
        model: the T5 model state
        tokenizer: the T5 tokenizer
        device: cpu or cuda
        sentence (str): the input string

    Returns:
        List of str: the augmentations
    """
    # text = "corrupt: " + sentence + " </s>"
    text = ["corrupt: " + str(i) + " </s>" for i in sentence]

    input_ids = []
    for i in text:
        encoding = tokenizer.encode_plus(i,
                                         max_length=experiment_configuration.max_input_length,
                                         truncation=True,
                                         pad_to_max_length=True)
        input_ids.append([encoding["input_ids"]])

    if not input_ids:
        return "empty"

    tensor_input_ids = torch.LongTensor(input_ids).squeeze(1).to(device)

    with torch.no_grad():
        beam_outputs = model.generate(
            input_ids=tensor_input_ids,
            do_sample=True,
            max_length=experiment_configuration.max_input_length,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1)

    result = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--model_path", type=str, default='t5_corrupt')
    parser.add_argument("--type", type=str, default='cls')
    hp = parser.parse_args()

    # t5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(hp.model_path)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # generate()
