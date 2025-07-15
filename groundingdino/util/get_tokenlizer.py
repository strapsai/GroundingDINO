from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type, model_path=None):
    print(f'Text_encoder_type in get_pretrained_language_model is {text_encoder_type}')
    if (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        print(f'Loading model from local path: {text_encoder_type}')
        return BertModel.from_pretrained(text_encoder_type)
    elif text_encoder_type == "bert-base-uncased":
        # return BertModel.from_pretrained(text_encoder_type)
        tokenizer = BertTokenizer.from_pretrained(text_encoder_type, do_lower_case=True, local_files_only=True)
        # model = BertModel.from_pretrained(bert_model_path)
        print(f'WARNING: Loading model from HuggingFace: {text_encoder_type}')
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
    # now it should load offline