from transformers import BertTokenizer
from .classifier import MultiModalClassifier, ImageClassifierProMoE, TextClassifierProMoE
from .promoe import MoEProtoCondnImputator


def get_mmclf(config):
    return MultiModalClassifier(config=config)


def get_clf(config, modality):
    if modality == 'text':
        return TextClassifierProMoE(config)
    elif modality == 'image':
        return ImageClassifierProMoE(config)


def get_tokenizer(config):
    if config.txt_type == 'bert-base-uncased':
        return BertTokenizer.from_pretrained("bert-base-uncased")
    if config.txt_type == 'tiny-bert':
        return BertTokenizer.from_pretrained("huawei-noah/TinyBERT_4L_zh")

    if config.txt_type == 'bert-base-multilingual-cased':
        return BertTokenizer.from_pretrained("bert-base-multilingual-cased")


def get_featuregen(config, num_expert=2):
    return MoEProtoCondnImputator(config, num_experts=num_expert)
