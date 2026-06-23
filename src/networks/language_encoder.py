import sys

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer



def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)


class EncoderBert(nn.Module):
    def __init__(self, embed_dim, txt_type, proj_dim=128):
        super(EncoderBert, self).__init__()
        if txt_type == 'bert-base-uncased':
            self.txt_enc = BertModel.from_pretrained("bert-base-uncased")
            # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.linear = nn.Linear(768, embed_dim)

            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, proj_dim)
            )
        else:
            raise NotImplementedError(f"txt_type {txt_type} not implemented")


    def forward(self, tokenizer,sentences):
        inputs = tokenizer(sentences, padding="max_length", return_tensors='pt', truncation=True, max_length=128)
        for a in inputs:
            inputs[a] = inputs[a].cuda()
        out = self.txt_enc(**inputs)
        cls_feat = out['last_hidden_state'][:, 0, :]

        out_feat = self.linear(cls_feat)
        projection = self.proj(out_feat)

        # Use this for projection
        out = {'embedding': l2_normalize(projection), 'features': l2_normalize(out_feat)}  # [bsz, 768]
        return out