import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import EncoderResNet
from .language_encoder import EncoderBert

class MultiModalClassifier(nn.Module):
    def __init__(self, config):
        super(MultiModalClassifier, self).__init__()
        self.image_encoder = EncoderResNet(embed_dim=config.embed_dim, cnn_type=config.cnn_type, proj_dim=config.proj_dim)
        self.text_encoder = EncoderBert(config.embed_dim, txt_type=config.txt_type, proj_dim=config.proj_dim)
        
        self.fc = nn.Linear(2 *config.embed_dim, 14)

        self.img_prototypes = nn.Parameter(torch.randn(14, config.proj_dim))
        self.txt_prototypes = nn.Parameter(torch.randn(14, config.proj_dim))
        nn.init.orthogonal_(self.img_prototypes)
        nn.init.orthogonal_(self.txt_prototypes)

    def forward(self, tokenizer=None, img=None, txt=None, labels=None, training=True):
        outputs = {}

        if img is not None:
            out_img = self.image_encoder(img)
            outputs['embed_img'] = out_img['embedding']

        if txt is not None:
            out_txt = self.text_encoder(tokenizer, txt)
            outputs['embed_txt'] = out_txt['embedding']
        
        concat_feat = torch.cat([out_img["features"], out_txt["features"]], dim=1)
        logits = self.fc(concat_feat)
        outputs['logits'] = logits
        outputs['preproj_caption_features'] = out_txt["features"]
        outputs['preproj_image_features'] = out_img["features"]
        return outputs

class ImageClassifierProMoE(nn.Module):
    def __init__(self, config):
        super(ImageClassifierProMoE, self).__init__()
        self.image_encoder = EncoderResNet(embed_dim=config.embed_dim, cnn_type=config.cnn_type, proj_dim=config.proj_dim)
        self.fc = nn.Linear(2 * config.embed_dim, 14)

        self.img_prototypes = nn.Parameter(torch.randn(14, config.proj_dim))
        nn.init.orthogonal_(self.img_prototypes)

    def forward(self, img, feature_gen=None, proto=None, label=None, present_proto=None):
        embed_img = self.image_encoder(img)

        if feature_gen is None:
            embed_text = torch.zeros_like(embed_img["features"])
            # embed_text = torch.rand_like(embed_img["features"])
            embed_text = F.normalize(embed_text, p=2, dim=-1)
        else:
            send_proto = proto.to(embed_img["features"])
            with torch.no_grad():
                embed_text = feature_gen(feat= embed_img["features"], proto=send_proto, label=label, _from=0, _to=1)[0]
        concat_embed  = torch.cat((embed_img["features"], embed_text), dim=1)
        out = self.fc(concat_embed)
        return {
            "logits":out,
            "image_features": embed_img["embedding"],
            "preproj_image_features": embed_img['features'],
            "caption_features":embed_text
        }
    

class TextClassifierProMoE(nn.Module):
    def __init__(self, config):
        super(TextClassifierProMoE, self).__init__()
        self.text_encoder = EncoderBert(config.embed_dim, txt_type=config.txt_type, proj_dim=config.proj_dim)
        self.fc = nn.Linear(2 * config.embed_dim, 14)

        self.txt_prototypes = nn.Parameter(torch.randn(14, config.proj_dim))
        nn.init.orthogonal_(self.txt_prototypes)

    def forward(self, tokenizer, txt, feature_gen=None, proto=None, label=None, present_proto=None):
        embed_text = self.text_encoder(tokenizer, txt)
        if feature_gen is None:
            embed_img = torch.zeros_like(embed_text["features"])
            # embed_img = torch.rand_like(embed_text["features"])
            embed_img = F.normalize(embed_img, p=2, dim=-1)
        else:
            send_proto = proto.to(embed_text["features"])
            with torch.no_grad():
                embed_img = feature_gen(feat=embed_text["features"],proto=send_proto ,label=label, _from=1, _to=0)[0]
        concat_embed  = torch.cat((embed_img, embed_text["features"]), dim=1)
        out = self.fc(concat_embed)
        return {
            "logits":out,
            "caption_features":embed_text["embedding"],
            "preproj_caption_features": embed_text["features"],
            "image_features": embed_img
        }
