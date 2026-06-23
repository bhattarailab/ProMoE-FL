import torch
import torch.nn as nn
import torch.nn.functional as F

class DeterministicProtoCondExpert(nn.Module):
    def __init__(self, config):
        super(DeterministicProtoCondExpert, self).__init__()
        self.mem_size = config.memorysize
        self.feat_dim = config.feature_dim
        self.num_class = config.num_class

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feat_dim, 
            nhead=4, 
            dim_feedforward=1024, 
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
    def forward(self, tgt, memory, memory_key_padding_mask=None):
        decoder_output = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
        )
        reconst_feat = F.normalize(decoder_output, p=2, dim=-1)

        return reconst_feat


class ImputationRouter(nn.Module):
    def __init__(self, config, num_experts):
        super().__init__()
        self.num_modalities = config.num_modalities
        self.feat_dim = config.feature_dim
        self.modality_embeddings = nn.Embedding(self.num_modalities, self.feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(4 * self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, num_experts)
        )

    def forward(self, tgt, memory, from_modality, to_modality):
        # tgt: (bs, 1, D)
        # memory: (bs, T, D)

        tgt_summary = tgt.mean(dim=1)        # (bs, D)
        mem_summary = memory.mean(dim=1)     # (bs, D)

        from_emb = self.modality_embeddings(from_modality)  # (bs, D)
        to_emb   = self.modality_embeddings(to_modality)    # (bs, D)

        router_input = torch.cat(
            [tgt_summary, mem_summary, from_emb, to_emb],
            dim=-1
        )  # (bs, 4D)
        router_input = router_input.to(tgt.dtype)
        gate_logits = self.mlp(router_input)     # (bs, N)
        gate = torch.softmax(gate_logits, dim=-1)

        return gate


class MoEImputer(nn.Module):
    def __init__(self, config, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.feat_dim = config.feature_dim

        self.experts = nn.ModuleList(
            [DeterministicProtoCondExpert(config=config) for _ in range(self.num_experts)]
        )
        self.router = ImputationRouter(
            config=config,
            num_experts=self.num_experts
        )

    def forward(self, tgt_feat, memory, from_modality, to_modality, memory_mask=None):
        # tgt_feat: (bs, D) or (bs, 1, D)
        if tgt_feat.dim() == 2:
            tgt = tgt_feat.unsqueeze(1)
        else:
            tgt = tgt_feat

        # -------- Router --------
        gate = self.router(tgt, memory, from_modality, to_modality)  # (bs, N)

        # -------- Experts --------
        expert_outputs = []
        for expert in self.experts:
            out = expert(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=memory_mask
            )  # (bs, 1, D)
            expert_outputs.append(out.squeeze(1))

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (bs, N, D)

        # -------- Mixture --------
        output = torch.sum(
            gate.unsqueeze(-1) * expert_outputs,
            dim=1
        )  # (bs, D)

        return F.normalize(output, dim=-1), gate


class MoEProtoCondnImputator(nn.Module):
    def __init__(self, config, num_experts=2):
        super().__init__()
        self.feat_dim = config.feature_dim
        self.num_class = config.num_class


        self.moe = MoEImputer(config=config, num_experts=num_experts)
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
    
    def build_memory(self, proto, label):
        # proto: (n_mm, K, D)
        # label: (bs, K)

        bs, K = label.shape
        n_mm, _, D = proto.shape

        proto_exp = proto.unsqueeze(0).expand(bs, n_mm, K, D)
        label_mask = label.unsqueeze(1).unsqueeze(-1)  # (bs, 1, K, 1)

        memory = proto_exp * label_mask.float()
        memory = proto_exp
        memory = memory.reshape(bs, n_mm * K, D)

        memory_key_padding_mask = (memory.abs().sum(dim=-1) == 0)
        return memory, memory_key_padding_mask

    def forward(self, feat, proto, label, _from, _to):
        memory, mem_mask = self.build_memory(proto, label)
        memory = memory.to(feat.dtype)

        pred, gate = self.moe(
            tgt_feat=feat,
            memory=memory,
            from_modality=torch.full((feat.size(0),), _from, device=feat.device),
            to_modality=torch.full((feat.size(0),), _to, device=feat.device),
            memory_mask=mem_mask
        )

        out_logits = self.classifier(pred)

        return pred, out_logits, gate
