

import sys
import os
import pickle

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))

if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.config import parse_config
from datasets.mimic import MimicMultiModal
from networks import get_mmclf, get_tokenizer, get_clf, get_featuregen
from networks.optimizers import get_optimizer
from losses import get_criterion
from losses.centroid import CentroidAlignmentLoss

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC



class ClassificationTrainer:
    def __init__(self, args, client_id, config_path, wandb=False, dset=None):
        self.args = args
        self.client_id = client_id
        self.wandb = wandb
        self.config = parse_config(config_path)
        if dset is None:
            self.dset = self.config.dataset_mimic
            self.dset_name = self.config.dataset_mimic.dset_name

            if self.args.hetero:
                self.dset = self.config.dataset_iu
                self.dset_name = self.config.dataset_iu.dset_name
        else:
            self.dset = dset
            self.dset_name = dset.dset_name

        self.load_data()
        self.load_model()

        self.val_track = []

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })
        self.local_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, f"client_{self.client_id}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def load_data(self, use_public=False):

        partition_path = f'partitions/{self.dset_name}_{self.dset.view}_{self.dset.partition}.pkl'
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)


        train_set = MimicMultiModal(self.dset.img_path, self.dset.ann_path, self.dset.view, "train", dset=self.dset_name)
        client_partition = data_partition["client"]
        train_idx = client_partition[self.client_id]["train"]
        self.local_train_idx = train_idx
        if use_public:
            public_train = data_partition["server"]
            train_idx += public_train

        val_idx = client_partition[self.client_id]["val"]
        self.train_set = Subset(train_set, train_idx)
        self.val_set = Subset(train_set, val_idx)

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=False, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        print("------------------------------Data Loaded Successfully-------------------------")


    def load_model(self):
        self.model = get_mmclf(config=self.config.model)
        self.criterion = get_criterion(self.config.criterion.name, self.config.criterion)
        self.optimizer = get_optimizer(self.config.optimizer.name, self.model.parameters(), self.config.optimizer)
        self.tokenizer = get_tokenizer(self.config.model)
        self.grad_scaler =  torch.cuda.amp.GradScaler()
        print("------------------------------Model Loaded Successfully-------------------------")

    def val(self):
        self.model.eval()
        self.model.cuda()
        print('Validating Model:')
        with tqdm(self.val_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"{self.client_id} client Val AUC at {int(self.local_epoch%self.config.train.local_epoch)} : {metrics['AUC']}")
        self.wandb.log({f"client{self.client_id}/valAUC":metrics['AUC'].item()}, step=int(self.local_epoch%self.config.train.local_epoch))
        self.evaluator.reset()
        return metrics['AUC'].item()

    def run(self, comms):
        self.model.cuda()
        for i in range(self.config.train.local_epoch):
            print(f"Client_id:{self.client_id}  local_epoch:{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch()
            self.local_epoch +=1
        self.val()
        self.model.cpu()
        import gc
        gc.collect()

    def train_epoch(self):
        self.model.train()
        print("Training Model:")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)

                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())


class UnimodalClassificationTrainer:
    def __init__(self, args, client_id, config_path, wandb, modality, dset=None):
        self.args = args
        self.client_id = client_id
        self.wandb = wandb
        self.config = parse_config(config_path)

        if dset is None:
            self.dset = self.config.dataset_mimic
            self.dset_name = self.config.dataset_mimic.dset_name

            if self.args.hetero:
                self.dset = self.config.dataset_chexpert
                self.dset_name = self.config.dataset_chexpert.dset_name
        else:
            self.dset = dset
            self.dset_name = dset.dset_name

        self.modality = modality
        self.load_data()
        self.load_model()

        self.val_track = []

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })
        self.local_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, f"{self.modality}_client_{self.client_id}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def load_data(self, use_public=False):

        partition_path = f'partitions/{self.dset_name}_{self.dset.view}_{self.dset.partition}.pkl'
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)

        train_set = MimicMultiModal(self.dset.img_path, self.dset.ann_path, self.dset.view, "train", dset=self.dset_name)
        client_partition = data_partition["client"]
        train_idx = client_partition[self.client_id]["train"]
        if use_public:
            public_train = data_partition["server"]
            train_idx += public_train
        val_idx = client_partition[self.client_id]["val"]
        self.train_set = Subset(train_set, train_idx)
        self.val_set = Subset(train_set, val_idx)

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=False, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        print("------------------------------Data Loaded Successfully-------------------------")


    def load_model(self):
        self.model = get_clf(config=self.config.model, modality=self.modality)
        self.criterion = get_criterion(self.config.criterion.name, self.config.criterion)
        self.optimizer = get_optimizer(self.config.optimizer.name, self.model.parameters(), self.config.optimizer)
        self.tokenizer = get_tokenizer(self.config.model)
        self.grad_scaler =  torch.cuda.amp.GradScaler()
        print("------------------------------Model Loaded Successfully-------------------------")

    def run(self, comms):
        self.model.cuda()
        for i in range(self.config.train.local_epoch):
            print(f"Client_id:{self.client_id}  local_epoch:{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch()
            self.local_epoch +=1
        self.model.cpu()
        import gc
        gc.collect()

    def train_epoch(self):
        self.model.train()
        print("Training Model:")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if self.modality == 'text':
                        output = self.model(self.tokenizer, text)
                    elif self.modality == 'image':
                        output = self.model(images)
                    loss = self.criterion(output["logits"], label)

                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())


class ProMoEClassificationTrainer(ClassificationTrainer):
    def __init__(self, args, client_id, config_path, wandb, dset=None):
        super(ProMoEClassificationTrainer, self).__init__(args, client_id, config_path, wandb, dset)
        self.save_dir = os.path.join(self.args.exp_dir, f"client_{self.client_id}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.img_structure = None
        self.txt_structure = None
        self.global_metadata = None
        self.global_img_proto = None
        self.global_txt_proto = None

        self.lambda_align = self.config.train.lambda_align

        self.contrastive_loss_fn = CentroidAlignmentLoss()

        self.setup_featuregen()

    def setup_featuregen(self):
        self.featuregen = get_featuregen(self.config.featuregen)
        self.optimizer_featuregen = get_optimizer(self.config.optimizer.name, self.featuregen.parameters(), self.config.optimizer)
        self.grad_scaler_featuregen =  torch.cuda.amp.GradScaler()
        self.loss_gen = torch.nn.MSELoss()
        self.feature_gen_epoch = 0

    def train_featuregen(self):
        self.featuregen.train()
        self.model.eval()
        total_loss = 0
        counter = 0
        total_img_gen_loss = 0
        total_txt_gen_loss = 0
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer_featuregen.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                    img_feature = output["preproj_image_features"]
                    text_feature = output["preproj_caption_features"]
                    proto_img = F.normalize(self.global_img_proto, p=2, dim=1).detach().cuda()
                    proto_txt = F.normalize(self.global_txt_proto, p=2, dim=1).detach().cuda()

                label = label.to(img_feature.dtype)
                proto_img = proto_img.to(img_feature.dtype)
                proto_txt = proto_txt.to(img_feature.dtype)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    txt_feat, txt_logits, _ = self.featuregen(feat=img_feature, proto=proto_txt, label=label, _from=0, _to=1)
                    text_gen_loss = self.loss_gen(txt_feat, text_feature)

                    img_feat, img_logits, _ = self.featuregen(feat=text_feature, proto=proto_img, label=label, _from=1, _to=0)
                    img_gen_loss = self.loss_gen(img_feat, img_feature)

                    loss = img_gen_loss + text_gen_loss

                self.grad_scaler_featuregen.scale(loss).backward()
                if self.config.train.grad_clip > 0:
                    self.grad_scaler_featuregen.unscale_(self.optimizer_featuregen)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.featuregen.parameters(),
                                                   self.config.train.grad_clip)
                self.grad_scaler_featuregen.step(self.optimizer_featuregen)
                self.grad_scaler_featuregen.update()

                loss_val = text_gen_loss.item() + img_gen_loss.item()
                total_loss += loss_val
                total_img_gen_loss += img_gen_loss.item()
                total_txt_gen_loss += text_gen_loss.item()
                counter +=1
                tepoch.set_postfix(Loss=loss_val)
            print(f"Average Total Loss: {total_loss/counter}")
            print(f"Average Text Feat Gen Loss: {total_txt_gen_loss/counter}")
            print(f"Average Image Feat Gen Loss: {total_img_gen_loss/counter}")

    def train_epoch(self):
        self.model.train()
        print("Training Model:")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text, labels=label)

                    loss_cls = self.criterion(output['logits'], label)

                    img_protos = F.normalize(self.model.img_prototypes, p=2, dim=1)
                    txt_protos = F.normalize(self.model.txt_prototypes, p=2, dim=1)

                    # Embeddings (Batch)
                    img_embeds = F.normalize(output['embed_img'], p=2, dim=1)
                    txt_embeds = F.normalize(output['embed_txt'], p=2, dim=1)

                    loss_proto_img = self.contrastive_loss_fn(img_embeds, label, img_protos)
                    loss_proto_txt = self.contrastive_loss_fn(txt_embeds, label, txt_protos)

                    loss = loss_cls + 5*(loss_proto_img + loss_proto_txt)
                self.grad_scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)

                self.txt_structure = self.model.txt_prototypes
                self.img_structure = self.model.img_prototypes

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    def run(self, comms):
        self.model.cuda()
        for i in range(self.config.train.local_epoch):
            print(f"Client_id:{self.client_id}  local_epoch:{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch()
            self.local_epoch +=1
        self.model.cpu()

        self.featuregen.cuda()
        self.model.cuda()
        print("Training Feature Generator:")
        for i in range(self.config.train.local_epoch):
            print(f"Client_id:{self.client_id}  local_epoch_featureGen:{self.feature_gen_epoch} communication round: {comms} round_epoch: {i}")
            self.train_featuregen()
            self.feature_gen_epoch +=1
        self.featuregen.cpu()
        self.model.cpu()

        import gc
        gc.collect()


class ProMoEUnimodalClassificationTrainer(UnimodalClassificationTrainer):
    def __init__(self, args, client_id, config_path, wandb, modality, dset=None):
        super(ProMoEUnimodalClassificationTrainer, self).__init__(args, client_id, config_path, wandb, modality, dset)
        self.save_dir = os.path.join(self.args.exp_dir, f"{self.modality}_client_{self.client_id}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.setup_featuregen()

        self.img_structure = None
        self.txt_structure = None

        self.global_metadata = None
        self.global_txt_proto = None
        self.global_img_proto = None

        self.lambda_align = self.config.train.lambda_align

        self.contrastive_loss_fn = CentroidAlignmentLoss()


    def setup_featuregen(self):
        self.featuregen = get_featuregen(self.config.featuregen)

    def train_epoch(self):
        self.model.train()
        self.featuregen.eval()
        print("Training Model:")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if self.modality == 'text':
                        protos = F.normalize(self.model.txt_prototypes, p=2, dim=1)
                        cross_protos = F.normalize(self.global_img_proto, p=2, dim=1)
                        det_proto = cross_protos.detach().cuda()
                        output = self.model(self.tokenizer, text, self.featuregen, det_proto, label)
                        embeds = F.normalize(output["caption_features"], p=2, dim=1)

                    elif self.modality == 'image':
                        protos = F.normalize(self.model.img_prototypes, p=2, dim=1)
                        cross_protos = F.normalize(self.global_txt_proto, p=2, dim=1)
                        det_proto = cross_protos.detach().cuda()
                        output = self.model(images, self.featuregen, det_proto, label)
                        embeds = F.normalize(output["image_features"], p=2, dim=1)

                    loss_proto_learning = self.contrastive_loss_fn(embeds, label, protos)
                    task_loss = self.criterion(output["logits"], label)

                    loss = task_loss + loss_proto_learning * 10

                self.grad_scaler.scale(loss).backward()

                with torch.no_grad():
                    if self.modality == 'text':
                        self.txt_structure = self.model.txt_prototypes
                    else:
                        self.img_structure = self.model.img_prototypes
                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    def run(self, comms):
        self.model.cuda()
        self.featuregen.cuda()
        for i in range(self.config.train.local_epoch):
            print(f"Client_id:{self.client_id}  local_epoch:{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch()
            self.local_epoch +=1
        self.model.cpu()
        self.featuregen.cpu()
        import gc
        gc.collect()
