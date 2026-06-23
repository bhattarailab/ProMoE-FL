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
from datasets.iu_xray import IUXrayMultiModal
from networks import get_mmclf, get_tokenizer, get_featuregen
from networks.optimizers import get_optimizer
from losses import get_criterion
from losses.centroid import CentroidAlignmentLoss

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC



class ClassificationTrainer:
    def __init__(self, args, config_path, wandb=False, dset_name="mimic-cxr"):
        self.args = args
        self.wandb = wandb

        self.config = parse_config(config_path)

        if dset_name == "mimic-cxr":
            self.dset_name = self.config.dataset_mimic.dset_name
            self.dset = self.config.dataset_mimic
        elif dset_name == "iuxray":
            self.dset_name = self.config.dataset_iuxray.dset_name
            self.dset = self.config.dataset_iuxray
        elif dset_name == "padchest":
            self.dset_name = self.config.dataset_padchest.dset_name
            self.dset = self.config.dataset_padchest

        self.load_data()
        self.load_model()

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })
        self.cur_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, "server")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.val_track = []

    def load_data(self):
        if self.dset_name == "mimic-cxr":
            partition_path = f'partitions/{self.dset_name}_{self.dset.view}_{self.dset.partition}.pkl'
            with open(partition_path, "rb") as f:
                data_partition = pickle.load(f)
            train_set = MimicMultiModal(self.dset.img_path, self.dset.ann_path, self.dset.view, "train")
            train_idx = data_partition["server"]
            self.train_set = Subset(train_set, train_idx)
        elif self.dset_name == "iuxray":
            partition_path = f'partitions/{self.dset_name}_{self.dset.view}_{self.dset.partition}.pkl'
            with open(partition_path, "rb") as f:
                data_partition = pickle.load(f)
            train_set = IUXrayMultiModal(self.dset.img_path, self.dset.ann_path, self.dset.view, "train")
            train_idx = data_partition["client"][0]['train']
            self.train_set = Subset(train_set, train_idx)
        elif self.dset_name == "padchest":
            partition_path = f'partitions/{self.dset_name}_{self.dset.view}_{self.dset.partition}.pkl'
            with open(partition_path, "rb") as f:
                data_partition = pickle.load(f)
            train_set = MimicMultiModal(self.dset.img_path, self.dset.ann_path, self.dset.view, "train", dset="padchest")
            train_idx = data_partition["server"]
            self.train_set = Subset(train_set, train_idx)

        self.val_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "val")
        self.test_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "test")

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        print("------------------------------Data Loaded Successfully-------------------------")

    def load_model(self):
        self.model = get_mmclf(config=self.config.model)
        self.tokenizer = get_tokenizer(config=self.config.model)
        self.criterion = get_criterion(self.config.criterion.name, self.config.criterion)
        self.optimizer = get_optimizer(self.config.optimizer.name, self.model.parameters(), self.config.optimizer)
        self.grad_scaler =  torch.cuda.amp.GradScaler()
        print("------------------------------Model Loaded Successfully-------------------------")

    def save_best(self, comms):
        ckpt_path = os.path.join(self.save_dir, f"model_best.pth")
        torch.save({"net":self.model.state_dict(), "comms":comms}, ckpt_path)

    def load_best(self):
        ckpt_path = os.path.join(self.save_dir, f"model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["net"])
        print(f"Best Model is at comms : {checkpoint['comms']}")

    def save_log(self):
        log_path = os.path.join(self.save_dir, "val_aucs.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def run(self, comms):
        self.model.cuda()
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        for i in range(self.config.train.local_epoch):
            print(f"Server:-  Comm round{comms} local_epoch:{self.cur_epoch}  round_epoch: {i}")
            self.train_epoch()
            self.cur_epoch +=1
            print("------------------------------------------------------------")
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

    def val(self):
        self.model.eval()
        print('Validating Model:')
        with tqdm(self.val_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"Val AUC : {metrics['AUC']}")
        if self.wandb:
            self.wandb.log({"Val AUC(Server)":metrics['AUC'].item()}, step=self.cur_epoch)
        self.evaluator.reset()
        return metrics['AUC'].item()


class ProMoEClassificationTrainer(ClassificationTrainer):
    def __init__(self, args, config_path, wandb, dset='mimic-cxr'):
        super(ProMoEClassificationTrainer, self).__init__(args, config_path, wandb, dset)
        self.save_dir = os.path.join(self.args.exp_dir, f"server_featuregen")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.global_metadata = None

        self.img_structure = None
        self.txt_structure = None
        self.lambda_align = self.config.train.lambda_align

        self.global_img_proto = None
        self.global_txt_proto = None
        self.client_id = -1
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

                with torch.no_grad():
                    # (14,128) x (128, 14) -> (14,14), correlation matrix
                    img_structure = torch.matmul(self.model.img_prototypes, self.model.img_prototypes.T)
                    txt_structure = torch.matmul(self.model.txt_prototypes, self.model.txt_prototypes.T)

                    self.global_metadata = (img_structure.detach() + txt_structure.detach()) * 0.5
                    self.txt_structure = self.model.txt_prototypes
                    self.img_structure = self.model.img_prototypes

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    def run(self, comms):
        self.model.cuda()
        for i in range(self.config.train.local_epoch):
            print(f"Server:  local_epoch:{self.cur_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch()
            self.cur_epoch +=1
        self.model.cpu()

        self.featuregen.cuda()
        self.model.cuda()
        print("Training Feature Generator:")
        for i in range(self.config.train.local_epoch):
            print(f"Server:  local_epoch_featureGen:{self.feature_gen_epoch} communication round: {comms} round_epoch: {i}")
            self.train_featuregen()
            self.feature_gen_epoch +=1
        self.featuregen.cpu()
        self.model.cpu()

        import gc
        gc.collect()
