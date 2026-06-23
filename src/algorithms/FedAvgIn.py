import copy
import os
import torch
import pickle
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC

from .ClientTrainers import ProMoEClassificationTrainer, ProMoEUnimodalClassificationTrainer
from .ServerTrainers import ProMoEClassificationTrainer as ProMoEClassificationServerTrainer

class FedAvgIn:
    def __init__(self, args, wandb):
        self.args = args
        self.wandb = wandb
        self.num_mm_clients = args.num_clients ## Needed from Args
        self.total_comms = args.comm_rounds ## Needed from Args
        self.num_img_clients = args.img_clients
        self.num_txt_clients = args.txt_clients
        self.num_clients = self.num_mm_clients + self.num_img_clients + self.num_txt_clients

        self.setup_clients()
        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None)
        })
        self.val_track = []



    def test(self):
        self.server.model.eval()
        self.server.model.cuda()
        with tqdm(self.server.test_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.server.model(self.server.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"AUC : {metrics['AUC']}")
        print(f"AUCperLabel : {metrics['AUCperLabel']}")
        if self.wandb:
            self.wandb.log({"Test AUC(Aggregrated)":metrics['AUC'].item()}, step=self.cur_comms)
        self.evaluator.reset()

    def dispatch(self):
        print(f"server: {self.server.dset.dset_name}")
        server_state = self.server.model.state_dict()

        print("-------------Distributing Models blueprints to training centers------------------------")
        for client in self.clients:
            print(f"{client.client_id}: {client.dset_name}")
            client.model.load_state_dict(server_state, strict=False) ## same dic_key value

    def val(self):
        self.server.model.eval()
        self.server.model.cuda()
        print('Validating Model:')
        with tqdm(self.server.val_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.server.model(self.server.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"Val AUC : {metrics['AUC']}")
        if self.wandb:
            self.wandb.log({"Val AUC(Aggregrated)":metrics['AUC'].item()}, step=self.cur_comms)
        self.evaluator.reset()
        self.server.model.cpu()
        return metrics['AUC'].item()

    def save_log(self):
        log_path = os.path.join(self.args.exp_dir, "server", "agg_val_aucs.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

class FedAvgInProMoE(FedAvgIn):
    def __init__(self, args, wandb):
        super(FedAvgInProMoE, self).__init__(args, wandb)

    def setup_clients(self):

        self.server = ProMoEClassificationServerTrainer(self.args, self.args.server_config_path, False)
        if not self.args.hetero:
            self.clients_id = self.server.config.clients_config.arr_homo
        else:
            self.clients_id = self.server.config.clients_config.arr_hetero

    
        if self.args.hetero:
            # heterogeneous setup

            # 1 iu + (1-n_mm) pd for mm
            mm_iu_clients = [ProMoEClassificationTrainer(self.args, i, self.args.client_config_path, False, self.server.config.dataset_iu) for i in range(min(self.num_mm_clients, 2))]
            mm_pc_clients = [ProMoEClassificationTrainer(self.args, i, self.args.client_config_path, False, self.server.config.dataset_padchest) for i in range(self.num_mm_clients-2)]
            mm_clients = mm_iu_clients + mm_pc_clients

            # use chexpert for image only
            img_clients = [ProMoEUnimodalClassificationTrainer(self.args, i, self.args.client_config_path, False, "image", self.server.config.dataset_chexpert) for i in range(self.num_img_clients)]
            # use padchest for text only
            txt_clients = [ProMoEUnimodalClassificationTrainer(self.args, len(mm_pc_clients) + i, self.args.client_config_path, False, "text", self.server.config.dataset_padchest) for i in range(self.num_txt_clients)]
            
        else:
            # homogeneous setup
    
            mm_clients = [ProMoEClassificationTrainer(self.args, self.clients_id[i], self.args.client_config_path, False) for i in range(self.num_mm_clients)]
            img_clients = [ProMoEUnimodalClassificationTrainer(self.args, self.clients_id[i+self.num_mm_clients], self.args.client_config_path, False, "image") for i in range(self.num_img_clients)]
            txt_clients = [ProMoEUnimodalClassificationTrainer(self.args, self.clients_id[i+self.num_mm_clients + self.num_img_clients], self.args.client_config_path, False, "text") for i in range(self.num_txt_clients)]
        self.clients = mm_clients + img_clients + txt_clients

        self.setup_metadata()


    def setup_metadata(self):

        num_station = self.num_mm_clients

        if self.args.with_server:
            num_station+=1
        
        global_txt_proto = torch.nn.Parameter(torch.randn(num_station, 14, self.server.config.model.proj_dim))
        torch.nn.init.orthogonal_(global_txt_proto)

        global_img_proto = torch.nn.Parameter(torch.randn(num_station, 14, self.server.config.model.proj_dim))
        torch.nn.init.orthogonal_(global_img_proto)

        if self.args.with_server:
            station_list = self.clients + [self.server]
        else:
            station_list = self.clients

        for station in station_list:
            station.global_img_proto = global_img_proto
            station.global_txt_proto = global_txt_proto

        
    def aggregate_and_broadcast_G(self):

        if self.args.with_server:
            stations = self.clients[:self.num_mm_clients] + [self.server]
            
        else:
            stations = self.clients[:self.num_mm_clients]


        img_stations = stations + self.clients[self.num_mm_clients:self.num_img_clients + self.num_mm_clients]
        txt_stations = stations + self.clients[self.num_mm_clients+self.num_img_clients:]

        img_proto = [station.img_structure for station in img_stations]
        txt_proto = [station.txt_structure for station in txt_stations]
        # print(f"Img Stations: {len(img_stations)} Text Stations: {len(txt_stations)}")

        stacked_txt_proto = torch.stack(txt_proto)
        stacked_img_proto = torch.stack(img_proto)


        # broadcast to every clients
        if self.args.with_server:
            stations = self.clients + [self.server]
        else:
            stations = self.clients 

        for station in stations:
            station.global_txt_proto = stacked_txt_proto
            station.global_img_proto = stacked_img_proto


    def aggregrate(self):
        print("Aggregrating Models")
        global_dict = copy.deepcopy(self.server.model.state_dict())
        if self.args.with_server:
            station_list = self.clients + [self.server]
        else:
            station_list = self.clients
        for k in global_dict.keys():
            if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids", "prototype"]):
                continue
            else:
                params = []
                weights = []
                for station in station_list:
                    para = station.model.state_dict().get(k)
                    if para is not None:
                        params.append(para)
                        weights.append(len(station.train_set))
                    total_weights = sum(weights)
                    normalized_weights = [w / total_weights for w in weights]
                weighted_params = [nw * para for nw, para in zip(normalized_weights, params)]
                weighted_sum = torch.sum(torch.stack(weighted_params, 0), dim=0)
                global_dict[k] = weighted_sum
                
        self.server.model.load_state_dict(global_dict)

        print("Aggregrating Feature Generator")
        global_gen_dict = copy.deepcopy(self.server.featuregen.state_dict())
        if self.args.with_server:
            station_list = self.clients[:self.num_mm_clients] + [self.server]
        else:
            station_list = self.clients[:self.num_mm_clients]

        for k in  global_gen_dict.keys():
            params = []
            weights = []
            for station in station_list:
                para = station.featuregen.state_dict().get(k)
                if para is not None:
                    params.append(para)
                    weights.append(len(station.train_set))
                total_weights = sum(weights)
                normalized_weights = [w / total_weights for w in weights]
            weighted_params = [nw * para for nw, para in zip(normalized_weights, params)]
            weighted_sum = torch.sum(torch.stack(weighted_params, 0), dim=0)
            global_gen_dict[k] = weighted_sum
        self.server.featuregen.load_state_dict(global_gen_dict)

    def dispatch(self):
        super().dispatch()
        print("-------------Distributing Feature Generator blueprint to training centers------------------------")
        for client in self.clients:
            client.featuregen.load_state_dict(self.server.featuregen.state_dict())

    def save_proto(self):
        save_pth_proto_bank = os.path.join(self.server.save_dir, f"global_proto{self.cur_comms}.pth")

        if self.args.with_server:
            stations = self.clients[:self.num_mm_clients] + [self.server]
            
        else:
            stations = self.clients[:self.num_mm_clients]
            
        img_stations = stations + self.clients[self.num_mm_clients:self.num_mm_clients + self.num_img_clients]
        txt_stations = stations + self.clients[self.num_mm_clients+self.num_img_clients:]

        img_proto = [station.img_structure for station in img_stations]
        txt_proto = [station.txt_structure for station in txt_stations]
        print(f"Img Stations: {len(img_stations)} Text Stations: {len(txt_stations)}")


        
        stacked_txt_proto = torch.stack(txt_proto)
        stacked_img_proto = torch.stack(img_proto)

        torch.save(
            {
                "txt_proto": stacked_txt_proto.cpu(),
                "img_proto": stacked_img_proto.cpu(),
            },
            save_pth_proto_bank
        )

    def save_log(self):
        super().save_log()
        ckpt_path = os.path.join(self.server.save_dir, f"feat_gen{self.cur_comms}.pth")
        torch.save({"featuregen":self.server.featuregen.state_dict(), "comms":self.cur_comms}, ckpt_path)

    def run(self):
        self.cur_comms = 0
        self.val_auc = 0
        
        for comms in range(self.total_comms):
            print(f"-----------------------Communication Round: {comms}-------------------------------")
            self.dispatch()
            if self.args.with_server:
                print(f"-----------------------Training Server Model in Server Data------------------------------")
                self.server.run(comms)
            print(f"-----------------------Training Client Models in Clients Data------------------------------")
            for client in self.clients:
                client.run(comms)
            print(f"-----------Performing global aggregration--------------------------")
            self.aggregrate()
            self.aggregate_and_broadcast_G()
            print("---------------Evaluating Aggregrated Model in Val Set-------------------------------")
            cur_auc = self.val()
            self.val_track.append(cur_auc)
            if cur_auc > self.val_auc:
                self.val_auc = cur_auc
                self.server.save_best(self.cur_comms)
                self.save_proto()
            self.cur_comms +=1
            self.save_log()
            import gc
            gc.collect()
        self.server.load_best()
        self.test()

