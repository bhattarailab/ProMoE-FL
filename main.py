import os
import random
import numpy as np
import torch

import argparse

from src.algorithms.FedAvgIn import FedAvgInProMoE

parser = argparse.ArgumentParser(description='ProMoE-FL')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #To let the cuDNN use the same convolution every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_wandb(args):
    import wandb
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    name = f"{str(args.name)}"

    wandb.init(
        project="promoe-fl",
        name = name,
        resume = None,
        config=args
    )

    return wandb

def args():
    parser.add_argument('--name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='./experiments/',
                        help='Locations to save different experimental runs.')
    parser.add_argument('--server_config_path', type=str, default='src/configs/server_configs.yaml',
                        help='Location for server configs')
    parser.add_argument('--client_config_path', type=str, default='src/configs/client_configs.yaml',
                        help='Location for client configs')
    parser.add_argument('--comm_rounds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', type=str, default='promoefl', choices=['promoefl'],
                        help='Choice of Federated Averages')
    parser.add_argument('--num_clients', type=int, default=10,
                        help='total number of multimodal clients')
    parser.add_argument('--img_clients', type=int, default=10,
                        help='total number of image clients')
    parser.add_argument('--txt_clients', type=int, default=10,
                        help='total number of text clients')
    parser.add_argument('--with_server', action="store_true", default=False)
    parser.add_argument('--hetero', action="store_true", default=False)
    parser.add_argument('--wandb', action="store_true", default=False)


args()
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    if args.wandb:
        wandb = init_wandb(args)
    else:
        wandb=False

    if args.algorithm == 'promoefl':
        engine = FedAvgInProMoE(args, wandb)
        engine.run()
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} is not implemented.")