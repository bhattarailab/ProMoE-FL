#!/bin/bash --login

#SBATCH --account <acc>
#SBATCH --job-name <run_name>
#SBATCH --output /scratch/achhetr1/promoe/m2_i4_t4_%A_%a.log
#SBATCH --error /scratch/achhetr1/promoe/m2_i4_t4_%A_%a.err

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --time 12:00:00


source /users/achhetr1/miniconda3/bin/activate promoe-env


python main.py --name promoe_m2_i4_t4_homo --exp_dir /scratch/achhetr1/promoe/experiment/promoe_m2_i4_t4_homo/ckpt_1\
                --server_config_path ./src/configs/fedavgin_server.yaml --comm_rounds 30\
                --algorithm promoefl --num_clients 2 --img_clients 4 --txt_clients 4 --wandb
