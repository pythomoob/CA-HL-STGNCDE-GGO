import os
import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time
import pandas as pd
import math
import random
from tqdm import tqdm
import logging

from BasicTrainer_cde import Trainer
from lib.TrainInits import init_seed, print_model_parameters
from lib.dataloader import get_dataloader_cde
from lib.metrics import MAE_torch
from Make_model import make_model
from os.path import join
from torch.utils.tensorboard import SummaryWriter

# **************************************************************** #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Mode = 'train'
DEBUG = 'False'
DATASET = ''
MODEL = 'GCDE'
config_file = f'./model/{DATASET}_{MODEL}.conf'
config = configparser.ConfigParser()
config.read(config_file)

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)

        mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        penalty_weight = 5

        negative_penalty = torch.relu(-preds)

        if mask_value is not None:
            mask = torch.gt(labels, mask_value)
            negative_penalty = torch.masked_select(negative_penalty, mask)

        penalty_loss = negative_penalty.mean()

        total_loss = mae_loss + (penalty_weight * penalty_loss)

        return total_loss

    return loss

def get_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--debug', default=DEBUG, type=eval)
    parser.add_argument('--model', default=MODEL, type=str)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--comment', default='', type=str)

    # data
    parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    parser.add_argument('--lag', default=config['data']['lag'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--tod', default=config['data']['tod'], type=eval)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)

    # model
    parser.add_argument('--model_type', default=config['model']['type'], type=str)
    parser.add_argument('--g_type', default=config['model']['g_type'], type=str)
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--hid_dim', default=config['model']['hid_dim'], type=int)
    parser.add_argument('--hid_hid_dim', default=config['model']['hid_hid_dim'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    parser.add_argument('--solver', default='rk4', type=str)

    # train
    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--weight_decay', default=config['train']['weight_decay'], type=eval)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser.add_argument('--teacher_forcing', default=False, type=bool)
    parser.add_argument('--real_value', default=config['train']['real_value'], type=eval)

    parser.add_argument('--missing_test', default=False, type=bool)
    parser.add_argument('--missing_rate', default=0.1, type=float)

    # test
    parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    parser.add_argument('--model_path', default='', type=str)

    # log
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser.add_argument('--plot', default=config['log']['plot'], type=eval)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to the checkpoint file to resume training from.')

    parser.add_argument('--ggo_iterations', default=1, type=int, help='Number of iterations for GGO')
    parser.add_argument('--ggo_solutions', default=1, type=int, help='Number of solutions (population size) for GGO')
    parser.add_argument('--excel_path', default='./results/best_parameters.xlsx', type=str)

    return parser.parse_args()


def objective_func(X, train_loader, val_loader, scaler, times, device):

    learning_rate = 1e-5 + X[0] * (1e-2 - 1e-5)
    weight_decay = 1e-5 + X[1] * (1e-2 - 1e-5)

    model, _, _ = make_model(args)
    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.L1Loss().to(device) if args.loss_func == 'mae' else nn.MSELoss().to(device)

    try:
        model.train()
        for batch in train_loader:
            batch = tuple(b.to(device, dtype=torch.float) for b in batch)
            *train_coeffs, target = batch
            label = target[..., :args.output_dim]
            optimizer.zero_grad()
            output = model(times, train_coeffs)
            if args.real_value: label = scaler.inverse_transform(label)
            loss = loss_fn(output, label)
            loss.backward()
            if args.grad_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        model.eval()
        valid_loss = []
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(b.to(device, dtype=torch.float) for b in batch)
                *valid_coeffs, target = batch
                label = target[..., :args.output_dim]
                output = model(times, valid_coeffs)
                if args.real_value: label = scaler.inverse_transform(label)
                val_l = loss_fn(output, label)
                if not torch.isnan(val_l): valid_loss.append(val_l.item())

        avg_val_loss = np.mean(valid_loss) if valid_loss else float('inf')
        logger.info(f"GGO Eval | LR: {learning_rate:.6f}, WD: {weight_decay:.6f} | Loss: {avg_val_loss:.6f}")
    except Exception as e:
        logger.error(f"GGO Error: {e}")
        avg_val_loss = float('inf')

    return avg_val_loss, 1.0, 2, 0.0

def GGO(func, nb_solu, taille_solu, itération, *func_args):
    T = itération
    nb_solution = nb_solu
    Matrice_Solution = np.random.rand(nb_solution, taille_solu + 1)

    logger.info("Evaluating initial GGO population...")
    for i in tqdm(range(nb_solution), desc="Initial GGO Eval"):
        Matrice_Solution[i, -1], _, _, _ = func(Matrice_Solution[i, :-1], *func_args)

    idxMin = np.argmin(Matrice_Solution[:, taille_solu])
    solution_best_in_itra = np.copy(Matrice_Solution[idxMin, :])
    best_fitness_history = [solution_best_in_itra[-1]]

    n1 = max(1, int(nb_solution / 2))
    n2 = nb_solution - n1
    a = np.linspace(0, 2, num=T)

    logger.info("Starting GGO optimization iterations...")
    for t in tqdm(range(1, T + 1), desc="GGO Progress"):
        z = 1 - (t / T) ** 2
        for i in range(nb_solution):
            r1, r2, r3, r4, r5 = (random.uniform(0, 1) for _ in range(5))
            A = 2 * a[t - 1] * r1 - a[t - 1]
            C = 2 * r2
            w, w1, w2, w3, w4 = (random.uniform(0, 2) for _ in range(5))
            b = 10
            l_rand = random.uniform(-1, 1)

            current_solution = Matrice_Solution[i, :-1]
            new_solution = np.copy(current_solution)

            if i < n1:  # Leader group
                if (t % 2) == 0:
                    if r3 < 0.5:
                        if abs(A) < 1:
                            new_solution = solution_best_in_itra[:-1] - A * abs(C * solution_best_in_itra[:-1] - current_solution)
                        else:
                            if n1 > 0:
                                Sa, Sb, Sc = random.randint(0, n1 - 1), random.randint(0, n1 - 1), random.randint(0, n1 - 1)
                                new_solution = w1 * Matrice_Solution[Sa, :-1] + z * w2 * (Matrice_Solution[Sb, :-1] - Matrice_Solution[Sc, :-1]) + (1 - z) * w3 * (current_solution - Matrice_Solution[Sa, :-1])
                    else:
                        new_solution = w4 * abs(solution_best_in_itra[:-1] - current_solution) * math.exp(b * l_rand) * math.cos(2 * math.pi * l_rand) + (2 * w1 * (r4 + r5)) * solution_best_in_itra[:-1]
                else:
                    new_solution = current_solution + (1 + z) * w * (current_solution - solution_best_in_itra[:-1])
            else:  # Follower group
                if (t % 2) == 0:
                    if n2 > 0:
                        idx1, idx2, idx3 = random.randint(n1, nb_solution - 1), random.randint(n1, nb_solution - 1), random.randint(n1, nb_solution - 1)
                        S1 = Matrice_Solution[idx1, :-1] - A * abs(C * Matrice_Solution[idx1, :-1] - current_solution)
                        S2 = Matrice_Solution[idx2, :-1] - A * abs(C * Matrice_Solution[idx2, :-1] - current_solution)
                        S3 = Matrice_Solution[idx3, :-1] - A * abs(C * Matrice_Solution[idx3, :-1] - current_solution)
                        new_solution = (S1 + S2 + S3) / 3
                else:
                    new_solution = current_solution + (1 + z) * w * (current_solution - solution_best_in_itra[:-1])

            new_solution = np.clip(new_solution, 0.0, 1.0)
            new_fitness, _, _, _ = func(new_solution, *func_args)

            if new_fitness < Matrice_Solution[i, -1]:
                Matrice_Solution[i, :-1], Matrice_Solution[i, -1] = new_solution, new_fitness

        current_best_idx = np.argmin(Matrice_Solution[:, -1])
        if Matrice_Solution[current_best_idx, -1] < solution_best_in_itra[-1]:
            solution_best_in_itra = np.copy(Matrice_Solution[current_best_idx, :])

        best_fitness_history.append(solution_best_in_itra[-1])
        stagnated = (t >= 1 and best_fitness_history[t] >= best_fitness_history[t-1])
        if stagnated and n1 < nb_solution - 1:
            n1 += 1; n2 -= 1
        elif not stagnated and n1 > 1:
            n2 += 1; n1 -= 1

    return solution_best_in_itra

# **************************************************************** #
if __name__ == '__main__':
    args = get_args()
    init_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.cuda else 'cpu')

    logger.info("Loading data for the optimization process...")
    train_loader, val_loader, test_loader, scaler, times = get_dataloader_cde(
        args, normalizer=args.normalizer, tod=False, dow=False, weather=False, single=False
    )
    times = times.to(device)

    logger.info("--- Starting Hyperparameter Optimization with GGO ---")
    best_solution = GGO(objective_func, args.ggo_solutions, 2, args.ggo_iterations,
                        train_loader, val_loader, scaler, times, device)

    best_lr = 1e-5 + best_solution[0] * (1e-2 - 1e-5)
    best_wd = 1e-5 + best_solution[1] * (1e-2 - 1e-5)

    logger.info(f"Optimization Results | Best LR: {best_lr:.6f}, Best WD: {best_wd:.6f}, Loss: {best_solution[-1]:.6f}")

    os.makedirs(os.path.dirname(args.excel_path), exist_ok=True)
    pd.DataFrame({"Learning Rate": [best_lr], "Weight Decay": [best_wd], "Best Loss": [best_solution[-1]]}).to_excel(
        args.excel_path, index=False)

    # 2. 最终训练阶段
    logger.info("--- Starting Final Training with Best Hyperparameters ---")
    args.lr_init, args.weight_decay = best_lr, best_wd

    save_name = f"{time.strftime('%m%d_%H%M')}_{args.model}_lr{args.lr_init:.5f}_wd{args.weight_decay:.5f}"
    args.log_dir = join(args.log_dir, args.dataset, save_name)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    w = SummaryWriter(args.log_dir) if args.tensorboard else None
    model, vff, vfg = make_model(args)
    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    loss_func = masked_mae_loss(scaler, 0.0) if args.loss_func == 'mask_mae' else (
        nn.L1Loss().to(device) if args.loss_func == 'mae' else nn.MSELoss().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.lr_decay:
        milestones = [int(i) for i in args.lr_decay_step.split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay_rate)

    trainer = Trainer(model, vff, vfg, loss_func, optimizer, train_loader, val_loader, test_loader, scaler, args,
                      lr_scheduler, device, times, w)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test(model, args, test_loader, scaler, trainer.logger, times=times, path=None)

    logger.info("--- Process Finished ---")