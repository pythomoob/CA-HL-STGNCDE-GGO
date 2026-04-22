import os
import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time

from BasicTrainer_cde import Trainer
from lib.TrainInits import init_seed, print_model_parameters
from lib.dataloader import get_dataloader_cde
from lib.metrics import MAE_torch
from Make_model import make_model
from os.path import join
from torch.utils.tensorboard import SummaryWriter

# **************************************************************** #
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

    return parser.parse_args()

# **************************************************************** #
if __name__ == '__main__':
    args = get_args()

    init_seed(args.seed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    save_name = time.strftime("%m-%d-%Hh%Mm") + args.comment + "_" + args.dataset + "_" + args.model + "_" + args.model_type + "_" + \
                f"embed{{{args.embed_dim}}}hid{{{args.hid_dim}}}hidhid{{{args.hid_hid_dim}}}lyrs{{{args.num_layers}}}lr{{{args.lr_init}}}wd{{{args.weight_decay}}}"
    path = args.log_dir
    log_dir = join(path, args.dataset, save_name)
    args.log_dir = log_dir

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    w = SummaryWriter(log_dir) if args.tensorboard else None

    if args.model_type == 'type1':
        model, vector_field_f, vector_field_g = make_model(args)
    elif args.model_type == 'type1_temporal':
        model, vector_field_f = make_model(args)
        vector_field_g = None
    elif args.model_type == 'type1_spatial':
        model, vector_field_g = make_model(args)
        vector_field_f = None
    else:
        raise ValueError("Check args.model_type")

    model = model.to(device)
    if vector_field_f is not None:
        vector_field_f = vector_field_f.to(device)
    if vector_field_g is not None:
        vector_field_g = vector_field_g.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    train_loader, val_loader, test_loader, scaler, times = get_dataloader_cde(
        args, normalizer=args.normalizer, tod=args.tod, dow=False, weather=False, single=False
    )

    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = nn.L1Loss().to(device)
    elif args.loss_func == 'mse':
        loss = nn.MSELoss().to(device)
    elif args.loss_func == 'huber_loss':
        loss = nn.HuberLoss(delta=1.0).to(device)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in args.lr_decay_step.split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)

    start_epoch = 1
    loaded_best_loss = float('inf')

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Resuming training from checkpoint: {args.resume_path}")
        try:
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler and checkpoint.get('lr_scheduler_state_dict'):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            loaded_best_loss = checkpoint['best_loss']
            print(f"Resuming from Epoch {start_epoch} with best loss {loaded_best_loss:.6f}")
        except Exception as e:
            print(f"Could not load checkpoint. Starting from scratch. Error: {e}")

    trainer = Trainer(model, vector_field_f, vector_field_g, loss, optimizer,
                      train_loader, val_loader, test_loader, scaler,
                      args, lr_scheduler, device, times, w)
    trainer.best_loss = loaded_best_loss

    if args.mode == 'train':
        trainer.train(start_epoch=start_epoch)
    elif args.mode == 'test':
        model.load_state_dict(torch.load(f'./pre-trained/{args.dataset}.pth'))
        print("Load saved model")
        trainer.test(model, args, test_loader, scaler, trainer.logger, times)
    else:
        raise ValueError("Unknown mode")