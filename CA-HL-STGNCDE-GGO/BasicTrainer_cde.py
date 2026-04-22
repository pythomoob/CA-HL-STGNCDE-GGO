import torch
import torch.nn.functional as F
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.TrainInits import print_model_parameters
import pandas as pd

class Trainer(object):
    def __init__(self, model, vector_field_f, vector_field_g, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler, device, times,
                 w):
        super(Trainer, self).__init__()
        self.model = model
        self.vector_field_f = vector_field_f
        self.vector_field_g = vector_field_g
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        total_param = print_model_parameters(model, only_num=False)
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)
        self.logger.info(self.model)
        self.logger.info("Total params: {}".format(str(total_param)))
        self.device = device
        self.times = times.to(self.device, dtype=torch.float)
        self.w = w
        self.best_loss = float('inf')

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
            # for iter, batch in enumerate(val_dataloader):
                batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
                *valid_coeffs, target = batch
                # data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(self.times, valid_coeffs)
                # if self.args.real_value:
                     # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        if self.args.tensorboard:
            self.w.add_scalar(f'valid/loss', val_loss, epoch)
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        # for batch_idx, (data, target) in enumerate(self.train_loader):
        # for batch_idx, (data, target) in enumerate(self.train_loader):
        for batch_idx, batch in enumerate(self.train_loader):
            batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
            *train_coeffs, target = batch
            # data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()
            output = self.model(self.times, train_coeffs)
            loss = self.loss(output.cuda(), label)

            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        if self.args.tensorboard:
            self.w.add_scalar(f'train/loss', train_epoch_loss, epoch)

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def save_adaptive_adj_matrix(self, file_path):
        self.logger.info("Calculating and saving the adaptive adjacency matrix...")
        self.vector_field_g.eval()
        with torch.no_grad():
            node_embeddings = self.vector_field_g.node_embeddings
            adj_matrix = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
            adj_matrix_np = adj_matrix.detach().cpu().numpy()
            df = pd.DataFrame(adj_matrix_np)
            df.to_excel(file_path, index=False, header=False)

            self.logger.info(f"Adaptive adjacency matrix successfully saved to {file_path}")

    def train(self, start_epoch=1):
        best_model = None
        best_loss = self.best_loss
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        epoch_records = []
        start_time = time.time()

        for epoch in range(start_epoch, self.args.epochs + 1):
            epoch_start_time = time.time()

            train_epoch_loss = self.train_epoch(epoch)

            if self.val_loader is None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} finished in {epoch_time:.2f} seconds")

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            epoch_records.append({
                'Epoch': epoch,
                'Train_Loss': train_epoch_loss,
                'Val_Loss': val_epoch_loss
            })

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                self.best_loss = best_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn't improve for {} epochs. Training stops.".format(
                        self.args.early_stop_patience))
                    break

            if best_state:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                checkpoint_path = os.path.join(self.args.log_dir, 'latest_checkpoint.pth')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'best_loss': best_loss,
                }
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Saved latest checkpoint to {checkpoint_path}")
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
        loss_df = pd.DataFrame(epoch_records)
        excel_path = f''
        loss_df.to_excel(excel_path, index=False)
        self.logger.info(f"Epoch losses saved to {excel_path}")

        # 加载最佳模型并测试
        self.model.load_state_dict(best_model)
        adj_matrix_path = os.path.join(r'')
        self.save_adaptive_adj_matrix(adj_matrix_path)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, None, self.times)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path, times):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        try:
            feat_layer = model.feature_attention
            feat_layer.record_weights = True
            feat_layer.weight_history = []
            logger.info("Input Feature Attention recording started.")
        except AttributeError:
            feat_layer = None
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
                *test_coeffs, target = batch
                label = target[..., :args.output_dim]
                output = model(times.to(args.device, dtype=torch.float), test_coeffs)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        if feat_layer is not None and len(feat_layer.weight_history) > 0:
            logger.info("Exporting Input Feature Attention weights...")
            feat_weights_all = np.concatenate(feat_layer.weight_history, axis=0)

            feat_weights_avg = np.mean(feat_weights_all, axis=0)

            df_feat_avg = pd.DataFrame(feat_weights_avg.reshape(1, -1),
                                       columns=[f'Feature_{i}' for i in range(len(feat_weights_avg))])

            sample_size = min(1000, feat_weights_all.shape[0])
            df_feat_raw = pd.DataFrame(feat_weights_all[:sample_size],
                                       columns=[f'Feature_{i}' for i in range(len(feat_weights_avg))])

            save_path_feat = os.path.join(r'')
            with pd.ExcelWriter(save_path_feat) as writer:
                df_feat_avg.to_excel(writer, sheet_name='Global_Average', index=False)
                df_feat_raw.to_excel(writer, sheet_name='Raw_Samples', index=False)

            logger.info(f"Feature Attention weights saved to {save_path_feat}")

            # 关闭开关
            feat_layer.record_weights = False
            feat_layer.weight_history = []
        np.save(args.log_dir + '/{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save(args.log_dir + '/{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())

        # Calculate metrics for each horizon
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, r2 = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                 args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, R^2: {:.4f}".format(
                t + 1, mae, rmse, mape, r2))

        # Calculate average metrics
        mae, rmse, mape, _, r2 = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, R^2: {:.4f}".format(
            mae, rmse, mape, r2))
        # Save predictions and true values for each step to CSV
        for t in range(y_true.shape[1]):
            step_pred = y_pred[:, t, ...].cpu().numpy().reshape(-1, y_pred.shape[-2])
            step_true = y_true[:, t, ...].cpu().numpy().reshape(-1, y_true.shape[-2])
            df_pred = pd.DataFrame(step_pred, columns=[f'Node_{i + 1}_Pred' for i in range(step_pred.shape[1])])
            df_true = pd.DataFrame(step_true, columns=[f'Node_{i + 1}_True' for i in range(step_true.shape[1])])
            df = pd.concat([df_pred, df_true], axis=1)
            csv_path = f''
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved step {t + 1} predictions and true values to {csv_path}")

    @staticmethod
    def test_simple(model, args, data_loader, scaler, logger, path, times):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
            # for batch_idx, (data, target) in enumerate(data_loader):
                batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
                *test_coeffs, target = batch
                # data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(times.to(args.device, dtype=torch.float), test_coeffs)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return k / (k + math.exp(global_step / k))

def _add_weight_regularisation(total_loss, regularise_parameters, scaling=0.03):
    for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
    return total_loss