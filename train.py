import os 
import random 
import argparse
import errno
import time
import copy 
from tqdm import tqdm 

from lib.models.models import DARNN 
from lib.data.data import get_data_and_preprocess
from lib.utils.learning import AverageMeter
from lib.utils.utils import get_config, mape, EarlyStopper

import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset



device = torch.device("cuda")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/basic.yaml", help="Path to the config file.")
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def save_checkpoint(chk_path, epoch, lr, optimizer, scheduler, model, min_loss):
    model.eval()
              
    print("[INFO] : saving modelcheckpoint.")
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_loss' : min_loss, 
        'scheduler': scheduler.state_dict()
    }, chk_path)



def get_dataloader(args): 
     
    data, scale = get_data_and_preprocess(args.csv_file, args.target, args.timesteps, args.train_split, args.val_split) 
    X_train_t, X_val_t, X_test_t, y_his_train_t, y_his_val_t, y_his_test_t, target_train_t, target_val_t, target_test_t,  = data
    train_loader = DataLoader(TensorDataset(X_train_t, y_his_train_t, target_train_t), shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(TensorDataset(X_val_t, y_his_val_t, target_val_t), shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test_t, y_his_test_t, target_test_t), shuffle=False, batch_size=args.batch_size)
    return train_loader, val_loader, test_loader, scale 


def get_model(args): 
    model = DARNN(
        T = args.timesteps, 
        m = args.m_dim, 
        p = args.p_dim, 
        n = args.num_exogenous
    )
    model.to(device)
    print(f"[INFO] : Running model on device : {device}")
    return model  


def get_criterion(args): 
    criterion = torch.nn.MSELoss() 
    criterion.to(device)
    return criterion 


def lr_lambda(iteration):
    return 0.9 ** (iteration // 20)


def get_optimizer(args, model): 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)    
    return optimizer, scheduler 


def train_epoch(args, opts, model, train_loader, criterion, optimizer, scheduler, losses, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y_known, target = data
        x = x.to(device)  # exogenous [X1, ..., Xt-1]
        y_known = y_known.to(device)  # observed target [y1, ..., yt-1]
        target = target.to(device)[:, None]  # future value
        preds = model(x, y_known)  # (bs, 1)
        loss = criterion(target, preds)
        losses["train_MSE_loss"].update(loss.item(), preds.shape[0])
        loss.backward()
        optimizer.step()    
    
def validate_epoch(
    args, 
    opts, 
    model, 
    val_loader, 
    criterion, 
    losses, 
    epoch, 
    mode="val", 
    scale=None
): 
    model.eval()

    if scale:
        target_train_max, target_train_min = scale
    

    pred, gt = [], []

    with torch.no_grad(): 
        for data in val_loader: 
            x, y_known, target = data  
            
            x = x.to(device) # exogenous [X1, ..., Xt-1]
            y_known = y_known.to(device) # observed target [y1, ..., yt-1]
            if scale:
                target = target.to(device)[:, None]*(target_train_max - target_train_min) + target_train_min
                preds = model(x, y_known) *(target_train_max - target_train_min) + target_train_min
            else: 
                target = target.to(device)[:, None] 
                preds = model(x, y_known) # (bs, 1)
            loss = criterion(target, preds)

            mae_loss = torch.nn.functional.l1_loss(target, preds)
            mape_loss = mape(target, preds)

            losses[f"{mode}_MSE_loss"].update(loss.item(), preds.shape[0]) 
            losses[f"{mode}_MAE_loss"].update(mae_loss.item(), preds.shape[0])
            losses[f"{mode}_RMSE_loss"].update(torch.sqrt(loss).item(), preds.shape[0])
            losses[f"{mode}_MAPE_loss"].update(mape_loss.item(), preds.shape[0])
            
            gt.append(target.cpu().numpy())
            pred.append(preds.cpu().numpy())

    gt = np.concatenate(gt)
    pred = np.concatenate(pred)[:, 0]

    return model, gt, pred



def train_with_config(args, opts): 

    experiment_path = os.path.join("./experiments")
    if not os.path.exists(experiment_path): 
        os.mkdir(experiment_path)

    opts.checkpoint = os.path.join(experiment_path, opts.checkpoint)

    try: 
        os.mkdir(opts.checkpoint)
    except OSError as e: 
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
        
    train_writer = SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    train_loader, val_loader, test_loader, scale = get_dataloader(args)
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer, scheduler = get_optimizer(args, model)

    min_loss = np.inf
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    early_stopper = EarlyStopper(patience=16, min_delta=0.05)

    for epoch in tqdm(range(args.epochs)): 
        print('Training epoch %d.' % epoch)

        losses = {}
        losses["train_MSE_loss"] = AverageMeter()
        losses["val_MSE_loss"] = AverageMeter()
        losses["val_MAE_loss"] = AverageMeter()
        losses["val_RMSE_loss"] = AverageMeter()
        losses["val_MAPE_loss"] = AverageMeter()
        losses["test_MSE_loss"] = AverageMeter()
        losses["test_MAE_loss"] = AverageMeter()
        losses["test_RMSE_loss"] = AverageMeter()
        losses["test_MAPE_loss"] = AverageMeter()
        train_epoch(args, opts, model, train_loader, criterion, optimizer, scheduler, losses, epoch) 
        model, gt, pred = validate_epoch(args, opts, model, val_loader, criterion, losses, epoch, mode="val", scale=scale)
        scheduler.step()
        
        # logs
        lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar("train_MSE_loss", losses["train_MSE_loss"].avg, epoch + 1)
        train_writer.add_scalar("val_MSE_loss", losses["val_MSE_loss"].avg, epoch + 1)
        train_writer.add_scalar("lr", lr, epoch + 1)
        for i in range(len(gt)):

            train_writer.add_scalars(
                "validation_prediction", 
                {
                    "gt": gt[i], 
                    "pred": pred[i]
                }, 
                i
            )
        chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

        if early_stopper.early_stop(losses["val_MSE_loss"].avg):             
            break

        save_checkpoint(chk_path_latest, epoch, lr, optimizer, scheduler, model, min_loss)
        if losses["val_RMSE_loss"].avg < min_loss: 
            min_loss = losses["val_RMSE_loss"].avg
            save_checkpoint(chk_path_best, epoch, lr, optimizer, scheduler, model, min_loss)

    print("[INFO] : Training done. ")
    print("[INFO] : Performing inference on test data") 
    model, gt, pred = validate_epoch(args, opts, model, test_loader, criterion, losses, epoch, mode="test", scale=scale)
    for i in range(len(gt)):
        train_writer.add_scalars(
            "test_prediction", 
            {
                "gt": gt[i], 
                "pred": pred[i]
            }, 
            i
        )

    train_writer.add_scalar("test_MSE_loss", losses["test_MSE_loss"].avg, 0)
    train_writer.add_scalar("test_MAE_loss", losses["test_MAE_loss"].avg, 0)
    train_writer.add_scalar("test_RMSE_loss", losses["test_RMSE_loss"].avg, 0)
    train_writer.add_scalar("test_MAPE_loss", losses["test_MAPE_loss"].avg, 0)

    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__": 
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)