# -*- coding: utf-8 -*-
# @Time:     7/5/2021 9:14 AM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     training.py

import csv
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

from configs import get_params
from dataLoader_v1 import MOFDataset
from model_v1 import MOFNet
from model_v1_FNN import MOFNet_FNN
from utils import y_scaling, struc_scaling, EarlyStopping


def train(dim, aggr, ni, act, bs, task, target, rand_cycle=None, save_model=False):
    params = get_params(task)
    params["props"] = [target]
    if params["rand_test"]:
        params["rand_cycle"] = rand_cycle
    print(params, flush=True)
    random.seed(params["rand_seed"])
    torch.manual_seed(params["rand_seed"])
    np.random.seed(random.seed(params["rand_seed"]))

    dataset = MOFDataset(params)
    dataset.n2v_embedding(params["adj_gen_state"], params["n2v_emb_state"])
    dataset.assig_feature()
    if params["y_scaler"]:
        dataset, y_scaler = y_scaling(dataset)
    if params["use_struc"]:
        dataset, struc_scaler = struc_scaling(dataset)
        if save_model:
            joblib.dump(struc_scaler, "".join([params["output_path"], "struc_scaler_", params["props"][0], ".pkl"]))
    data_loader = dataset.data_load()
    node, prop = [], []
    for i in range(len(data_loader)):
        node.append(data_loader[i].atom_num)
        prop.append(data_loader[i].y)
    print("Largest graph with nodes of", np.max(node), np.argmax(node), flush=True)
    print("Average prop:", np.average(prop), flush=True)
    random.shuffle(data_loader)

    data_size = int(len(data_loader) * 0.1)
    test_dataset = data_loader[:data_size]
    val_dataset = data_loader[data_size:2 * data_size]
    train_dataset = data_loader[2 * data_size:]

    # id = open(params["output_path"] + "Training_record.csv", "w", newline="")
    # writer = csv.writer(id)
    # writer.writerow(["ni", "dim", "aggr", "act", "bs"] + ["epoch", "loss",
    #                                                       "mse_tr", "mae_tr", "r2_tr",
    #                                                       "mse_val", "mae_val", "r2_val",
    #                                                       "mse_te", "mae_te", "r2_te"])
    # id.close()

    print("Start training...", flush=True)

    ####
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=data_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=data_size, shuffle=False)
    print(f"MOF in training set: {len(train_dataset)}", flush=True)
    ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hyper = [ni, dim, aggr, act, bs]
    print(f"\nCase: {'-'.join(map(str, hyper))}", flush=True)
    params["iden_num"], params["node_num"], params["topo_num"], params["topo_pad"] = \
        len(dataset.iden_c2i), len(dataset.node_c2i), len(dataset.topo_c2i), dataset.topo_pad
    if params["use_chem"]:
        model = MOFNet(params, ni, dim, aggr, act)
    else:
        model = MOFNet_FNN(params, ni, dim, aggr, act)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=params["scheduler_factor"],
                                                           patience=params["scheduler_patience"],
                                                           min_lr=0.0001)
    earlystop_er = EarlyStopping(patience=params["earlystop_er_patience"])

    for i in range(params["epoch"]):
        if not earlystop_er.early_stop:
            lr = scheduler.optimizer.param_groups[0]['lr']
            model.train()
            loss_all = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()

                tmp_pred_data = model(data)
                data.y = torch.tensor(data.y, dtype=torch.float32)
                tmp_real_data = data.y[~torch.isnan(data.y)].view(-1, 1)
                tmp_pred_data_mod = tmp_pred_data[~torch.isnan(data.y)].view(-1, 1)
                loss = F.mse_loss(tmp_pred_data_mod, tmp_real_data)
                loss.backward()
                loss_all += loss.item() * data.num_graphs
                optimizer.step()
            loss = loss_all / len(train_loader.dataset)

            def test(loader, set):
                model.eval()
                dfs, names, sets = [], [], []
                for data in loader:
                    data.y = torch.tensor(data.y, dtype=torch.float32)
                    df = pd.DataFrame(np.concatenate((data.y, model(data).detach().numpy()), axis=1),
                                      columns=["y", "f"])
                    dfs.append(df)
                    names += data.name
                    sets += [set] * len(data.name)
                DF = pd.concat(dfs, ignore_index=True)
                DF["name"] = names
                DF["set"] = sets
                y, f = DF["y"].values, DF["f"].values
                r2 = r2_score(y, f)
                mae = mean_absolute_error(y, f)
                mse = mean_squared_error(y, f)

                return r2, mae, mse, DF

            r2_tr, mae_tr, mse_tr, df_tr = test(train_loader, "training")
            r2_val, mae_val, mse_val, df_val = test(val_loader, "validation")
            r2_te, mae_te, mse_te, df_te = test(test_loader, "test")

            scheduler.step(mse_val)
            earlystop_er(mse_val)

            print('Epoch: {:03d}, LR: {:7f}, Loss: {:.4f}, MSE: {:.4f}/{:.4f}/{:.4f}, '
                  'MAE: {:.4f}/{:.4f}/{:.4f}, R2: {:.4f}/{:.4f}/{:.4f}'
                  .format(i + 1, lr, loss, mse_tr, mse_val, mse_te, mae_tr, mae_val, mae_te,
                          r2_tr, r2_val, r2_te), flush=True)
            if earlystop_er.update:
                update = [i + 1, loss, mse_tr, mae_tr, r2_tr, mse_val, mae_val, r2_val,
                          mse_te, mae_te, r2_te]
                model_update = model
                df_tr_update = df_tr
                df_val_update = df_val
                df_te_update = df_te
                if save_model:
                    torch.save(model_update, "".join(
                        [params["output_path"], "_".join(map(str, hyper)), "_", params["props"][0], "_model.pkl"]))
    try:
        if save_model:
            id = open(params["output_path"] + "Training_record_" + params["props"][0] + "_single.csv", "a", newline="")
        else:
            id = open(params["output_path"] + "Training_record_" + params["props"][0] + ".csv", "a", newline="")
        writer = csv.writer(id)
        writer.writerow(hyper + update)
        id.close()
        df_all = pd.concat([df_tr_update, df_val_update, df_te_update], ignore_index=True)
        if save_model:
            df_all.to_csv(
                "".join([params["output_path"], "_".join(map(str, hyper)), "_", params["props"][0], "_all_single.csv"]))
        else:
            df_all.to_csv(
                "".join([params["output_path"], "_".join(map(str, hyper)), "_", params["props"][0], "_all.csv"]))
    except:
        pass
    print('BEST => Epoch: {:03d}, Loss: {:.4f}, MSE: {:.4f}/{:.4f}/{:.4f}, '
          'MAE: {:.4f}/{:.4f}/{:.4f}, R2: {:.4f}/{:.4f}/{:.4f}'
          .format(update[0], update[1], update[2], update[5], update[8],
                  update[3], update[6], update[9], update[4], update[7], update[10]), flush=True)


# if __name__ == "__main__":
#     train(16, "mean", 1, "elu", 256, task="C2", target="C2H6_100", save_model=False)
