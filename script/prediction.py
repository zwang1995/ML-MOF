# -*- coding: utf-8 -*-
# @Time:     9/24/2021 9:32 AM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     prediction.py

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader

from configs import get_params
from dataLoader_v1 import MOFDataset

pred_new_data = True


def get_setting(prop):
    if prop == "C2H4_100":
        scaler_file = "../outputs_PRED/struc_scaler_C2H4_100.pkl"
        model_file = "../outputs_PRED/0_16_mean_tanh_256_C2H4_100_model.pkl"
    elif prop == "C2H6_100":
        scaler_file = "../outputs_PRED/struc_scaler_C2H6_100.pkl"
        model_file = "../outputs_PRED/0_16_mean_elu_256_C2H6_100_model.pkl"
    elif prop == "H2_100":
        scaler_file = "../outputs_PRED/struc_scaler_H2_100.pkl"
        model_file = "../outputs_PRED/1_32_mean_sigmoid_64_H2_100_model.pkl"
    elif prop == "H2_2":
        scaler_file = "../outputs_PRED/struc_scaler_H2_2.pkl"
        model_file = "../outputs_PRED/3_16_mean_sigmoid_256_H2_2_model.pkl"
    return scaler_file, model_file


for prop in ["H2_100", "H2_2", "C2H4_100", "C2H6_100"]:
    start = time.time()
    params = get_params("H2")
    if pred_new_data:
        params["input_data"] = "../data/hMOF_for_screening.csv"
        params["output_path"] = "../outputs_PRED/"
        params["props"] = ["blank"]
        params["struc"] = ["lcd", "pld", "void_fraction", "surface_area_m2g", "surface_area_m2cm3"]
    else:
        params["input_data"] = "../data/hMOF_C2.csv"
        params["props"] = [prop]
    dataset = MOFDataset(params)
    dataset.assig_feature()

    params["props"] = [prop]
    print(prop)
    scaler_file, model_file = get_setting(prop)
    scaler = joblib.load(scaler_file)
    for item in dataset.items:
        item.struc = scaler.transform(item.struc.reshape(1, -1))[0]

    data_loader = dataset.data_load()
    test_dataset = data_loader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = torch.load(model_file)
    model.eval()

    dfs, names, sets = [], [], []
    for data in test_loader:
        tmp_pred_data = model(data)
        tmp_pred_data_mod = tmp_pred_data.view(-1, 1)
        f_ver = tmp_pred_data_mod.detach().numpy()
        df = pd.DataFrame(np.concatenate((data.y, model(data).detach().numpy()), axis=1), columns=["y", "f"])
        dfs.append(df)
        names += data.name
        sets += [set] * len(data.name)
    DF = pd.concat(dfs, ignore_index=True)
    DF["name"] = names
    DF["set"] = sets
    DF.to_csv("../outputs_PRED/Prediction_" + str(prop) + ".csv")
    print(DF, flush=False)
    print(len(DF))
    if not pred_new_data:
        y, f = DF["y"].values, DF["f"].values
        r2 = r2_score(y, f)
        mae = mean_absolute_error(y, f)
        mse = mean_squared_error(y, f)
        print(r2, mse, mae)

    end = time.time()
    print("Used time:", end - start)
