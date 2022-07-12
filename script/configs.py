# -*- coding: utf-8 -*-
# @Time:     7/5/2021 9:34 AM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     configs.py


def get_params(case):
    params = {}

    if case == "H2":
        params["input_data"] = "../data/hMOF_H2.csv"
        params["output_path"] = "../outputs_H2/"
        params["props"] = ["H2_100"]
        params["struc"] = ["lcd", "pld", "vf", "gsa", "vsa"]
    elif case == "C2":
        params["input_data"] = "../data/hMOF_C2.csv"
        params["output_path"] = "../outputs_C2/"
        params["props"] = ["C2H4_100"]
        params["struc"] = ["LCD", "PLD", "VF", "GSA", "VSA"]

    params["rand_seed"] = 7

    # Data loader
    params["y_scaler"] = False
    params["run_state"] = False  # if True, only {params["run_number"]} data points will be loaded
    params["run_number"] = 12000

    # Neural network
    params["use_chem"] = True
    if not params["use_chem"]:
        params["output_path"] = "../outputs_FNN/"
    params["use_struc"] = True
    params["use_pres"] = False

    # Random test state
    params["rand_test"] = False  # perform randomization (shuffle props while keeping MOF ordered)
    params["rand_cycle"] = 10
    if params["rand_test"]:
        params["output_path"] = "../outputs_RAND/"

    # node2vec
    params["adj_gen_state"] = False
    params["n2v_emb_state"] = False
    params["use_n2v_emb"] = False

    params["embed_dim"] = 2

    # optimizer
    params["lr"] = 0.005
    params["epoch"] = 2000
    params["scheduler_patience"] = 5
    params["scheduler_factor"] = 0.9
    params["earlystop_er_patience"] = 10

    return params
