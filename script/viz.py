# -*- coding: utf-8 -*-
# @Time:     10/14/2021 9:06 PM
# @Author:   Zihao Wang, zwang@mpi-magdeburg.mpg.de
# @File:     viz.py

from viz_utils import *

# Setting
plt.rcParams['font.size'] = "11"
plt.rcParams['font.family'] = "Arial"
plt.rcParams["figure.figsize"] = (3.3, 3)

# H2 data
mof_list = ["metal_smiles", "topo"]
geom_list = ["lcd", "pld", "vf", "gsa", "vsa"]
prop_list = ["H2_2", "H2_100"]
df = pd.read_csv("../data/hMOF_H2.csv", usecols=mof_list + geom_list + prop_list)
df["delta_H2"] = df["H2_100"] - df["H2_2"]
df["color"] = [color_metal(t) for t in df["metal_smiles"]]

" Figure 2 "
# H2_Ads_Des_DC(df)

" Figure 3 & S2 "
# H2_Prop_Ads_DC(df)

" Figure S3 "
H2_screening()

# C2 data
geom_list = ["LCD", "PLD", "VF", "GSA", "VSA"]
prop_list = ["C2H4_100", "C2H6_100"]
df = pd.read_csv("../data/hMOF_C2.csv", usecols=mof_list + geom_list + prop_list)
df["S"] = df["C2H6_100"] / df["C2H4_100"] * 15
df["color"] = [color_metal(t) for t in df["metal_smiles"]]

" Figure 4 & 9 "
# pasity_plot()

" Figure 5 & 10 "
# hist_plot()

" Figure 7 & 8 "
# C_Prop_Ads_DC(df)

" Figure S5 "
# C2_Selec_Prop(df)

" Figure S7 "
# C2_screening()

# PSE_pasity_plot()
# PSE_hist_plot()