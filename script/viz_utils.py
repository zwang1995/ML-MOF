import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

pd.set_option("display.max_columns", None)


def color_metal(t):
    if t == "[Cu][Cu]":
        return "indigo"
    elif t == "[V]":
        return "green"
    elif t == "[Zn][Zn]":
        return "tab:orange"
    elif t == "[Zn][O]([Zn])([Zn])[Zn]":
        return "blue"
    else:
        return "red"


def H2_Ads_Des_DC(df):
    plt.clf()
    df_ = df[df["delta_H2"] < 0]
    x_, y_ = df_["H2_2"].values, df_["H2_100"].values
    df = df[df["delta_H2"] >= 0]
    x, y, z = df["H2_2"].values, df["H2_100"].values, df["delta_H2"].values
    idx = z.argsort()  # [::-1]
    x, y, z = x[idx], y[idx], z[idx]
    print("H2_Ads_Des_DC:", len(x))

    plt.plot([0, 70], [0, 70], color="k", linewidth=1)
    plt.scatter(x_, y_, c="silver", linewidth=0, s=20)
    plt.scatter(x, y, c=z, cmap="viridis", s=20, linewidths=0)
    plt.xlim([0, 70])
    plt.ylim([0, 70])

    plt.xlabel("H$_2$ uptake at 2 bar (g/L)", fontsize=12)
    plt.ylabel("H$_2$ uptake at 100 bar (g/L)", fontsize=12)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    cbar = plt.colorbar(pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label="H$_2$ deliverable capacity (g/L)", size=11)
    plt.clim(0, 50)
    plt.savefig("H2_Ads_Des_DC", dpi=300, bbox_inches="tight", transparent=True)


def H2_Prop_Ads_DC(df):
    def get_setting(prop):
        if prop == "vf":
            z = df["vf"].values
            color = "tab:red"
            label = "Void fraction"
            x_low, x_up = 0, 1
        elif prop == "gsa":
            z = df["gsa"].values
            color = "tab:green"
            label = "Gravimetric surface area (m$^2$/g)"
            x_low, x_up = 0, 7000
        elif prop == "vsa":
            z = df["vsa"].values
            color = "tab:cyan"
            label = "Volumetric surface area (m$^2$/cm$^3$)"
            x_low, x_up = 0, 3500
        elif prop == "pld":
            z = df["pld"].values
            color = "tab:blue"
            label = "Pore limiting diameter (Å)"
            x_low, x_up = 0, 25
        elif prop == "lcd":
            z = df["lcd"].values
            color = "gray"
            label = "Largest cavity diameter (Å)"
            x_low, x_up = 0, 30
        fig_name = "H2_100_" + prop + "_C"
        return z, label, color, fig_name, x_low, x_up

    for prop in ["vf", "gsa", "vsa", "pld"]:
        plt.clf()

        df = df[df["delta_H2"] >= 0]
        z, label, color, fig_name, x_low, x_up = get_setting(prop)
        x = z
        y = df["H2_100"].values
        print("H2_Prop_Ads_DC:", prop, len(x))

        z = df["delta_H2"].values
        idx = z.argsort()  # [::-1]
        x, y, z = x[idx], y[idx], z[idx]

        plt.scatter(x, y, c=z, cmap="jet", linewidth=0, s=20)
        plt.xlim([x_low, x_up])
        plt.ylim([0, 60])
        plt.xlabel(label, fontsize=12)
        plt.ylabel("H$_2$ uptake at 100 bar (g/L)", fontsize=12)
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        cbar = plt.colorbar(pad=0.03)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(label="H$_2$ deliverable capacity (g/L)", size=11)
        plt.clim(0, 50)
        plt.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)


def H2_screening():
    df = pd.read_csv("../data/H2_Top100_ML_GCMC.csv")
    ML = df["ML_DC"].values
    GCMC = df["GCMC_DC"].values
    sns.histplot(ML, discrete=True, color='tab:red')
    sns.histplot(GCMC, discrete=True, color='tab:blue')
    plt.xlim([40.5, 48.5])
    plt.ylim([0, 60])
    plt.xlabel("H$_2$ deliverable capacity (g/L)", size=12)
    plt.ylabel("Count", size=12)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    plt.legend(labels=["ML", "GCMC"], prop={'size': 10})
    plt.savefig("H2_screening", dpi=300, bbox_inches="tight", transparent=True)

    plt.clf()
    x1, y1 = [39.5, 48.5], [39.5, 48.5]
    plt.xlim(x1)
    plt.ylim(y1)
    plt.plot(x1, y1, color="k", linewidth=1)
    plt.xlabel("Simulated capacity (g/L)", fontsize=12)
    plt.ylabel("ML predicted capacity (g/L)", fontsize=12)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    plt.scatter(GCMC, ML, c="tab:blue", s=20, alpha=0.7, linewidths=0)
    plt.savefig("H2_screening_scatter", dpi=300, bbox_inches="tight", transparent=True)

def pasity_plot(density=False):
    def get_setting(prop):
        if prop == "H2_100":
            df = pd.read_csv("./Pasity/1_32_mean_sigmoid_64_H2_100_all_single.csv")
            x_label = "Simulated uptake (g/L)"
            y_label = "ML predicted uptake (g/L)"
            x_low, x_up = -2.4, 62.4
            ticks = [0, 10, 20, 30, 40, 50, 60]
        elif prop == "H2_2":
            df = pd.read_csv("./Pasity/3_16_mean_sigmoid_256_H2_2_all_single.csv")
            x_label = "Simulated uptake (g/L)"
            y_label = "ML predicted uptake (g/L)"
            x_low, x_up = -1.6, 41.6
            ticks = [0, 10, 20, 30, 40]
        elif prop == "C2H4_100":
            df = pd.read_csv("./Pasity/0_16_mean_tanh_256_C2H4_100_all_single.csv")
            x_label = "Simulated uptake (cm$^3$/g)"
            y_label = "ML predicted uptake (cm$^3$/g)"
            x_low, x_up = -6.4, 166.4
            ticks = [0, 40, 80, 120, 160]
        elif prop == "C2H6_100":
            df = pd.read_csv("./Pasity/0_16_mean_elu_256_C2H6_100_all_single.csv")
            x_label = "Simulated uptake (cm$^3$/g)"
            y_label = "ML predicted uptake (cm$^3$/g)"
            x_low, x_up = -0.8, 20.8
            ticks = [0, 5, 10, 15, 20]

        fig_name = "Pasity_" + prop
        return df, x_label, y_label, fig_name, x_low, x_up, ticks

    props = ["H2_100", "H2_2", "C2H4_100", "C2H6_100", "C2H4_10", "C2H6_10"]
    for prop in props:
        plt.clf()
        df, x_label, y_label, fig_name, x_low, x_up, ticks = get_setting(prop)

        if density:
            x, y = df["y"], df["f"]
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            plt.scatter(x, y, c=z, cmap="Blues", s=20)
            plt.colorbar()
        else:
            df_train = df[df["set"] == "training"]
            df_valid = df[df["set"] == "validation"]
            df_test = df[df["set"] == "test"]
            x_train, y_train = df_train["y"], df_train["f"]
            x_valid, y_valid = df_valid["y"], df_valid["f"]
            x_test, y_test = df_test["y"], df_test["f"]

            plt.scatter(x_train, y_train, c="tab:red", s=20, alpha=0.7, linewidths=0)
            # plt.scatter(x_valid, y_valid, c="tab:blue", s=20)
            plt.scatter(x_test, y_test, c="tab:blue", s=20, alpha=0.7, linewidths=0)
            plt.legend(labels=["Training set", "Test set"], prop={'size': 10})

        plt.xlim([x_low, x_up])
        plt.ylim([x_low, x_up])
        x1, y1 = [-10, 210], [-10, 210]
        plt.plot(x1, y1, color="k", linewidth=1)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)


def hist_plot():
    # lables = ["H$_2$ at 100 bar", "H$_2$ at 2 bar"]
    # MAE_1, R2_1 = [1.25, 1.29], [0.976, 0.928]
    # MAE_2, R2_2 = [1.33, 2.02], [0.974, 0.842]
    lables = ["C$_2$H$_4$ at 1 bar", "C$_2$H$_6$ at 1 bar"]
    MAE_1, R2_1 = [5.79, 0.77], [0.896, 0.897]
    MAE_2, R2_2 = [9.00, 1.17], [0.742, 0.772]
    x = range(len(lables))

    plt.bar(x, height=MAE_1, width=0.4, color="tab:cyan")
    plt.bar([xi + 0.4 for xi in x], height=MAE_2, color="tab:gray", width=0.4)
    plt.xticks([xi + 0.2 for xi in x], lables)
    for index, value in enumerate(MAE_1):
        plt.text(index-0.12, value-0.6, format(value, '.2f'), color="w", fontsize=10) # H2-MAE: -0.12/+0.28, -0.13
    for index, value in enumerate(MAE_2):
        plt.text(index + 0.28, value-0.6, format(value, '.2f'), color="w", fontsize=10) # H2-R2: -0.15/+0.25, -0.03
    # for index, value in enumerate(R2_1):
    #     plt.text(index - 0.15, value -0.03, format(value, '.3f'), color="w", fontsize=10)  # C2-MAE: -0.12/+0.28, -0.6
    # for index, value in enumerate(R2_2):
    #     plt.text(index + 0.25, value -0.03, format(value, '.3f'), color="w", fontsize=10)  # C2-H2: -0.15/0.25, -0.025
    # plt.tick_params(labelbottom=False)
    plt.xlim(-0.4, 1.8)
    plt.ylim(0, 10)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("MAE (cm$^3$/g)", fontsize=12)  # MAE (cm$^3$/g) (g/L) #R$^2$
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    # plt.legend(labels=["w/ chemical features", "w/o chemical features"], prop={'size': 9})#, loc=2)
    plt.savefig("C2_R2", dpi=300, bbox_inches="tight", transparent=True)


def C_Prop_Ads_DC(df):
    def get_setting(prop):
        if prop == "VF":
            z = df["VF"].values
            color = "tab:red"
            label = "Void fraction"
            x_low, x_up = 0, 1
        elif prop == "GSA":
            z = df["GSA"].values
            color = "tab:green"
            label = "Gravimetric surface area (m$^2$/g)"
            x_low, x_up = 0, 7000
        elif prop == "VSA":
            z = df["VSA"].values
            color = "tab:cyan"
            label = "Volumetric surface area (m$^2$/cm$^3$)"
            x_low, x_up = 0, 3500
        elif prop == "PLD":
            z = df["PLD"].values
            color = "tab:blue"
            label = "Pore limiting diameter (Å)"
            x_low, x_up = 0, 25
        elif prop == "LCD":
            z = df["LCD"].values
            color = "gray"
            label = "Largest cavity diameter (Å)"
            x_low, x_up = 0, 30
        fig_name = "C2_100_" + prop + "_C"
        return z, label, color, fig_name, x_low, x_up

    for prop in ["VF", "GSA", "VSA", "PLD"]:
        df["C2_100"] = df["C2H4_100"] + df["C2H6_100"]
        df["C2_10"] = df["C2H4_10"] + df["C2H6_10"]
        df["WC"] = df["C2_100"] - df["C2_10"]
        df = df[df["WC"] > 0]
        plt.clf()

        z, label, color, fig_name, x_low, x_up = get_setting(prop)
        x = z
        y = df["C2_100"].values
        print("C2_Prop_Ads_DC:", prop, len(x))

        z = df["WC"].values
        idx = z.argsort()  # [::-1]
        x, y, z = x[idx], y[idx], z[idx]

        plt.scatter(x, y, c=z, cmap="jet", linewidth=0, s=20)
        plt.xlim([x_low, x_up])
        plt.ylim([0, 200])  # 100, 200
        plt.xlabel(label, fontsize=12)
        plt.ylabel("C2 uptake at 1 bar (cm$^3$/g)", fontsize=12)
        # plt.yticks([0, 40, 80, 120, 160])
        # plt.yticks([0, 3, 6, 9, 12, 15])
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        cbar = plt.colorbar(pad=0.03)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(label="C2 working capacity (cm$^3$/g)", size=11)
        plt.clim(0, 140)  # 120, 16
        plt.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)


def C2_Selec_Prop(df, metal=False):
    def get_setting(prop):
        if prop == "VF":
            z = df["VF"].values[~np.isnan(df["S"].tolist())]
            label = "Void fraction"
            cbar_low, cbar_up = 0, 1
        elif prop == "Density":
            z = df["Density"].values[~np.isnan(df["S"].tolist())]
            label = "Framework density (g/cm$^3$)"
            cbar_low, cbar_up = 0, 2
        elif prop == "GSA":
            z = df["GSA"].values[~np.isnan(df["S"].tolist())]
            label = "Gravimetric surface area (m$^2$/g)"
            cbar_low, cbar_up = 0, 7000
        elif prop == "VSA":
            z = df["VSA"].values[~np.isnan(df["S"].tolist())]
            label = "Volumetric surface area (m$^2$/cm$^3$)"
            cbar_low, cbar_up = 0, 3500
        elif prop == "PLD":
            z = df["PLD"].values[~np.isnan(df["S"].tolist())]
            label = "Pore limiting diameter (Å)"
            cbar_low, cbar_up = 0, 25
        elif prop == "LCD":
            z = df["LCD"].values[~np.isnan(df["S"].tolist())]
            label = "Largest cavity diameter (Å)"
            cbar_low, cbar_up = 0, 30
        fig_name = "C2_Cap_Selec_" + prop
        return z, label, fig_name, cbar_low, cbar_up

    labels = ["Void fraction",
              "Gravimetric surface area (m$^2$/g)",
              "Volumetric surface area (m$^2$/cm$^3$)",
              "Pore limiting diameter (Å)",
              "Largest cavity diameter (Å)",
              "Working capacity (cm$^3$/g)",
              ]

    # df = df[df["S"] > 0]

    for prop in ["VF", "GSA", "VSA", "PLD"]:  # "Density", "LCD",
        plt.clf()

        # df = df[df["color"] == "indigo"]
        # df = df[df["color"] == "green"]
        # df = df[df["color"] == "tab:orange"]
        #
        # df = df[df["color"] != "indigo"]
        # df = df[df["color"] != "green"]
        # df = df[df["color"] != "tab:orange"]

        z, label, fig_name, cbar_low, cbar_up = get_setting(prop)
        df = df[df["C2_100"] >= 0]
        df["delta_C2H6"] = df["C2H6_100"] - df["C2H6_10"]
        x = df["delta_C2H6"].values[~np.isnan(df["S"].tolist())]
        y = df["S"].values[~np.isnan(df["S"].tolist())]
        z = z
        idx = z.argsort()  # [::-1]
        x, y, z = x[idx], y[idx], z[idx]
        print("C2_Selec_Prop:", prop, len(x))

        # x_in = np.linspace(0.2, 20, 100)
        # y_in_1, y_in_2 = np.exp(2.5 / x_in) + 1, np.exp(4 / x_in) + 1
        #
        if metal:

            # df = df[df["color"] == "indigo"]
            # df = df[df["color"] == "green"]
            df = df[df["color"] == "tab:orange"]

            # df = df[df["color"] != "indigo"]
            # df = df[df["color"] != "green"]
            # df = df[df["color"] != "tab:orange"]

            z, label, fig_name, cbar_low, cbar_up = get_setting(prop)
            x = df["delta_C2H6"]
            y = df["S"].values[~np.isnan(df["S"].tolist())]
            colors = df["color"].values[~np.isnan(df["S"].tolist())]
            plt.scatter(z, y, c=colors, linewidth=0, alpha=0.7, s=20)

        else:
            plt.scatter(z, y, cmap="jet", linewidth=0, s=20)
        # plt.plot(x_in, y_in_1, "red")
        plt.ylim(0, 7)

        plt.xlim([cbar_low, cbar_up])
        plt.tick_params(axis='x', direction='in')
        plt.tick_params(axis='y', direction='in')

        plt.xlabel(label, fontsize=12)
        plt.ylabel("C$_2$H$_6$/C$_2$H$_4$ selectivity", fontsize=12)

        # plt.clim(cbar_low, cbar_up)
        # cbar = plt.colorbar(pad=0.03)
        # cbar.ax.tick_params(labelsize=10)
        # cbar.set_label(label, fontsize=11)
        plt.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)
        # plt.show()


def C2_screening():
    df = pd.read_csv("../data/C2_Top100_ML_GCMC.csv")
    plt.clf()
    ML = df["ML_S"].values
    GCMC = df["GCMC_S"].values
    sns.histplot(ML, discrete=True, color='tab:red')
    sns.histplot(GCMC, discrete=True, color='tab:blue')
    plt.xlim([0, 10])
    plt.ylim([0, 70])
    plt.xlabel("C$_2$H$_6$/C$_2$H$_4$ adsorption selectivity", size=12)
    plt.ylabel("Count", size=12)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    plt.legend(labels=["ML", "GCMC"], prop={'size': 10})
    plt.savefig("C2_screening_S", dpi=300, bbox_inches="tight", transparent=True)

def PSE_pasity_plot(density=False):
    def get_setting(prop):
        if prop == "H2_100":
            df = pd.read_csv("../Pasity/1_32_mean_sigmoid_64_H2_100_all_single.csv")
            x_label = "Simulated uptake (g/L)"
            y_label = "ML predicted uptake (g/L)"
            x_low, x_up = -2.4, 62.4
            ticks = [0, 10, 20, 30, 40, 50, 60]
        elif prop == "H2_2":
            df = pd.read_csv("../Pasity/3_16_mean_sigmoid_256_H2_2_all_single.csv")
            x_label = "Simulated uptake (g/L)"
            y_label = "ML predicted uptake (g/L)"
            x_low, x_up = -1.6, 41.6
            ticks = [0, 10, 20, 30, 40]
        elif prop == "C2H4_100":
            df = pd.read_csv("../Pasity/0_16_mean_tanh_256_C2H4_100_all_single.csv")
            x_label = "Simulated uptake (cm$^3$/g)"
            y_label = "ML predicted uptake (cm$^3$/g)"
            x_low, x_up = -6.4, 166.4
            ticks = [0, 40, 80, 120, 160]
        elif prop == "C2H6_100":
            df = pd.read_csv("../Pasity/0_16_mean_elu_256_C2H6_100_all_single.csv")
            x_label = "Simulated uptake (cm$^3$/g)"
            y_label = "ML predicted uptake (cm$^3$/g)"
            x_low, x_up = -0.8, 20.8
            ticks = [0, 5, 10, 15, 20]

        fig_name = "Pasity_" + prop
        return df, x_label, y_label, fig_name, x_low, x_up, ticks

    props = ["H2_100", "H2_2"]
    for prop in props:
        plt.clf()
        df, x_label, y_label, fig_name, x_low, x_up, ticks = get_setting(prop)

        if density:
            x, y = df["y"], df["f"]
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            plt.scatter(x, y, c=z, cmap="Blues", s=20)
            plt.colorbar()
        else:
            df_train = df[df["set"] == "training"]
            df_valid = df[df["set"] == "validation"]
            df_test = df[df["set"] == "test"]
            x_train, y_train = df_train["y"], df_train["f"]
            x_valid, y_valid = df_valid["y"], df_valid["f"]
            x_test, y_test = df_test["y"], df_test["f"]

            plt.scatter(x_train, y_train, c="tab:red", s=20, alpha=0.7, linewidths=0)
            # plt.scatter(x_valid, y_valid, c="tab:blue", s=20)
            plt.scatter(x_test, y_test, c="tab:blue", s=20, alpha=0.7, linewidths=0)
            plt.legend(labels=["Training set", "Test set"], prop={'size': 9})

        plt.xlim([x_low, x_up])
        plt.ylim([x_low, x_up])
        x1, y1 = [-10, 210], [-10, 210]
        plt.plot(x1, y1, color="k", linewidth=1)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)

def PSE_hist_plot():
    def get_setting(prop):
        if prop == "H2_100":
            df = pd.read_csv("../Pasity/1_32_mean_sigmoid_64_H2_100_all_single.csv")
        elif prop == "H2_2":
            df = pd.read_csv("../Pasity/3_16_mean_sigmoid_256_H2_2_all_single.csv")
        fig_name = "Error_" + prop
        return df, fig_name

    props = ["H2_100", "H2_2"]
    for prop in props:
        plt.clf()
        df, fig_name = get_setting(prop)
        df_train, df_test = df[df.set == "training"], df[df.set == "test"]
        error_train = np.array(df_train["y"]-df_train["f"])
        error_test = np.array(df_test["y"] - df_test["f"])
        print(len(error_train), len(error_test), np.min(error_train), np.max(error_train), np.min(error_test), np.max(error_test))

        sns.histplot(error_train, discrete=True, color='tab:red',linewidth=0.8)
        sns.histplot(error_test, discrete=True, color='tab:blue', linewidth=0.8)

        plt.xlim([-11, 11])
        plt.ylim([0, 3000])
        plt.xlabel("Prediction error (g/L)", size=12)
        plt.ylabel("Count", size=12)
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        plt.legend(labels=["Training set", "Test set"], prop={'size': 9})
        plt.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)


