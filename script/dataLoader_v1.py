import csv
import os
import random

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm


class MOF_encode():
    def __init__(self, name, node, linker, topo, prop, struc=None, p=None):
        self.name = name
        self.node = pre_processing(node, False)
        self.linker = [".".join(pre_processing(linker))]
        self.topo = pre_processing(topo, False)
        self.y = prop
        self.struc = struc
        self.p = p


def pre_processing(string, bracket=True):
    if bracket:
        list = string.replace(" ", "").split("'")
        for i in list:
            if i == ",":
                list.remove(i)
        list.remove("[")
        list.remove("]")
    else:
        list = string.split(",")
    return list


def dict_create(item, item_dict):
    item_list = item
    for n in item_list:
        item_dict[n]
    return item_dict


def get_iden(atom):
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic():
        symbol = symbol.lower()
    charge = atom.GetFormalCharge()
    H_num = atom.GetTotalNumHs()
    return "".join(["[", symbol, "H", str(H_num), "|", "+" * charge, "-" * -charge, "]"])


def get_adjacency(l, path, index):
    m = Chem.MolFromSmiles(l)
    matrix = np.array(Chem.GetAdjacencyMatrix(m))
    matrix_data = pd.DataFrame(matrix)
    symbols = []
    for a in m.GetAtoms():
        symbols.append(get_iden(a))  # a.GetSymbol()
    for i in range(matrix_data.shape[0]):
        matrix_data.rename(columns={i: symbols[i]}, inplace=True)
        matrix_data.rename({i: symbols[i]}, inplace=True)
    matrix_data.to_csv(path + str(index) + ".csv")
    index += 1
    return index


def read_embeddings(path, dict):
    path = open(path, "r")
    reader = csv.reader(path)
    for row in reader:
        dict[row[0]] = row[1:]
    return dict


class MOFDataset():
    def __init__(self, params):
        self.params = params
        self.items = self._csv_reader()
        self._extract_feature()
        self._generate_feature()

    def _csv_reader(self):
        items = []

        data = pd.read_csv(self.params["input_data"])
        lit = list(range(len(data)))
        # if self.params["run_state"]:
        #     random.shuffle(lit)
        #     lit = lit[:self.params["run_number"]]
        if self.params["rand_test"]:
            lit_rand = lit.copy()
            for i in range(self.params["rand_cycle"]):
                random.shuffle(lit_rand)
                print(lit_rand[:10])
        else:
            lit_rand = lit.copy()
        for (i, j) in tqdm(zip(lit, lit_rand)):
            name, node, linker, topo = data["name"][i], \
                                       data["metal_smiles"][i], \
                                       data["organ_smiles"][i], \
                                       data["topo"][i],
            prop = data[self.params["props"]].values[j] if self.params["rand_test"] else \
                data[self.params["props"]].values[i]

            struc = data[self.params["struc"]].values[i] if self.params["use_struc"] else None
            p = data["pressure"].values[i] if self.params["use_pres"] else None
            item = MOF_encode(name, node, linker, topo, prop, struc, p)
            items.append(item)
        return items

    def _extract_feature(self):
        bond_dic = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1.5}
        for item in self.items:
            symbols, edges = [], []
            edge_attrs = []
            atom_num = 0
            linker_list = item.linker
            for linker in linker_list:
                symbol, edge = [], []
                edge_attr = []
                m = Chem.MolFromSmiles(linker)
                atom_num += len(m.GetAtoms())
                for a in m.GetAtoms():
                    symbol.append(get_iden(a))  # a.GetSymbol()
                symbols.append(symbol)

                for b in m.GetBonds():
                    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    edge.append([i, j])
                    edge.append([j, i])
                    bond = m.GetBondBetweenAtoms(i, j)
                    type = bond.GetBondType()
                    edge_attr.append([bond_dic[str(type)]])
                    edge_attr.append([bond_dic[str(type)]])

                edges.append(torch.tensor(np.transpose(edge), dtype=torch.long))
                edge_attrs.append(edge_attr)
            item.linker_sym = symbols
            item.edge_index = edges
            item.edge_attr = edge_attrs
            item.atom_num = atom_num

    def _generate_feature(self):
        items = self.items
        print(f"MOF number: {len(items)}")

        iden_set, node_set, topo_set = [], [], []
        for item in items:
            for i in list(item.linker_sym):
                iden_set += i
            for i in list(item.node):
                node_set.append(i)
            for i in list(item.topo):
                topo_set.append(i)

        iden_set, node_set, topo_set = list(set(iden_set)), list(set(node_set)), list(set(topo_set))
        iden_set.sort()
        node_set.sort()
        try:
            topo_set.remove("None")
        except:
            pass
        topo_set.sort()
        topo_set += ["None"]
        self.iden_i2c = {i: c for i, c in enumerate(iden_set)}
        self.iden_c2i = {c: i for i, c in enumerate(iden_set)}
        self.node_i2c = {i: c for i, c in enumerate(node_set)}
        self.node_c2i = {c: i for i, c in enumerate(node_set)}
        self.topo_i2c = {i: c for i, c in enumerate(topo_set)}
        self.topo_c2i = {c: i for i, c in enumerate(topo_set)}
        self.topo_pad = self.topo_c2i["None"]

    def n2v_embedding(self, adj_gen_state=False, n2v_emb_state=False):
        if adj_gen_state is True:
            smiles = set()
            for item in tqdm(self.items):
                smiles.update(item.linker)
            smiles = list(smiles)

            import shutil, time
            try:
                shutil.rmtree(self.params["store_matrix_path"])
            except:
                pass
            time.sleep(3)
            try:
                os.makedirs(self.params["store_matrix_path"])
            except:
                pass
            print("Generating adjacency for molecules...")
            l_1 = 0
            for smile in tqdm(smiles):
                l_1 = get_adjacency(smile, self.params["store_matrix_path"], l_1)

    def assig_feature(self):
        for item in self.items:
            linker_features = []
            for linker_fea in item.linker_sym:
                linker_feature = []
                for a in linker_fea:
                    if not self.params["use_n2v_emb"]:
                        linker_feature.append(self.iden_c2i[a])
                linker_features.append(torch.tensor(linker_feature, dtype=torch.float))
            item.x = linker_features

            node_features = []
            for node in item.node:
                node_features.append(self.node_c2i[node])
            item.node = node_features

            topo_features = []
            for topo in item.topo:
                topo_features.append(self.topo_c2i[topo])
            item.topo = topo_features

    def data_load(self):
        items = self.items
        data_load_er = []
        for item in items:
            if not self.params["use_n2v_emb"]:
                if not self.params["use_pres"]:
                    data = Data(x=torch.tensor(item.x[0], dtype=torch.long),
                                edge_index=item.edge_index[0],
                                node=torch.tensor(item.node, dtype=torch.long),
                                topo=torch.tensor(item.topo, dtype=torch.long),
                                edge_attr=torch.tensor(item.edge_attr[0], dtype=torch.float32),
                                y=item.y,
                                struc=item.struc,
                                name=item.name,
                                atom_num=item.atom_num,
                                pos=None)
                else:
                    data = Data(x=torch.tensor(item.x[0], dtype=torch.long),
                                edge_index=item.edge_index[0],
                                node=torch.tensor(item.node, dtype=torch.long),
                                topo=torch.tensor(item.topo, dtype=torch.long),
                                edge_attr=torch.tensor(item.edge_attr[0], dtype=torch.float32),
                                y=item.y,
                                struc=item.struc,
                                name=item.name,
                                atom_num=item.atom_num,
                                p=item.p,
                                pos=None)
            else:
                data = Data(x=item.x[0],
                            edge_index=item.edge_index[0],
                            node=torch.tensor(item.node, dtype=torch.long),
                            topo=torch.tensor(item.topo, dtype=torch.long),
                            edge_attr=torch.tensor(item.edge_attr[0], dtype=torch.float32),
                            y=item.y,
                            struc=item.struc,
                            name=item.name,
                            atom_num=item.atom_num,
                            pos=None)
            data_load_er.append(data)
        return data_load_er
