import os
import re
import torch
import random
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat, chain
def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges

def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(smiles):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    mol = Chem.MolFromSmiles(smiles)
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data
import pandas as pd
import numpy as np
import os 
text = pd.read_csv('data/ddi_data/Interaction_information.csv')
#首先解决testing的数据集
test_data = pd.read_csv('data/ddi_data/DDI_data/ddi_test.csv')
text = np.array(text)
print(text[0][3])
dic = {}
for i in range(len(text)):
    dic[text[i][3]] = text[i][1]
test_data = np.array(test_data)
#print(text)
#print(test_data)
all_drug = pd.read_excel("data/ddi_data/drug.xlsx",sheet_name='Sheet1')
all_drug = np.array(all_drug)
#开始操作在这里
for i in range(len(test_data)):
    print(i)
    os.makedirs("data/ddi_data/drugbank/test/graph1/"+str(i))
    os.makedirs("data/ddi_data/drugbank/test/graph2/"+str(i))
    os.makedirs("data/ddi_data/drugbank/test/smiles1/"+str(i))
    os.makedirs("data/ddi_data/drugbank/test/smiles2/"+str(i))
    os.makedirs("data/ddi_data/drugbank/test/text/"+str(i))
    data = mol_to_graph_data_obj_simple(test_data[i][6])
    torch.save(data, "data/ddi_data/drugbank/test/graph1/"+str(i)+'/graph_data.pt')
    data = mol_to_graph_data_obj_simple(test_data[i][7])
    torch.save(data, "data/ddi_data/drugbank/test/graph2/"+str(i)+'/graph_data.pt')
    text = dic["DDI type "+str(test_data[i][4]+1)]
    file = open("data/ddi_data/drugbank/test/text/"+str(i)+"/text.txt","w")
    file.write(text)
    file.close()
    smiles1 = test_data[i][6]
    smiles2 = test_data[i][7]
    file = open("data/ddi_data/drugbank/test/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("data/ddi_data/drugbank/test/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()
text = pd.read_csv('data/ddi_data/Interaction_information.csv')
#首先解决training的数据集
train_data = pd.read_csv('data/ddi_data/DDI_data/ddi_training.csv')
text = np.array(text)
print(text[0][3])
dic = {}
for i in range(len(text)):
    dic[text[i][3]] = text[i][1]
train_data = np.array(train_data)
#print(text)
#print(train_data)
all_drug = pd.read_excel("data/ddi_data/drug.xlsx",sheet_name='Sheet1')
all_drug = np.array(all_drug)
#开始操作在这里
for i in range(len(train_data)):
    print(i)
    os.makedirs("data/ddi_data/drugbank/train/graph1/"+str(i))
    os.makedirs("data/ddi_data/drugbank/train/graph2/"+str(i))
    os.makedirs("data/ddi_data/drugbank/train/smiles1/"+str(i))
    os.makedirs("data/ddi_data/drugbank/train/smiles2/"+str(i))
    os.makedirs("data/ddi_data/drugbank/train/text/"+str(i))
    data = mol_to_graph_data_obj_simple(train_data[i][6])
    torch.save(data, "data/ddi_data/drugbank/train/graph1/"+str(i)+'/graph_data.pt')
    data = mol_to_graph_data_obj_simple(train_data[i][7])
    torch.save(data, "data/ddi_data/drugbank/train/graph2/"+str(i)+'/graph_data.pt')
    text = dic["DDI type "+str(train_data[i][4]+1)]
    file = open("data/ddi_data/drugbank/train/text/"+str(i)+"/text.txt","w")
    file.write(text)
    file.close()
    smiles1 = train_data[i][6]
    smiles2 = train_data[i][7]
    file = open("data/ddi_data/drugbank/train/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("data/ddi_data/drugbank/train/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()
text = pd.read_csv('data/ddi_data/Interaction_information.csv')
#首先解决validing的数据集
valid_data = pd.read_csv('data/ddi_data/DDI_data/ddi_validation.csv')
text = np.array(text)
print(text[0][3])
dic = {}
for i in range(len(text)):
    dic[text[i][3]] = text[i][1]
valid_data = np.array(valid_data)
#print(text)
#print(valid_data)
all_drug = pd.read_excel("data/ddi_data/drug.xlsx",sheet_name='Sheet1')
all_drug = np.array(all_drug)
#开始操作在这里
for i in range(len(valid_data)):
    print(i)
    os.makedirs("data/ddi_data/drugbank/valid/graph1/"+str(i))
    os.makedirs("data/ddi_data/drugbank/valid/graph2/"+str(i))
    os.makedirs("data/ddi_data/drugbank/valid/smiles1/"+str(i))
    os.makedirs("data/ddi_data/drugbank/valid/smiles2/"+str(i))
    os.makedirs("data/ddi_data/drugbank/valid/text/"+str(i))
    data = mol_to_graph_data_obj_simple(valid_data[i][6])
    torch.save(data, "data/ddi_data/drugbank/valid/graph1/"+str(i)+'/graph_data.pt')
    data = mol_to_graph_data_obj_simple(valid_data[i][7])
    torch.save(data, "data/ddi_data/drugbank/valid/graph2/"+str(i)+'/graph_data.pt')
    text = dic["DDI type "+str(valid_data[i][4]+1)]
    file = open("data/ddi_data/drugbank/valid/text/"+str(i)+"/text.txt","w")
    file.write(text)
    file.close()
    smiles1 = valid_data[i][6]
    smiles2 = valid_data[i][7]
    file = open("data/ddi_data/drugbank/valid/smiles1/"+str(i)+"/text.txt","w")
    file.write(smiles1)
    file.close()
    file = open("data/ddi_data/drugbank/valid/smiles2/"+str(i)+"/text.txt","w")
    file.write(smiles2)
    file.close()