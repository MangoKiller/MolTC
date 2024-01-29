import torch
from torch_geometric.data import Dataset
import os
import re


def read_iupac(path):
    regex = re.compile('\[Compound\((\d+)\)\]')
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    cid2iupac = {}
    for line in lines:
        smiles, cid, iupac = line.split('\t')
        match = regex.match(cid)
        if match:
            cid = match.group(1)
            cid2iupac[cid] = iupac
    return cid2iupac


class IUPACDataset(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(IUPACDataset, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        
        # self.text_name_list = os.listdir(root+'text/')
        # self.text_name_list.sort()
        
        self.tokenizer = None
        self.cid2iupac = read_iupac(root + 'iupac.txt')
        
        smiles_name_list = os.listdir(root+'smiles/')
        smiles_name_list.sort()
        self.smiles_name_list = []
        graph_name_list = os.listdir(root+'graph/')
        graph_name_list.sort()
        self.graph_name_list = []
        for smiles_name, graph_name in zip(smiles_name_list, graph_name_list):
            cid = smiles_name[7:-4]
            assert cid == graph_name[6:-3]
            if cid in self.cid2iupac:
                self.smiles_name_list.append(smiles_name)
                self.graph_name_list.append(graph_name)
        self.smiles_name_list.sort()
        self.graph_name_list.sort()

        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self.smiles_name_list)

    def __len__(self):
        return len(self.smiles_name_list)

    def __getitem__(self, index):
        graph_name = self.graph_name_list[index] #, self.text_name_list[index]
        smiles_name = self.smiles_name_list[index]
        cid = smiles_name[7:-4]
        iupac = self.cid2iupac[cid]

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()

        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return data_graph, iupac + '\n', smiles_prompt


if __name__ == '__main__':
    dataset = IUPACDataset('../data/PubChemDataset_v4/test/', 128, )
    print(dataset[0], len(dataset))
    dataset = IUPACDataset('../data/PubChemDataset_v4/train/', 128, )
    print(dataset[0], len(dataset))
    dataset = IUPACDataset('../data/PubChemDataset_v4/valid/', 128, )
    print(dataset[0], len(dataset))