import torch
from torch_geometric.data import Dataset
import os

class MoleculeCaption(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        smiles_name = self.smiles_name_list[index]

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list) + '\n'

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return data_graph, text, smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token

class MoleculeCaption_double(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):

        if self.root ==  "data/solve_data/random_test/":
            return 324689
        else :
            return 32
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        text1_name_list = os.listdir(self.root+'text1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        text2_name_list = os.listdir(self.root+'text2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text1/'+str(index)+'/', text1_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text1 = ' '.join(text_list) + '\n'
        text1 = 'The description of the first molecule is that '+text1
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles1/'+str(index)+'/', smiles1_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt1 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt1 = self.prompt
        
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph2/'+str(index)+'/',graph2_name_list[0])
        data_graph2 = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text2/'+str(index)+'/', text2_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text2 = ' '.join(text_list) + '\n'
        text2 = ' And the description of the second molecule is that '+text2
        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles2/'+str(index)+'/', smiles2_name_list[0])
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()
        #这里的smiles是正常的smiles呢
        if self.prompt.find('{}') >= 0:
            smiles_prompt2 = self.prompt.format(smiles[:128])
        else:
            smiles_prompt2 = self.prompt
        graph_prompt1="The first molecule is that"
        graph_prompt2=" and the secoond molecule is that"
        smiles_prompt = graph_prompt1+smiles_prompt1+graph_prompt2+smiles_prompt2
        text =text1+text2
        return data_graph1,data_graph2, text ,smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token
    
    
    
    
if __name__ == '__main__':
    import numpy as np
    pretrain = MoleculeCaption('../data/PubChemDataset_v4/pretrain/', 1000, '')
    train = MoleculeCaption('../data/PubChemDataset_v4/train/', 1000, '')
    valid = MoleculeCaption('../data/PubChemDataset_v4/valid/', 1000, '')
    test = MoleculeCaption('../data/PubChemDataset_v4/test/', 1000, '')

    for subset in [pretrain, train, valid, test]:
        g_lens = []
        t_lens = []
        for i in range(len(subset)):  
            data_graph, text, _ = subset[i]
            g_lens.append(len(data_graph.x))
            t_lens.append(len(text.split()))
            # print(len(data_graph.x))
        g_lens = np.asarray(g_lens)
        t_lens = np.asarray(t_lens)
        print('------------------------')
        print(g_lens.mean())
        print(g_lens.min())
        print(g_lens.max())
        print(t_lens.mean())
        print(t_lens.min())
        print(t_lens.max())
