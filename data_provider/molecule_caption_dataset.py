import torch
from torch_geometric.data import Dataset
import os
import pandas as pd
import numpy as np

def count_subdirectories(folder_path):
    try:
        # 获取文件夹下的所有文件和子文件夹名
        entries = os.listdir(folder_path)

        # 过滤出子文件夹
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]

        # 返回子文件夹的数量
        return len(subdirectories)
    except FileNotFoundError:
        print(f"文件夹 '{folder_path}' 不存在。")
        return -1  # 返回 -1 表示文件夹不存在
    except Exception as e:
        print(f"发生错误：{e}")
        return -2  # 返回 -2 表示发生了其他错误
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
        text1 = ' '.join(text_list)
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
            
        smiles_prompt = smiles_prompt1+"The front is the first molecule, followed by the second molecule."+smiles_prompt2
        text =text1+"The front is a description of the first molecule, followed by a description of the second molecule."+text2
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

class MoleculeCaption_double_value(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_value, self).__init__(root)
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

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
            #return 100
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
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
            
        smiles_prompt = smiles_prompt1+"The front is solvent, followed by solute."+smiles_prompt2+",what is the solvation Gibbs free energy of this pair of molecules?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
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
    
class MoleculeCaption_double_DDIvalue(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_DDIvalue, self).__init__(root)
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

        if 'train' in self.root:
            return count_subdirectories(self.root+"text/")
        else :
            return count_subdirectories(self.root+"text/")
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
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
            
        smiles_prompt = smiles_prompt1+"The front is the first molecule, followed by the second molecule."+smiles_prompt2+"What are the side effects of these two drugs?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
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
class MoleculeCaption_double_fgtvalue(Dataset):
    def __init__(self, root, text_max_len, prompt=None):
        super(MoleculeCaption_double_fgtvalue, self).__init__(root)
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

        if self.root ==  "data/faguangtuan_data/data1/solve_data/train/":
            return 15999
        else :
            return 2100
    #return 5

    def __getitem__(self, index):
        graph1_name_list = os.listdir(self.root+'graph1/'+str(index)+'/')
        smiles1_name_list = os.listdir(self.root+'smiles1/'+str(index)+'/')
        graph2_name_list = os.listdir(self.root+'graph2/'+str(index)+'/')
        smiles2_name_list = os.listdir(self.root+'smiles2/'+str(index)+'/')
        text_name_list = os.listdir(self.root+'text/'+str(index)+'/')
        
        # load and process graph
        graph_path = os.path.join(self.root, 'graph1/'+str(index)+'/',graph1_name_list[0])
        data_graph1 = torch.load(graph_path)
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
            
        smiles_prompt = smiles_prompt1+"The front is the chromophore, followed by the solvent."+smiles_prompt2+"What is the Emission max?"
        # load and process text
        text_path = os.path.join(self.root, 'text/'+str(index)+'/', text_name_list[0])
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list)
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
