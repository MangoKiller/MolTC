import torch
from torch_geometric.data import Dataset

import os
from transformers import BertTokenizer

class RetrievalDataset(Dataset):
    def __init__(self, root, args):
        super(RetrievalDataset, self).__init__(root)
        self.root = root
        self.graph_aug = 'noaug'
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root + 'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrained/')
        self.use_smiles = args.use_smiles

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name, smiles_name = self.graph_name_list[index], self.text_name_list[index], self.smiles_name_list[index]
        assert graph_name[len('graph_'):-len('.pt')] == text_name[len('text_'):-len('.txt')] == smiles_name[len('smiles_'):-len('.txt')], print(graph_name, text_name, smiles_name)

        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        text = ''
        if self.use_smiles:
            text_path = os.path.join(self.root, 'smiles', smiles_name)
            text = 'This molecule is '
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                line = line.strip('\n')
                text += f' {line}'
                if count > 1:
                    break
            text += '. '
        
        text_path = os.path.join(self.root, 'text', text_name)
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text += f' {line}'
            if count > 100:
                break
        text += '\n'
        # para-level
        text, mask = self.tokenizer_text(text)
        return data_graph, text.squeeze(0), mask.squeeze(0)  # , index

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


class RetrievalDatasetKVPLM(Dataset):
    def __init__(self, root, args):
        super(RetrievalDatasetKVPLM, self).__init__(root)
        self.root = root
        self.graph_aug = 'noaug'
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        # self.smiles_name_list = os.listdir(root + 'smiles/')
        # self.smiles_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrained/')
        self.use_smiles = args.use_smiles

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        # assert graph_name[len('graph_'):-len('.pt')] == text_name[len('text_'):-len('.txt')] == smiles_name[len('smiles_'):-len('.txt')], print(graph_name, text_name, smiles_name)

        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        text = ''
        text_path = os.path.join(self.root, 'text', text_name)
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text += f' {line}'
            break
        text += '\n'
        # para-level
        text, mask = self.tokenizer_text(text)
        return data_graph, text.squeeze(0), mask.squeeze(0)  # , index

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask
