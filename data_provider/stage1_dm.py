# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import torch_geometric
from data_provider.pretrain_dataset import GINPretrainDataset
from data_provider.retrieval_dataset import RetrievalDataset

class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        graph_aug: str = 'dnodes',
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        if root.find('PubChemDataset_v4') > 0:
            print('Loading MoLa dataset')
            self.train_dataset = GINPretrainDataset(root+'/pretrain/', text_max_len, graph_aug, args.text_aug)
        else:
            print('Loading old veresion dataset')
            self.train_dataset = GINPretrainDataset(root+'/train/', text_max_len, graph_aug, args.text_aug)
        self.val_dataset = GINPretrainDataset(root + '/valid/', text_max_len, graph_aug, args.text_aug)
        self.val_dataset_match = RetrievalDataset(root + '/valid/', args).shuffle()
        self.test_dataset_match = RetrievalDataset(root + '/test/', args).shuffle()
        self.val_match_loader = torch_geometric.loader.DataLoader(self.val_dataset_match, 
                                                                  batch_size=self.match_batch_size,
                                                                  shuffle=False,
                                                                  num_workers=self.num_workers, 
                                                                  pin_memory=False, 
                                                                  drop_last=False, 
                                                                  persistent_workers=True)
        self.test_match_loader = torch_geometric.loader.DataLoader(self.test_dataset_match, 
                                                                   batch_size=self.match_batch_size,
                                                                   shuffle=False,
                                                                   num_workers=self.num_workers, 
                                                                   pin_memory=False, 
                                                                   drop_last=False, 
                                                                   persistent_workers=True)
    
    def train_dataloader(self):
        loader = torch_geometric.loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True
        )
        # print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = torch_geometric.loader.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True
        )

        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        #parser.add_argument('--root', type=str, default='data/PubChemDataset/PubChem-320k')
        parser.add_argument('--root', type=str, default='data/kv_data/')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--graph_aug', type=str, default='dnodes')
        parser.add_argument('--text_aug', action='store_true', default=False)
        parser.add_argument('--use_phy_eval', action='store_true', default=False)
        return parent_parser
    