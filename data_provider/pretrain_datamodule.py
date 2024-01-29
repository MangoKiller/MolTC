# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch_geometric
from data_provider.pretrain_dataset import GINPretrainDataset


class GINPretrainDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        graph_aug: str = 'dnodes',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = GINPretrainDataset(root+'/train/', text_max_len, graph_aug)
        self.val_dataset = GINPretrainDataset(root + '/valid/', text_max_len, graph_aug)


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
