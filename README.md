# MolTC: Towards Molecular Relational Modeling In Language Models



## Requirements

See `environment.yml`. Run the following command to create a new anaconda environment `molca`: 

```bash
conda env create -f environment.yml
```

## Dataset

* **Drugbank, ZhangDDI, ChChMiner, DeepDDI, TWOSIDES**.
* **CombiSolv-QM, CompSol, FreeSolv, Abraham, CombiSolv.**
* You can download data and pre-trained models from the link  [link](directory.https://pan.baidu.com/s/1JiZOJPTTsiri9M_lJN_l9w?pwd=9ww5 ), [link](https://pan.baidu.com/s/1Ee-FK5NL_j1Twwsfdq2Cbg?pwd=go8h), [link](https://pan.baidu.com/s/1mnl_mUjs7--noZfv8lctEw?pwd=mdr7), [link](https://pan.baidu.com/s/1BrkcIr7KmrU4eBqGFB6ajg?pwd=8eeo), [link](https://pan.baidu.com/s/1vXXeIH0B-ofxmRD_3hf1ew?pwd=vi73),[link](https://pan.baidu.com/s/1kX5bn08TSFG4mOonRJmfyw?pwd=nffp)and put them in the current folder. 


## Reproduce the results

### Training the Model from DDI

**data processing.** Run the following script for data processing on the Drugbank dataset:

```bash
python drugbank_ddi.py
```

**training stage.** Run the following script for training stage on the Drugbank dataset:

```bash
python stage2.py --root 'data/ddi_data/drugbank/train/' --valid_root 'data/ddi_data/drugbank/valid/' --devices '0,1,2,3' --filename "ft_pubchem324k_new" --stage2_path "all_checkpoints/ft_pubchem324k_1/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 10  --batch_size 32 --DDI True --caption_eval_epoch 100
```


### Training the Model from Solvation Gibbs Free Energy Prediction

**data processing.** Run the following script for data processing on the CombiSolv-QM and CompSol dataset:

```bash
python CompSol.py
```

```bash
python pretrain_data.py
```

**pretraining stage.** Run the following script for pretraining stage on the CombiSolv-QM dataset:

```bash
python stage2.py --root 'data/solve_data/pre_train/train/' --valid_root 'data/solve_data/pre_train/valid/' --devices '0,1,2,3' --filename "ft_pubchem324k_solve_value_new_new" --stage2_path "all_checkpoints/ft_pubchem324k_1/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 200 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 10  --batch_size 36 --solve True --caption_eval_epoch 200
```
**Fine-tune stage.** Run the following script for Fine-tune stage on the CompSol dataset(At the same time, we provide you with corresponding pre-training models):

```bash
python stage2.py --root 'data/solve_data/CompSol/train/' --valid_root 'data/solve_data/CompSol/valid/' --devices '1,6' --filename "ft_pubchem324k_solve_value_Abraham_new" --stage2_path "all_checkpoints/ft_pubchem324k_solve_value_pre_pre_new/epoch=99.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 1000 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 10  --batch_size 2 --solve True --caption_eval_epoch 1 --init_checkpoint "all_checkpoints/ft_pubchem324k_solve_value_pre_pre_new/epoch=99.ckpt" --peft_dir "logger.save_dir_200/lora_epoch_99"
```

## Citation

If you use our codes or checkpoints, please cite our paper:

...
