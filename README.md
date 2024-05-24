# MolTC: Towards Molecular Relational Modeling In Language Models
Junfeng Fang, Shuai Zhang, Chang Wu, Zhengyi Yang, Zhiyuan Liu, Sihang Li, Kun Wang, Wenjie Du and Xiang Wang

Arxiv: [https://arxiv.org/abs/2402.03781](https://arxiv.org/abs/2402.03781)

If you have any questions, please contact fjf@mail.ustc.edu.cn.


## Requirements

See `environment.yml`. Run the following command to create a new anaconda environment `molca`: 

```bash
conda env create -f environment.yml
```

## Dataset and pretrained model

* **Drugbank, ZhangDDI, ChChMiner, DeepDDI, TWOSIDES**.
* **CombiSolv-QM, CompSol, FreeSolv, Abraham, CombiSolv.**
* You can download all the data, pre-trained models, backbone GNN models, bert_pretrained model and backbone galactica-1.3b model from the [link](https://huggingface.co/chang04/ddi)
* data should be put in the /data folder. galactica-1.3b should be put in the /galactica-1.3b folder. gin_pretrained should be put in the /gin_pretrained folder. bert_pretrained should be put in the /bert_pretrained folder. pretrain1/last.ckpt should be put in the /all_checkpoints/pretrain1/ folder. 
* For DDI-tasks,We expose the code for training separately on all ddi datasets. We will further release the code for joint training on all datasets in the future.
* For Solvation Gibbs Free Energy Prediction-tasks,You can execute this pretraining stage and use this pretrain_data, or you can download our model directly and then fine-tune it on a small data set.pretrain_model_100w_solve should be put in the  /all_checkpoints/pretrain_model_100w_solve/ folder.This fine-tuning process may end in a few eopch, so it needs to be truncated in time. For the Freesolve data set, we found that fine-tuning directly leads to overfitting. So we directly use the pre-trained model to predict.

## Reproduce the results



**pretraining stage1.** We randomly recombine the molecules in the molecule set in pairs, so that the large language model can recognize two molecules:

```bash
python q-former.py
```
```bash
python stage2.py --root 'qformer_data/train/'  --devices '4,5' --valid_root 'qformer_data/val/'  --filename "stage2" --stage2_path "all_checkpoints/pretrain1/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES].' --tune_gnn --llm_tune freeze --inference_batch_size 2  --double True --batch_size 16
```

### Training the Model from DDI

**data processing.** Run the following script for data processing on the Drugbank, ZhangDDI, ChChMiner, DeepDDI, TWOSIDES dataset:

```bash
python drugbank_ddi.py 
python ZhangDDI.py
python ChChMiner.py
python DeepDDI.py
python TWOSIDES.py
```

**Fine-tune stage.** Run the following script for training stage on the Drugbank, ZhangDDI, ChChMiner, DeepDDI, TWOSIDES dataset:
If you don't have all_checkpoints/stage2/last.ckpt, you can still use all_checkpoints/pretrain1/last.ckpt, which can also achieve good results.We will provide the trained all_checkpoints/stage2/last.ckpt in the future.
```bash
python stage2.py --root 'data/ddi_data/drugbank/train/' --valid_root 'data/ddi_data/drugbank/valid/'  --devices '2,3' --filename "ft_ddi_value_stage2_new" --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 10  --batch_size 36 --DDI True --caption_eval_epoch 50    --max_len 30  --init_checkpoint  "all_checkpoints/stage2/last.ckpt" 
```
```bash
python stage2.py --root 'data/ddi_data/Zhangddi_data/train/' --valid_root 'data/ddi_data/Zhangddi_data/valid/' --devices '4,6,7' --filename "ft_ddi_value_stage2_new16" --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 10  --batch_size 42 --DDI True --caption_eval_epoch 50    --max_len 30  --init_checkpoint  "all_checkpoints/stage2/last.ckpt" 
```
```bash
python stage2.py --root 'data/ddi_data/ChChMiner/train/' --valid_root 'data/ddi_data/ChChMiner/valid/' --devices '4,5,6,7' --filename "ft_ddi_value_stage2_new18" --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 50 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 5  --batch_size 48  --DDI True --caption_eval_epoch 50    --max_len 30  --init_checkpoint  "all_checkpoints/stage2/last.ckpt" 
```
```bash
python stage2.py --root 'data/ddi_data/DeepDDI/train/' --valid_root 'data/ddi_data/DeepDDI/valid/' --devices '4,5,6,7' --filename "ft_ddi_value_stage2_new20" --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 40 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 5  --batch_size 36  --DDI True --caption_eval_epoch 40    --max_len 30  --init_checkpoint  "all_checkpoints/stage2/last.ckpt"
```

### Training the Model from Solvation Gibbs Free Energy Prediction

**data processing.** Run the following script for data processing on the CombiSolv-QM, CompSol, FreeSolv, Abraham and CombiSolv dataset:

```bash
python pretrain_data.py
python CompSol.py
python FreeSolv.py
python Abraham.py
python CombiSolv.py
```

**pretraining stage.** Run the following script for pretraining stage on the pretrain_data dataset:

```bash
python stage2.py --root 'data/solve_data/pre_train/train/' --valid_root 'data/solve_data/pre_train/valid/' --devices '0,1,2,3' --filename "pretrain_model_100w_solve" --stage2_path "all_checkpoints/pretrain1/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 200 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 10  --batch_size 36 --solve True --caption_eval_epoch 200
```
**Fine-tune stage.** Run the following script for Fine-tune stage on the CompSol dataset(At the same time, we provide you with corresponding pre-training models):

```bash
python stage2.py --root 'data/solve_data/CompSol/train/' --valid_root 'data/solve_data/CompSol/valid/' --devices '0,1,2,3' --filename "ft_pubchem324k_solve_value_CompSol_new" --stage2_path "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 1000 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 100  --batch_size 40 --solve True --caption_eval_epoch 1 --init_checkpoint "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --peft_dir "all_checkpoints/pretrain_model_100w_solve/lora_epoch_99"
```

```bash
python stage2.py --root 'data/solve_data/Combisolv/train/' --valid_root 'data/solve_data/Combisolv/valid/' --devices '0,1,2,3' --filename "ft_pubchem324k_solve_value_Combisolv_new_1" --stage2_path "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 5  --batch_size 40 --solve True --caption_eval_epoch 1  --max_len 40 --init_checkpoint "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --peft_dir "all_checkpoints/pretrain_model_100w_solve/lora_epoch_99"
```

```bash
python stage2.py --root 'data/solve_data/Abraham/train/' --valid_root 'data/solve_data/Abraham/valid/' --devices '0,1,2,3' --filename "ft_pubchem324k_solve_value_Abraham_new" --stage2_path "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 1000 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 100  --batch_size 40 --solve True --caption_eval_epoch 1 --init_checkpoint "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --peft_dir "all_checkpoints/pretrain_model_100w_solve/lora_epoch_99"
```
```bash
python stage2.py --root 'data/solve_data/FreeSolv/train/' --valid_root 'data/solve_data/FreeSolv/valid/' --devices '0,1,2,3' --filename "ft_pubchem324k_solve_value_FreeSolv_new" --stage2_path "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 1000 --mode eval --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --save_every_n_epochs 100  --batch_size 40 --solve True --caption_eval_epoch 1 --init_checkpoint "all_checkpoints/pretrain_model_100w_solve/epoch=99.ckpt" --peft_dir "all_checkpoints/pretrain_model_100w_solve/lora_epoch_99"
```

## Citation

Welcome to cite our paper! :)

@misc{fang2024moltc,
      title={MolTC: Towards Molecular Relational Modeling In Language Models}, 
      author={Junfeng Fang and Shuai Zhang and Chang Wu and Zhengyi Yang and Zhiyuan Liu and Sihang Li and Kun Wang and Wenjie Du and Xiang Wang},
      year={2024},
      eprint={2402.03781},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
