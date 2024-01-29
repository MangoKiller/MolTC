# MolTC: A Unified Multi-modal  Large Language Model For Molecular Relational Learning Enhanced by Chain-of-thought

Codes of our ACL2024 paper.

Authors: Junfeng Fang, Shuai Zhang, Chang Wu, Zhiyuan Liu, Sihang Li, Kun Wang, Wenjie Du, Xiang Wang, Xiangnan He


## Comparison to Previous Molecule-Text Modeling Methods

Molecular Relational Learning (MRL), aiming to understand interactions between molecular pairs, plays a pivotal role in advancing biochemical research.
Recently, the adoption of large language models (LLMs), known for their vast knowledge repositories and advanced logical inference capabilities, has emerged as a promising way for efficient and effective MRL.
Despite their potential, these methods predominantly rely on the textual data, thus not fully harnessing the wealth of structural information inherent in molecular graphs.
Moreover, the absence of a unified framework exacerbates the information underutilization, as it hinders the sharing of interaction rationale learned across diverse datasets.
To address these challenges, this work proposes a novel multi-modal LLM for Molecular in Teraction prediction following Chain-of-Thought (CoT) theory, termed MolTC, which can efficiently integrate rich graphical information of molecular pairs. For achieving a unified MRL, MolTC innovatively develops a dynamic parameter-sharing strategy for cross-dataset information exchange, and introduces a Multi-hierarchical CoT principle to refine training paradigm. 

## MolCA's Training Pipeline



## Requirements

See `environment.yml`. Run the following command to create a new anaconda environment `molca`: 

```bash
conda env create -f environment.yml
```

## Dataset

* **Drugbank**.
* **CombiSolv-QM, CompSol.**


## Reproduce the results

### Training the Model from DDI

**data processing.** Run the following script for data processing on the Drugbank dataset:

```bash
python new.py
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
python Pretrain.py
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
