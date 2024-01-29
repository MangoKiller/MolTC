# MolTC: A Unified Multi-modal  Large Language Model For Molecular Relational Learning Enhanced by Chain-of-thought

Codes of our EMNLP2023 paper. [[Paper Link](https://arxiv.org/abs/2310.12798)], [[Website](https://acharkq.github.io/MolCA/)], [[Demo](https://8b8760bb1ba284ef54.gradio.live)]

Authors: Junfeng Fang, Shuai Zhang, Chang Wu, Zhiyuan Liu, Sihang Li, Kun Wang, Wenjie Du, Xiang Wang, Xiangnan He


## Comparison to Previous Molecule-Text Modeling Methods

Molecular Relational Learning (MRL), aiming to understand interactions between molecular pairs, plays a pivotal role in advancing biochemical research.
% However, the exhaustive experimental validation of these interactions is notoriously time-consuming and costly. In response, 
Recently, the adoption of large language models (LLMs), known for their vast knowledge repositories and advanced logical inference capabilities, has emerged as a promising way for efficient and effective MRL.
Despite their potential, these methods predominantly rely on the textual data, thus not fully harnessing the wealth of structural information inherent in molecular graphs.
Moreover, the absence of a unified framework exacerbates the information underutilization, as it hinders the sharing of interaction rationale learned across diverse datasets.
To address these challenges, this work proposes a novel multi-modal LLM for \textbf{Mol}ecular in\textbf{T}eraction prediction following \textbf{C}hain-of-Thought (CoT) theory, termed \textbf{MolTC}, which can efficiently integrate rich graphical information of molecular pairs. For achieving a unified MRL, MolTC innovatively develops a dynamic parameter-sharing strategy for cross-dataset information exchange, and introduces a \textit{Multi-hierarchical CoT} principle to refine training paradigm. 

## MolCA's Training Pipeline

![fig3](./static/images/stage1.jpg)

* <b>Pretrain Stage 1.</b> The projector and the encoder are trained to extract the molecule features that are the most relevant to the text. This stage endows the resulting model with powerful molecule-text retrieval ability. 

![fig4](./figures/stage23_cropped.png)

* <b>Pretrain Stage 2 (left).</b> The cross-modal projector is connected to a frozen LM and trained for molecule captioning. This task forces the cross-modal projector to produce soft prompts that the LM can understand
* <b>Finetune Stage (right).</b> MolCA is fine-tuned for downstream generation tasks. The example shows the prediction of a molecule's IUPAC name.

## Requirements

See `environment.yml`. Run the following command to create a new anaconda environment `molca`: 

```bash
conda env create -f environment.yml
```

## Dataset

* **PubChem324k**. Download the dataset from [link](https://huggingface.co/datasets/acharkq/PubChem324kV2), and unzip it under the `./data/` directory.
* **CheBI-20, KV-PLM, and MoMu.** Unzip the `./dataset.zip` under the `./data/` directory. 


## Reproduce the results

### Training the Model from Scratch

**Pretrain Stage 1.** Run the following script for stage 1 pretraining on the PubChem324k dataset:

```bash
python stage1.py --root 'data/PubChem324kV2' --gtm --lm --devices '0,1' --mode train --filename stage1 --rerank_cand_num 128 --num_query_token 8 --tune_gnn
```

**Pretrain Stage 2.** Run the following script for stage 2 pretraining on the PubChem324k dataset:

```bash
python stage2.py --root 'data/PubChem324kV2' --devices '0,1' --filename "stage2" --stage1_path "all_checkpoints/stage1/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES].' --tune_gnn --llm_tune freeze --inference_batch_size 4
```

**Fine-tune Stage.** Run the following script for fine-tuning on the PubChem324k dataset:

```bash
python stage2.py --root 'data/PubChem324kV2' --devices '0,1' --filename "ft_pubchem324k" --stage2_path "all_checkpoints/stage2/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 8
```


### Evaluation on Our Pretrained Checkpoints 

We share the checkpoints for reproducing results of molecule-text retrieval and for reproducing results of molecule captioning on the CheBI-20 dataset.

Please download the checkpoints from this [link](https://huggingface.co/acharkq/MolCA/tree/main) and put them under the `./all_checkpoints` directory.

**Molecule-Text Retrieval for PCDes.** Run the following script for evaluation on the PCDes dataset.

```bash
python stage1.py --root 'data/kv_data' --gtm --lm --devices '[0]'  --filename pcdes_evaluation --init_checkpoint "all_checkpoints/share/stage1.ckpt" --rerank_cand_num 128 --num_query_token 8 --match_batch_size 64 --mode eval
```

**Molecule-Text Retrieval for MoMu.** Run the following script for evaluation on the MoMu dataset.

```bash
python stage1.py --root 'data/kv_data' --gtm --lm --devices '[0]'  --filename momu_evaluation --init_checkpoint "all_checkpoints/share/stage1.ckpt" --rerank_cand_num 128 --num_query_token 8 --match_batch_size 64 --mode eval --use_phy_eval
```

**Molecule Captioning.** Run the following script for evaluation on the CheBI-20 dataset.

```bash
python stage2.py --devices '[0]' --filename chebi_evaluation --stage2_path "all_checkpoints/share/chebi.ckpt" --opt_model 'facebook/galactica-1.3b' --mode eval --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 8 --root "data/ChEBI-20_data" --peft_dir "all_checkpoints/share/chebi_lora" --init_checkpoint all_checkpoints/share/chebi.ckpt;
```

## Citation

If you use our codes or checkpoints, please cite our paper:

```bib
@inproceedings{liu2023molca,
    title={MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter},
    author={Liu, Zhiyuan and Li, Sihang and Luo, Yanchen and Fei, Hao and Cao, Yixin and Kawaguchi, Kenji and Wang, Xiang and Chua, Tat-Seng},
    booktitle={EMNLP},
    year={2023},
    url={https://openreview.net/forum?id=14WRhMNq7H}
}
```
