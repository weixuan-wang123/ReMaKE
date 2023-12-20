<div align="center">
<h1>
ReMaKE
</h1>
</div>

This repository contains the data and codes for our paper "[Retrieval-augmented Multilingual Knowledge Editing]()".
### 1. Data & Model

MzsRE is located in ./data/MzsRE/

Models are located in ./model/ You can download from [google drive](https://drive.google.com/drive/folders/1uvGMUapE775srRd9GAWyb9o0ENmMU2lo?usp=sharing)

### 2. Edit
```
python run_bizsre.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b-16.yaml --data_dir=./data --metrics_save_dir ./results/llama2-7b/16shot/ --backbone llama2_7b-16shot_classifier --search classifier
```

### 3. Evaluate
```
python evaluate.py
```
### Acknowledgement
- Our codes are based on [Bi-ZsRE](https://github.com/krystalan/Bi-ZsRE/tree/main), and we thank their outstanding open-source contributions.
- Our data is based on vanilla ZsRE dataset ([Levy et al., 2017](https://aclanthology.org/K17-1034/)), Bi-ZsRE dataset [Wang et al., 2023](https://github.com/krystalan/Bi-ZsRE/tree/main) and the portability QA pairs collect by [Yao et al. (2023)](https://arxiv.org/abs/2305.13172).
    - [Zero-Shot Relation Extraction via Reading Comprehension](https://aclanthology.org/K17-1034/) (CoNLL 2017)
    - [Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172) (arXiv preprint 2023)
    - [Cross-Lingual Knowledge Editing in Large Language Models](https://arxiv.org/abs/2309.08952) (arXiv preprint 2023)


### Citation
If you find this work is useful or use the data in your work, please consider cite our paper:
