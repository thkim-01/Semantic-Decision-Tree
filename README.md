# Semantic Decision Tree (SDT) for Molecular Classification

## Overview
SMILES 기반 분자 구조를 화학 온톨로지로 변환하고, Semantic Decision Tree를 사용하여 MoleculeNet 벤치마크 독성 데이터셋 6가지 (bbbp,clintox,hiv,tox21,bace,sider) Classification 수행모델.


## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python experiments/bbbp_experiment.py
```

## Evaluation Metric
- AUC-ROC (Area Under the ROC Curve)
