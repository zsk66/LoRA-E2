# LoRA-E2: Effective and Efficient Low-rank Adaptation

This repository contains the source code for the paper "LoRA-E2: Effective and Efficient Low-rank Adaptation". The project explores the use of Low-rank Adaptation (LoRA) techniques to enhance the performance and efficiency of transformer-based models.

## Repository Structure

The repository is organized into two main directories:

- **NLG**: Contains code for training the **Llama2-7B** model using the **MetaMathQA** dataset.
- **NLU**: Contains code for training the **T5-base** model using the **GLUE** dataset.

### Directory Overview

LoRA-E2/
├── NLG/
│ ├── data/
│ ├── logs/
│ ├── models/
│ ├── results/
│ ├── evaluate.py
│ ├── main.py
│ ├── utils.py
├── NLU/
│ ├── data/
│ ├── models/
│ ├── results/
│ ├── convert.py
│ ├── main.py
│ ├── utils.py


### NLG: Llama2-7B with MetaMathQA
This folder contains the code and configurations for training the **Llama2-7B** model. The training is done on the **MetaMathQA** dataset. 

### NLU: T5-base with GLUE
This folder contains the code and configurations for training the **T5-base** model using the **GLUE** dataset. 

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/zsk66/LoRA-E2.git
cd LoRA-E2
pip install -r requirements.txt
