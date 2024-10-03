# Question Answering with Mamba

## Overview

This repository contains a framework for training, evaluating, and fine-tuning various language models using Hydra for configuration management and Hugging Face's Transformers library for model handling. The framework supports a wide range of models, including GPT-NeoX, Mamba, and a bidirectional Mamba.

## Acknowledgments

This project builds upon the following codebases:

- [Mamba: State Spaces for Sequence Modeling](https://github.com/state-spaces/mamba)
- [PTM-Mamba](https://github.com/programmablebio/ptm-mamba)
- [Transformers SSM](https://github.com/sjelassi/transformers_ssm_copy)

This project also includes the following libraries as submodules:
- [Mamba](https://github.com/path/to/mamba) (Apache 2.0 License)
- [Causal Conv1D](https://github.com/path/to/causal-conv1d) (MIT License)

## Project Structure

```plaintext
.
├── README.md
├── conf
│   ├── config.yaml          # Main configuration file
│   ├── dataset/             # Dataset-specific configurations
│   ├── model/               # Model-specific configurations
│   └── run/                 # Run-specific configurations
├── main.py                  # Entry point for running tasks (train, evaluate, preprocess)
├── models
│   ├── bidirectional_mamba.py  # Model definition for Bidirectional Mamba
│   ├── libs/                   # Supporting libraries
│   └── model_utils.py          # Utilities for loading models and tokenizers
└── scripts
    ├── evaluate.py            # Script for evaluating models
    ├── preprocess.py          # Script for preprocessing datasets
    ├── preprocess_flipped.py  # Script for preprocessing flipped datasets
    ├── preprocess_original.py # Script for preprocessing original datasets
    └── train.py               # Script for training models
```

## Getting Started

### Installation
Install required dependencies:

   ```bash
pip install transformers datasets accelerate python-dotenv wandb evaluate torchprofile gputil hydra-core --upgrade causal-conv1d mamba-ssm
pip3 install torch torchvision torchaudio
   ```

### Configuration

All configurations are managed using Hydra. The primary configuration file is `conf/config.yaml`, which controls the default and broad configs. You can override any configuration parameter from the command line.


### Wandb Integration

The scripts in this project utilize Weights & Biases (Wandb) for experiment tracking and logging. The scripts expect a `.env` file to be present in the project root directory. This file should contain the following environment variables:

```plaintext
WANDB_USERNAME=<your_wandb_username>
WANDB_PROJECT=<your_wandb_project_name>
``` 

### Running the Project

#### Training a Model

You can train models with various configurations directly from the command line.

```bash
python main.py model=gpt-neox run=basic-training dataset=squad task=train group="test" tags=[gpt-neox] run.batch_size=64
```

### Data Sources and Processing

The datasets used in this project were obtained using the `datasets` library from Hugging Face. Specifically, the SQuAD v2 dataset was used.


### Data Processing

To preprocess the dataset for training, use the following command:

```bash
python main.py -m dataset=squad_v2 task=preprocess
```

This will apply the necessary preprocessing steps as defined in the `scripts/preprocess.py` script and the configuration files located in the `conf/` directory.

#### Evaluating a Model

To evaluate a trained model, use:

```bash
python main.py \
    -m model=gpt-neox \
    run=basic-evaluation \
    dataset=squad_v2 \
    task=evaluate \
    group="evaluation" \
    tags=[evaluate_best] \
    run.eval_dataset=test \
    model.checkpoint=./models/checkpoints/best_model_gpt-neox/
```

## Additional Features


### Supported Models

- GPT-NeoX
- Mamba (multiple configurations)
- Bidirectional Mamba
- Pythia (multiple configurations)
