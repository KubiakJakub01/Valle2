# Valle2

Implementation of a Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) model inspired by the VALL-E X architecture.

## Overview

This project provides tools and scripts to train TTS and potentially ASR models based on the principles outlined in the VALL-E X paper. It aims to replicate parts of the architecture and offer a framework for experimenting with speech synthesis and recognition using similar techniques.

## Project Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KubiakJakub01/Valle2.git
    cd Valle
    ```

2.  **Install dependencies:**
    Ensure you have Poetry installed. If not, follow the instructions [here](https://python-poetry.org/docs/#installation).
    ```bash
    poetry install
    ```
    This command creates a virtual environment and installs all necessary packages specified in `pyproject.toml`.

3.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

## Workflow: LJSpeech Example

This section outlines the steps for preparing the LJSpeech dataset, training a model, and synthesizing audio.

### 1. Data Preparation (LJSpeech)

Prepare the LJSpeech dataset by following these steps:
   - Download LJSpeech dataset: `wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2`
   - Unzip the dataset.
   - Run the preprocessing script:
   ```bash
   poetry run python -m valle.scripts.prepare_dataset -h
   ```

### 2. Configs

Create a config file for ValleAR and ValleNAR in `.json` format. For more information see class [ConfigValle](valle/config.py).

### 3. Training

Train the model by following these steps:
```bash
poetry run python -m valle.scripts.train -h
```

### 4. Synthesis

Synthesize speech by following these steps:
```bash
poetry run python -m valle.scripts.synthesize -h
```

## Current Status & Future Work

### Done:

-   [x] ValleAR and ValleNAR implementation.
-   [x] Training loop implementation.
-   [x] Synthesis loop implementation.
-   [ ] Add support for other datasets (e.g., VCTK, LibriTTS).
-   [ ] Evaluation scripts (e.g., MOS, WER calculation).
-   [ ] Pre-trained model release.
-   [ ] Improved documentation and examples.
-   [ ] Code optimization and performance improvements.
-   [ ] Integration with MLOps tools (e.g., Weights & Biases).

### Citations

```bibtex
@misc{wang2023neuralcodeclanguagemodels,
      title={Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers}, 
      author={Chengyi Wang and Sanyuan Chen and Yu Wu and Ziqiang Zhang and Long Zhou and Shujie Liu and Zhuo Chen and Yanqing Liu and Huaming Wang and Jinyu Li and Lei He and Sheng Zhao and Furu Wei},
      year={2023},
      eprint={2301.02111},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2301.02111}, 
}
```

```bibtex
@misc{chen2024valle2neuralcodec,
      title={VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers}, 
      author={Sanyuan Chen and Shujie Liu and Long Zhou and Yanqing Liu and Xu Tan and Jinyu Li and Sheng Zhao and Yao Qian and Furu Wei},
      year={2024},
      eprint={2406.05370},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.05370}, 
}
```
