# EEG TCNTransformer

##### Core idea: EEGNet + TCN + self-attention

## Abstract
![Network Architecture](/visualization/Fig1.png)

EEG-TCNTransformer, inspired by EEG-TCNet and EEG-Conformer. EEG-TCNTransformer has the advantage of learning the low-level temporal features and spatial features by the novel convolution architecture from EEG-TCNet and extracting the global temporal correlation from the self-attention mechanism.

### Installing
Navigate to EEG-Conformer's main folder and create the environment using Miniconda3:
```
$ conda create -n tcntransformer python=3.10
$ conda activate tcntransformer 
$ pip install -r requirements.txt
```


## Datasets
Please use consistent train-val-test split when comparing with other methods.
- [BCI_competition_IV2a](https://www.bbci.de/competition/iv/) - acc 82.97%


