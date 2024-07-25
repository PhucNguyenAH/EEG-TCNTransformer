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
- [BCI_competition_IV2a](https://bnci-horizon-2020.eu/database/data-sets) - acc 82.97%

Download the official BCI Competition IV (from A01T, A01E to A09T, A09E) and organize the downloaded files as follows:
``` 
BCIIV2a
│── A01T.mat
│── A01E.mat
│── A02T.mat
│── A02E.mat
│── A03T.mat
│── A03E.mat
│── A04T.mat
│── A04E.mat
│── A05T.mat
│── A05E.mat
│── A06T.mat
│── A06E.mat
│── A07T.mat
│── A07E.mat
│── A08T.mat
│── A08E.mat
│── A09T.mat
│── A09E.mat
```

## Train
Training with the following scripts:
```
python TCNTransformer.py
```