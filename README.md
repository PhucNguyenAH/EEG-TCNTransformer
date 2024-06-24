# EEG-Conformer

### EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization [[Paper](https://ieeexplore.ieee.org/document/9991178)]
##### Core idea: spatial-temporal conv + pooling + self-attention

### News
🎉🎉🎉 We've joined in [braindecode](https://braindecode.org/stable/index.html) toolbox. Use [here](https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html) for detailed info.


Thanks to [Bru](https://github.com/bruAristimunha) and colleagues for helping with the modifications.

## Abstract
![Network Architecture](/visualization/Fig1.png)

```

### Installing
Navigate to EEG-Conformer's main folder and create the environment using Miniconda3:
```
$ conda create -n tcntransformer python=3.10
$ conda activate tcntransformer 
$ pip install -r requirements.txt
```


## Datasets
Please use consistent train-val-test split when comparing with other methods.
- [BCI_competition_IV2a](https://www.bbci.de/competition/iv/) - acc 78.66% (hold out)


