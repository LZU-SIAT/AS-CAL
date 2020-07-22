# AS_CAL

## Introduction
This is the official implementation of "Augmented Skeleton Based Contrastive Action Learning with Momentum LSTM for Unsupervised Action Recognition". 
## Requirements
- Python 3.6
- pytorch 1.0.1
## Datasets
- NTU RGB+D 60:  
Use `st-gcn/tools/ntu_gendata.py` in https://github.com/yysijie/st-gcn to prepare data
- NTU RGB+D 120
same as NTU 60 but needs some modification on NTU 120.
- SBU
- UWA3D
- N-UCLA


## Usage
- pretrain and then linear evaluation:  
  python  pretrain_and_linEval.py

- eload and linear:  
python linEval.py --mode eval --model_path ./pretrained_model.pth

- supervised:  
python linEval.py --mode supervise

- reload and semi-supervised:  
python linEval.py --mode semi --model_path ./pretrained_model.pth

For more customized parameters setting, you can change them in parse_option(). 
## License
AS-CAL is released under the MIT License.
