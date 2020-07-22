# AS-CAL

## Introduction
This is the official implementation of "Augmented Skeleton Based Contrastive Action Learning with Momentum LSTM for Unsupervised Action Recognition". 
## Requirements
- Python 3.6
- pytorch 1.0.1
## Datasets
- NTU RGB+D 60:  
Download raw data from https://github.com/shahroudy/NTURGB-D  
Use `st-gcn/tools/ntu_gendata.py` in https://github.com/yysijie/st-gcn to prepare data
- NTU RGB+D 120:  
Same as NTU RGB+D 60 but needs some modification for NTU 120.
- SBU, UWA3D, N-UCLA  
Download：https://share.weiyun.com/gCfTvj0Y password：ktp4x8



## Usage
- pretrain and then linear evaluation:  
  `python  pretrain_and_linEval.py`

- load and linear:  
`python linEval.py --mode eval --model_path ./pretrained_model.pth`

- supervised:  
`python linEval.py --mode supervise`

- reload and semi-supervised:  
`python linEval.py --mode semi --model_path ./pretrained_model.pth`

For more customized parameters setting, you can change them in parse_option(). 
## License
AS-CAL is released under the MIT License.
