# AS-CAL

## Introduction
This is the official implementation of "Augmented Skeleton Based Contrastive Action Learning with Momentum LSTM for Unsupervised Action Recognition". 
## Requirements
- Python 3.6
- Pytorch 1.0.1
## Datasets
- NTU RGB+D 60:  
Download raw data from https://github.com/shahroudy/NTURGB-D  
Use `st-gcn/tools/ntu_gendata.py` in https://github.com/yysijie/st-gcn to prepare data
- NTU RGB+D 120:  
Same as NTU RGB+D 60 but needs some modification for NTU RGB+D 120.
- SBU, UWA3D, N-UCLA  
Unzip the `.zip` file in `/data` and put them into the directory corresponding to the one in codes.



## Usage
- pretrain and then linear evaluation:  
  `python  pretrain_and_linEval.py`

- reload pre-trained models and linear evaluation:  
`python linEval.py --mode eval --model_path ./pretrained_model.pth`

- supervised:  
`python linEval.py --mode supervise`

- reload pre-trained models and semi-supervised:  
`python linEval.py --mode semi --model_path ./pretrained_model.pth`

For more customized parameter settings, you can change them in `parse_option()` and/or `parse_option_lin_eval()` 

## Tips  
To debug, we suggest to set `epochs` to 2 and `save_freq` to 1 in `parse_option()`.  
Besides, we suggest to set `epochs` to 1 in `parse_option_lin_eval()`.

## License
AS-CAL is released under the MIT License.

## Citation  
@article{RAO202190,  
title = {Augmented Skeleton Based Contrastive Action Learning with Momentum LSTM for Unsupervised Action Recognition},  
journal = {Information Sciences},  
volume = {569},  
pages = {90-109},  
year = {2021},  
doi = {https://doi.org/10.1016/j.ins.2021.04.023},  
author = {Haocong Rao and Shihao Xu and Xiping Hu and Jun Cheng and Bin Hu},  
}
