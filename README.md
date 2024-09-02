
[![arXiv](https://img.shields.io/badge/arXiv-2408.17175-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2408.17175)  
# X-Codec

Unified  Semantic and Acoustic Codec  for Audio Language Model.

# Paper 
 

**Title**: Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model

**Authors**: Zhen Ye, Peiwen Sun, Jiahe Lei, Hongzhan Lin, Xu Tan, Zheqi Dai, Qiuqiang Kong, Jianyi Chen, Jiahao Pan, Qifeng Liu, Yike Guo*, Wei Xue*

<img src="fig1.png" alt="Overview" width="600"/>

# Experiments on VALL-E
<img src="exp.png" alt="Exp" width="900"/>

# ckpts

Speech ckpts [downlaod link](https://drive.google.com/file/d/1oF1_R0Z2JNnqdPbuqiL8tJeY6pDwuQG1/view?usp=sharing)
 
General audio ckpts [Soon]

# Inference

```bash
python inference.py
```

# Training
```bash
torchrun --nnodes=1 --nproc-per-node=8 main_launch_vqdp.py
```

## Acknowledgement
I would like to extend a special thanks to authors of Uniaudio and DAC, since our code base is mainly borrowed from  [Uniaudio](https://github.com/yangdongchao/UniAudio/tree/main/codec) and [DAC](https://github.com/descriptinc/descript-audio-codec).

## Citation
If you find this repo helpful, please consider citing in the following format:
