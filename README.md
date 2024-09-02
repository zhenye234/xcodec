
[![arXiv](https://img.shields.io/badge/arXiv-2408.17175-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2408.17175)  
# X-Codec

Unified  Semantic and Acoustic Codec  for Audio Language Model.

# Paper 
 

**Title**: Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model

**Authors**: Zhen Ye, Peiwen Sun, Jiahe Lei, Hongzhan Lin, Xu Tan, Zheqi Dai, Qiuqiang Kong, Jianyi Chen, Jiahao Pan, Qifeng Liu, Yike Guo*, Wei Xue*

<img src="fig1.png" alt="Overview" width="600"/>

# Experiments on VALL-E
<img src="exp.png" alt="Exp" width="900"/>

<!-- # ckpts -->

<!-- Speech ckpts [downlaod link](https://drive.google.com/file/d/1oF1_R0Z2JNnqdPbuqiL8tJeY6pDwuQG1/view?usp=sharing)
 
General audio ckpts [Soon] -->

## Available models
🤗 links to the Huggingface model hub.

| Model name                                  | Hugging Face                                                                                           | Config                                                                                                   | Semantic Model                                                        | Domain        | Training Data                 |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|---------------|-------------------------------|
| xcodec_hubert_librispeech                   | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_speech_hubert_librispeech.pth)            | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert.yaml)                                | [🤗 Hubert-base](https://huggingface.co/facebook/hubert-base-ls960)               | Speech        | Librispeech                   |
| xcodec_wavlm_mls (not mentioned in paper)   | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_speech_wavlm_mls.pth)                     | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_wavlm.yaml)                                 | [🤗 Wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)                | Speech        | MLS English                   |
| xcodec_wavlm_more_data (not mentioned in paper) | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_speech_wavlm_more_data.pth)               | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_wavlm.yaml)                                 | [🤗 Wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)                | Speech        | MLS English + Internal data   |
| xcodec_hubert_general_audio                 | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_general.pth)                              | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert%20_general.yaml)                     | [🤗Hubert-base-general-audio](https://huggingface.co/ZhenYe234/hubert_base_general_audio)      | General audio | 200k hours internal data      |
| xcodec_hubert_general_audio_more_data (not mentioned in paper) | Coming Soon | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert%20_general.yaml) | [🤗](https://huggingface.co/ZhenYe234/hubert_base_general_audio) | General audio | More balanced data            |





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

```bibtex
@article{ye2024codecdoesmatterexploring,
      title={Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model}, 
      author={Zhen Ye and Peiwen Sun and Jiahe Lei and Hongzhan Lin and Xu Tan and Zheqi Dai and Qiuqiang Kong and Jianyi Chen and Jiahao Pan and Qifeng Liu and Yike Guo and Wei Xue},
      journal={arXiv preprint arXiv:2408.17175},
      year={2024},
}
```