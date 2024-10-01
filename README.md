
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

# Highlight

You can easily apply our approach to enhance any existing acoustic codec:

For example

```python
class Codec():
    def __init__(self):
        # Acoustic codec components
        self.encoder = Encoder(...)       # Acoustic encoder
        self.decoder = Decoder(...)       # Acoustic decoder
        self.quantizer = RVQ(...)         # Residual Vector Quantizer (RVQ)

        # Adding the semantic module
        self.semantic_model = AutoModel.from_pretrained(...)  # e.g., Hubert, WavLM

        # Adding Projector
        self.fc_prior = nn.Linear(...)     
        self.fc_post1 = nn.Linear(...)     
        self.fc_post2 = nn.Linear(...)     

    def forward(self, x, bw):
        # Encode the input acoustically and semantically
        e_acoustic = self.encoder(x)
        e_semantic = self.semantic_model(x)

        # Combine acoustic and semantic features
        combined_features = torch.cat([e_acoustic, e_semantic])

        # Apply prior transformation
        transformed_features = self.fc_prior(combined_features)

        # Quantize the unified  semantic and acoustic features
        quantized, codes, bandwidth, commit_loss = self.quantizer(transformed_features, bw)

        # Post-process the quantized features
        quantized_semantic = self.fc_post1(quantized)
        quantized_acoustic = self.fc_post2(quantized)

        # Decode the quantized acoustic features
        output = self.decoder(quantized_acoustic)



    def semantic_loss(self,semantic,quantized_semantic):
        return F.mse_loss(semantic,quantized_semantic)     
```
For more details, please refer to our code.

# Available models
🤗 links to the Huggingface model hub.

| Model name                                  | Hugging Face                                                                                           | Config                                                                                                   | Semantic Model                                                        | Domain        | Training Data                 |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|---------------|-------------------------------|
| xcodec_hubert_librispeech                   | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_speech_hubert_librispeech.pth)            | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert.yaml)                                | [🤗 Hubert-base](https://huggingface.co/facebook/hubert-base-ls960)               | Speech        | Librispeech                   |
| xcodec_wavlm_mls (not mentioned in paper)   | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_speech_wavlm_mls.pth)                     | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_wavlm.yaml)                                 | [🤗 Wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)                | Speech        | MLS English                   |
| xcodec_wavlm_more_data (not mentioned in paper) | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_speech_wavlm_more_data.pth)               | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_wavlm.yaml)                                 | [🤗 Wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)                | Speech        | MLS English + Internal data   |
| xcodec_hubert_general_audio                 | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_hubert_general_audio.pth)                              | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert_general.yaml)                     | [🤗Hubert-base-general-audio](https://huggingface.co/ZhenYe234/hubert_base_general_audio)      | General audio | 200k hours internal data      |
| xcodec_hubert_general_audio_more_data (not mentioned in paper) | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_hubert_general_audio_v2.pth) | [🤗](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert_general.yaml) | [🤗Hubert-base-general-audio](https://huggingface.co/ZhenYe234/hubert_base_general_audio) | General audio | More balanced data            |





# Inference

To run inference, first download the model and config from hugging face.

```bash
python inference.py
```

# Training
Prepare  the training_file and validation_file in config. The file should list the paths to your audio files:
```bash
/path/to/your/xxx.wav
/path/to/your/yyy.wav
...
```
Then:

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
