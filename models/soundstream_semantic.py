 

from typing import Sequence, Optional, Union
import sys

import math
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import descriptaudiocodec.dac.model.dac  as dac2
 
 
from quantization  import ResidualVectorQuantizer 
from transformers import  AutoModel
  
from modules.semantic_module import Encoder,Decoder
 

 
class SoundStream(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2], #  downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        normalize: bool = False,
        causal: bool = False,
        semantic_techer: str = 'hubert_base_general'
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
 
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios)) # 50 Hz
        self.bits_per_codebook = int(math.log2(bins)) # 1024 => 10
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        self.encoder = dac2.Encoder(64,ratios,D)
 
        self.encoder_semantic = Encoder(input_channels=768,encode_channels=768)
        self.decoder_semantic = Decoder(code_dim=768,output_channels=768,decode_channels=768)
        # out_D=D+768
        self.quantizer = ResidualVectorQuantizer(dimension=D+768, n_q=n_q, bins=bins)
 
        self.decoder_2 = dac2.Decoder(D,1024,ratios,)

        if semantic_techer=='hubert_base':
            self.semantic_model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
        elif semantic_techer=='wavlm_base_plus':
            self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
        elif semantic_techer=='hubert_base_general':
            
            self.semantic_model = AutoModel.from_pretrained("ZhenYe234/hubert_base_general_audio")
        self.semantic_model.eval()
         
        self.fc_prior = nn.Linear(D+768, D+768 )
 
        self.fc_post1= nn.Linear( D+768, 768 )
        self.fc_post2= nn.Linear( D+768,  D)

    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
    def calculate_rec_loss(self, rec, target):  
 
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()
 
        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x ):
        x= x[:,0,:]
        x = F.pad(x, (160, 160))
        target = self.semantic_model(x, output_hidden_states=True) .hidden_states
        target = torch.stack(target, dim=1)#.transpose(-1, -2)#.flatten(start_dim=1, end_dim=2)
        
        # average for all layers
        target = target.mean(1)   
        # target = target[9]
        return target

 

    def forward(self, x: torch.Tensor, bw: int):
        x=x.unsqueeze(1)
        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)
 
 
        e= torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)

 
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)

        quantized_semantic = self.fc_post1(quantized.transpose(1, 2)).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
 
        o_semantic = self.decoder_semantic(quantized_semantic )
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(),o_semantic)

        return o, commit_loss, semantic_recon_loss,None
   
    def encode(self, x: torch.Tensor,target_bw: Optional[int] = None) -> torch.Tensor:
 
        bw = target_bw
 
        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)


        if e_acoustic.shape[2] != e_semantic.shape[2]:
            e_acoustic = self.encoder(F.pad(x[:,0,:], (160, 160)).unsqueeze(0)) 
 
        e= torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)

 
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode(codes)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        return o

# test
if __name__ == '__main__':
    soundstream = SoundStream(n_filters=32, D=256) 

    for i in range(10):
        print(f"Iter {i}: ")
        x = torch.rand(1, 1, 16000) 
        o, commit_loss, distill_loss,_= soundstream(x,soundstream.target_bandwidths[-1])
        print('output', o.shape)
