import torch

# 加载 .pth 文件
model_path = '/aifs4su/data/zheny/fairseq/vae_v2/codec/speech_ckpt/hubert_1k_data/ckpt_00300000.pth'
model_dict = torch.load(model_path)

new_model_dict=model_dict['codec_model']

# new_model_dict['quantizer.vq.layers.8._codebook.inited']

keys_to_remove = [key for key in new_model_dict.keys() if any(layer in key for layer in ['vq.layers.8', 'vq.layers.9', 'vq.layers.10', 'vq.layers.11'])]

# 删除这些参数
for key in keys_to_remove:
    del new_model_dict[key]

 
del new_model_dict._metadata


# 保存修改后的模型
new_model_path = 'modified_model.pth'
torch.save(new_model_dict, new_model_path)

print(f"Modified model saved to {new_model_path}.")
