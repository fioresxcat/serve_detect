import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pdb
import torch.nn as nn
import torch
from easydict import EasyDict
import os
import torch
from pathlib import Path
import yaml
from model_3d import *

if __name__ == '__main__':
    ckpt_path = 'ckpt_3d/exp2_no_mask/epoch=26-train_loss=0.000-val_loss=0.000-train_acc=1.000-val_acc=1.000.ckpt'
    state = torch.load(ckpt_path, map_location='cpu')
    state_dict = state['state_dict']
    with open(os.path.join(Path(ckpt_path).parent, 'config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    model = torch.hub.load('facebookresearch/pytorchvideo', config.model.version, pretrained=True)
    model.blocks[-1].proj = nn.Linear(in_features=2048, out_features=1, bias=True)

    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.replace('model.', '')] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    # torch.save(model.state_dict(), 'ckpt/exp4_ce_loss_less_regularized_cropped_data_320_128/epoch36.pt')

    imgs = torch.randn(2, 3, 15, 256, 256, dtype=torch.float)
    
    out = model(imgs)
    print('output shape: ', out.shape)

    torch.onnx.export(
        model,
        # {
        #     'imgs': imgs,
        # },
        imgs,
        'ckpt_3d/exp2_no_mask/serve_detect_ep26.onnx',
        input_names=['imgs'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={
            "imgs": {0: "batch_size"},
        }
    )
