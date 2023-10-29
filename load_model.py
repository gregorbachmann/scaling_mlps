import json
import torch

from models import get_architecture
from utils.parsers import get_finetune_parser
from utils.config import config_to_name, model_from_config


# Path to pre-trained checkpoints and the corresponding config file (e.g. the checkpoints downloaded from given link)
checkpoint_path = './path/to/checkpoint'
config_path = './path/to/config.txt'

model, architecture, crop_resolution, norm = model_from_config(checkpoint_path)
with open(config_path, 'r') as f:
        config = json.load(f)

model = get_architecture(**config).cuda()
print('Loading checkpoint', checkpoint_path)

params = {
    k: v
    for k, v in torch.load(checkpoint_path).items()
}

# Load pre-trained parameters 
print('Load_state output', model.load_state_dict(params, strict=False))