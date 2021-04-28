# from pl_bolts.models.self_supervised import AMDIM
# import pytorch_lightning as pl

import torch
t = torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def pretrain_strategy(pretrained, cp_path, arch = 'resnet18'):
  if pretrained == 'obow':
    cp = torch.load(cp_path)['network']
    n_classes = len(cp[list(cp.keys())[-1]])
    encoder = models.__dict__[arch](num_classes = n_classes)
    encoder.load_state_dict(cp)
  
  elif pretrained == 'byol':
    cp = torch.load(cp_path)
    byol_filtered = {}
    for key in list(cp['state_dict'].keys()):
      if 'target_encoder' in key:
        if 'projector' not in key:
          byol_filtered[key.split('.', 3)[3]] = cp['state_dict'][key]

    n_classes = len(byol_filtered[list(byol_filtered.keys())[-1]])
    encoder = models.__dict__[arch](num_classes = n_classes)
    encoder.load_state_dict(byol_filtered)

  elif pretrained == 'imagenet':
    # encoder = models.__dict__[arch](pretrained = True)
    encoder = models.__dict__[arch]()
    encoder.load_state_dict(torch.load(cp_path))
    n_classes = list(encoder.children())[-1].out_features

  elif pretrained == 'amdim':
    cp = torch.load(cp_path)
    amdim_filtered = {}
    for key in list(cp['state_dict'].keys()):
      new_key = key.split('.', 1)[1]
      amdim_filtered[new_key] = cp['state_dict'][key]
    
    encoder = models.__dict__[arch]()
    n_classes = encoder.fc.in_features
    encoder.fc = nn.Sequential()
    encoder.load_state_dict(amdim_filtered)

  elif pretrained == None:
    encoder = models.__dict__[arch]()
    n_classes = list(encoder.children())[-1].out_features
    pretrained = 'no_pretraining'

  else:
    raise Exception('Pretrained strategy not supported')
    
  print('Encoder selected: ' + pretrained)

  return encoder, pretrained, n_classes