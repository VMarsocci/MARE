from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import json
from datetime import datetime
from PIL import ImageFilter
import pathlib
import yaml
from argparse import ArgumentParser
import os

from tqdm import tqdm, trange

from dataset import train_dataset
from MACUNet import MACUNet
from MAResUNet import MAResUNet
from early_stopping import EarlyStopping
from cp import *
from sce import *
from eval.metrics import *

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __str__(self):
        str_transforms = f"GaussianBlur(sigma={self.sigma})"
        return str_transforms

class JaccardLoss():
    def __init__(self, class_num = 6):
      self.class_num = class_num

    def __call__(self, semantic_image_pred, semantic_image):
        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)
        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
        _, _, jacc, _ = eval_metrics(semantic_image_pred.long(), semantic_image.long(), 
                              self.class_num, learnable = True)
        return nn.Parameter(jacc, requires_grad = True)

def get_args():
    parser = ArgumentParser(description = "Hyperparameters", add_help = True)
    parser.add_argument('-c', '--config-name', type = str, help = 'YAML Config name', dest = 'CONFIG', default = 'MARE')
    return parser.parse_args()


args = get_args()

project_root = "."
config_name = args.CONFIG
config_path = 'config/'+config_name
default_dst_dir = str(pathlib.Path(project_root) / "experiments")

exp_directory = pathlib.Path(default_dst_dir) / config_name
os.makedirs(exp_directory, exist_ok=True)

# Load the configuration params of the experiment
full_config_path = pathlib.Path(project_root) / (config_path + ".yaml")
print(f"Loading experiment {full_config_path}")
with open(full_config_path, "r") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)
exp_dir = exp_directory

print(f"Logs and/or checkpoints will be stored on {exp_directory}")

pretrained = exp_config['model']['pretraining_arch']
CHECKPOINTS = exp_config['model']['checkpoints_path']

batch_size = exp_config['data']['train']['batch_size']
niter = exp_config['optim']['num_epochs']
class_num = exp_config['model']['num_classes']

learning_rate = exp_config['optim']['lr']
end_learning_rate = exp_config['optim']['end_lr']
betas = exp_config['optim']['beta']
weight_decay = exp_config['optim']['weight_decay']

cuda = True
num_workers = 4
num_GPU = 1
torch.cuda.set_device(0)

size_h = exp_config['data']['size'][0]
size_w = exp_config['data']['size'][1]
flip = 0
band = exp_config['data']['channels']

train_path = exp_config['data']['train']['path']
val_path = exp_config['data']['val']['path']
test_path = exp_config['data']['test']['path']

encoder, pretrained, _ = pretrain_strategy(pretrained, CHECKPOINTS)
net = MAResUNet(band, class_num, base_model= encoder)
net.name = exp_config['general']['test_id'] + '_' + exp_config['general']['test_type']

out_file = str(exp_dir) + '/' + net.name

index = exp_config['data']['train']['samples']
val_index = exp_config['data']['val']['samples']
test_index = exp_config['data']['test']['samples']


try:
    import os
    if os.path.exists(out_file):
      net.load_state_dict(torch.load(out_file+'/netG.pth'))
      print('Checkpoints successfully loaded!')
    else:
      os.makedirs(out_file)
      print('No checkpoints founds -> Directory created successfully')
except OSError:
    pass

manual_seed = np.random.randint(1, 10000)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
cudnn.benchmark = True

train_trans = T.Compose([
  T.ToPILImage(),
  T.RandomApply([T.ColorJitter(*exp_config['data']['train']['transforms']['cjitter'])], 
                 p= exp_config['data']['train']['transforms']['cjitter_p']),
  T.RandomGrayscale(p = exp_config['data']['train']['transforms']['gray_p']),
  T.RandomApply([GaussianBlur(exp_config['data']['train']['transforms']['gaussian_blur'])], 
                 p = exp_config['data']['train']['transforms']['gaussian_blur_p']),
  T.ToTensor(),
  T.Normalize(*exp_config['data']['train']['transforms']['normalization'])
])

print(train_trans)

val_trans = T.Compose([
  T.ToTensor(),
  T.Normalize(*exp_config['data']['val']['transforms']['normalization'])
])

train_dataset_ = train_dataset(train_path, size_w, size_h, 
                               flip, band, batch_size, transform = train_trans)

val_dataset_ = train_dataset(val_path, size_w, size_h, 0, 
                             band, transform =val_trans)

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if cuda:
    net.cuda()
if num_GPU > 1:
    net = nn.DataParallel(net)


###########   LOSS & OPTIMIZER   ##########
if exp_config['model']['loss'] == 'crossentropy':
    criterion = nn.CrossEntropyLoss(ignore_index=255)
elif exp_config['model']['loss'] == 'softcrossentropy':
    criterion = SoftCrossEntropyLoss(smooth_factor= 0.1, n_classes = class_num, ignore_index=255)
elif exp_config['model']['loss'] == 'jaccard':
    criterion = JaccardLoss(class_num = 6)
else:
    print('Loss not implemented yet. Cross Entropy selected by default')
    criterion = nn.CrossEntropyLoss(ignore_index=255)

if exp_config['optim']['optim_type'] == 'adamw':
    optimizer = torch.optim.AdamW(net.parameters(), 
                                  lr=learning_rate, 
                                  betas = betas,
                                  weight_decay=weight_decay)
elif exp_config['optim']['optim_type'] == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), 
                                  lr=learning_rate, 
                                  betas = betas,
                                  weight_decay=weight_decay)
else:
    print('Optim not implemented yet. Adam selected by default')


early_stopping = EarlyStopping(patience=exp_config['optim']['patience'], 
                               verbose=True)

if __name__ == '__main__':
    start = time.time()
    track_metrics = {}
    track_lr = []
    track_loss = []
    track_OA = []
    track_acc = []
    track_miou = []
    track_f1 = []
    net.train()
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=learning_rate * 0.01, last_epoch=-1)
    for epoch in range(1, niter + 1):
        sum_loss = 0
        train_iter = train_dataset_.data_iter_index(index=index)
        for param_group in optimizer.param_groups:
            track_lr.append(param_group['lr'])
            print("Epoch: %s" % epoch, " - Learning rate: ", param_group['lr'])

        for initial_image, semantic_image in tqdm(train_iter, desc='train'):
            initial_image = initial_image.cuda()
            semantic_image = semantic_image.cuda()

            semantic_image_pred = net(initial_image)

            loss = criterion(semantic_image_pred, semantic_image.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.detach().cpu().numpy()*initial_image.shape[0]
        
        epoch_loss = sum_loss/index
        track_loss.append(epoch_loss)
        print("Training loss: ", epoch_loss)

        lr_adjust.step()
        

        with torch.no_grad():
            net.eval()
            val_iter = val_dataset_.data_iter_index(index=val_index)
            metrics = []

            for initial_image, semantic_image in tqdm(val_iter, desc='val'):
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image).detach() 

                semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                semantic_image_pred = semantic_image_pred.argmax(dim=0)
                semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

                metric = eval_metrics(semantic_image_pred.long(), semantic_image.long(), class_num)

        metrics.append(metric)

        m_arr = np.mean(np.array(metrics), axis = 0)
        # track_metrics[epoch] = list(m_arr)
        track_OA.append(m_arr[0])
        track_acc.append(m_arr[1])
        track_miou.append(m_arr[2])
        track_f1.append(m_arr[3])
        print("Val metrics:")
        print("OA:", m_arr[0], '\tACC_per_Class:', m_arr[1], "\tmIoU:", m_arr[2], "\tF1:", m_arr[3])
        net.train()

        early_stopping(1 - m_arr[2], net, '%s/' % out_file + 'netG.pth')

        if early_stopping.early_stop:
            break
            
        print()

    end = time.time()

    print('Training completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

    test_datatset_ = train_dataset(test_path, time_series=band, transform = val_trans)
    start = time.time()
    test_iter = test_datatset_.data_iter_index(index = test_index)
    if os.path.exists('%s/' % out_file + 'netG.pth'):
        net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))
        print("Checkpoints correctly loaded: ", out_file)

    net.eval()

    for initial_image, semantic_image in tqdm(test_iter, desc='test'):
        initial_image = initial_image.cuda()
        semantic_image = semantic_image.cuda()

        semantic_image_pred = net(initial_image).detach()
        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)

        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

        m_arr = eval_metrics(semantic_image_pred.long(), semantic_image.long(), class_num)
        
        image = semantic_image_pred

    end = time.time()
    print('Test completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    print("OA:", m_arr[0], '\tACC_per_Class:', m_arr[1], "\tmIoU:", m_arr[2], "\tF1:", m_arr[3])
    
    track_metrics['train_loss'] = track_loss
    track_metrics['train_val_OA'] = track_OA
    track_metrics['train_val_acc_per_class'] = track_acc
    track_metrics['train_val_miou'] = track_miou
    track_metrics['train_val_f1'] = track_f1
    track_metrics['lr'] = track_lr
    track_metrics['test'] = m_arr

    with open(out_file+'/track_metrics.json', 'w') as outfile:
      json.dump(track_metrics, outfile)