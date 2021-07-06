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
import pathlib
import yaml
from argparse import ArgumentParser
import os
from functools import partial

from tqdm import tqdm, trange

from dataset import train_dataset
from MAResUNet import MAResUNet
from early_stopping import EarlyStopping
from cp import *
from optim import *
from utils import *
from eval.metrics import *

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

# if exp_config['general']['plot']:
#   print('Plotting enabled')
#   graph_directory = pathlib.Path(exp_directory) / 'graphs'
#   os.makedirs(graph_directory, exist_ok=True)
#   print("The gradient plots will be stored in ", graph_directory)

# if exp_config['general']['last_layer_debug']:
#   print('Gradient debugging enabled')
#   deb_directory = pathlib.Path(exp_directory) / 'last_layer'
#   os.makedirs(deb_directory, exist_ok=True)
#   print("The gradient plots will be stored in ", deb_directory)

batch_size = exp_config['data']['train']['batch_size']
niter = exp_config['optim']['num_epochs']
class_num = exp_config['model']['num_classes']

####TO DO: mettere gli argparse per questi due
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

manual_seed = 18
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

train_dataset_ = train_dataset(train_path, 
                             size_w, size_h, 
                             flip = 0, 
                             batch_size = batch_size,
                             transform = val_trans)

val_dataset_ = train_dataset(val_path, 
                             size_w, size_h, 
                             flip = 0, 
                             batch_size = 1,        #tenere 1
                             transform = val_trans)

if cuda:
    net.cuda()
if num_GPU > 1:
    net = nn.DataParallel(net)

########## NONLINEARITY ############
if exp_config['model']['nonlinearity'] == 'relu':
    nonlinearity = partial(F.relu, inplace=True)
if exp_config['model']['nonlinearity'] == 'leakyrelu':
    nonlinearity = partial(F.leaky_relu, inplace=True)

print('Non-linearity selected: ', exp_config['model']['nonlinearity'])

class_ignored = 0 #background

###########   LOSS    ##########
if exp_config['model']['loss'] == 'crossentropy':
    criterion = nn.CrossEntropyLoss(ignore_index=class_ignored)
    print('Class ignored:', class_ignored)
    # criterion = CrossEntropyLoss(ignore_index=255)
elif exp_config['model']['loss'] == 'weightedcrossentropy':
    weights = exp_config['model']['loss_weights']
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=class_ignored)
else:
    print('Loss not implemented yet. Cross Entropy selected by default')
    criterion = nn.CrossEntropyLoss(ignore_index=255)

print('Loss selected: ', exp_config['model']['loss'])

###########   OPTIMIZER AND SCHEDULER    ##########
optimizer = set_optimizer(exp_config['optim'], net)
print('Optimizer selected: ', exp_config['optim']['optim_type'])
lr_adjust = set_scheduler(exp_config['optim'], optimizer)
print('Scheduler selected: ', exp_config['optim']['lr_schedule_type'])
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
    for epoch in range(1, niter + 1):
        sum_loss = 0
        train_iter = train_dataset_.data_iter_index(index=index)
        for param_group in optimizer.param_groups:
            track_lr.append(param_group['lr'])
            print("Epoch: %s" % epoch, " - Learning rate: ", param_group['lr'])
        
        i = 0
        for initial_image, semantic_image in tqdm(train_iter, desc='train'):
            initial_image = initial_image.cuda()
            semantic_image = semantic_image.cuda()

            semantic_image_pred = net(initial_image)

            # if exp_config['general']['last_layer_debug']:
            #   if np.random.random_sample() > (1-exp_config['general']['p_debug']):
            #     # print('Saving graph')
            #     pred_batch = semantic_image_pred.detach().cpu().numpy()
            #     np.savez(str(deb_directory)+'/{}e_{}bs_sample.npz'.format(str(epoch), str(i)), 
            #             mask=pred_batch[-1], 
            #             final_conv=net.finalconv3.weight.detach().cpu().numpy())
            #     plt.figure(figsize=(15,10))
            #     plt.hist(pred_batch.flatten(), bins = 100)
            #     plt.title('{} epoch - {} batch size: mask distribution'.format(str(epoch), str(i)))
            #     plt.grid()
            #     plt.savefig(str(deb_directory)+'/{}e_{}bs_hist.png'.format(str(epoch), str(i)))
            #     plt.close()

            loss = criterion(semantic_image_pred, semantic_image.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.detach().cpu().numpy()*initial_image.shape[0]
            i += 1
        
        epoch_loss = sum_loss/index
        track_loss.append(epoch_loss)
        # if exp_config['general']['plot']:
        #      plot_grad_flow(net.named_parameters(), epoch, graph_directory)
        print("Training loss: ", epoch_loss)
        if exp_config['model']['gradient_clipping']:
          clips = exp_config['model']['gradient_clipping_values']
          for p in net.parameters():
               p.register_hook(lambda grad: torch.clamp(grad, clips[0], clips[1]))
          print('Gradient clipped between: ', clips)

        lr_adjust.step()
        

        with torch.no_grad():
            net.eval()
            val_iter = val_dataset_.data_iter_index(index=val_index)
            
            hist = torch.zeros((class_num, class_num)).to(device=device, dtype=torch.long)

            for initial_image, semantic_image in tqdm(val_iter, desc='val'):
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image).detach() 

                semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                semantic_image_pred = semantic_image_pred.argmax(dim=0)
                semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

                hist += fast_hist(semantic_image.flatten().type(torch.LongTensor), 
                                  semantic_image_pred.flatten().type(torch.LongTensor), 
                                 class_num)


        # m_arr = eval_metrics(hist)
        verb_m = eval_metrics(hist, verbose = True)
        OA = verb_m[0]
        m_acc = np.float64(np.mean(verb_m[1][1:]))
        miou = np.float64(np.mean(verb_m[2][1:]))
        m_f1= np.float64(np.mean(verb_m[3][1:]))
        track_OA.append(OA)
        track_acc.append(m_acc)
        track_miou.append(miou)
        track_f1.append(m_f1)
        print("Val metrics:")
        # print("OA:", m_arr[0], '\tACC_per_Class:', m_arr[1], "\tmIoU:", m_arr[2], "\tF1:", m_arr[3])
        print('ACC_per_Class: ', verb_m[1])
        # print('mIoU: ', np.mean(verb_m[2][1:]))
        print("OA:", OA, '\tACC_per_Class:', m_acc, "\tmIoU:", miou, "\tF1:", m_f1)
        net.train()

        early_stopping(1 - miou, net, '%s/' % out_file + 'netG.pth')

        if early_stopping.early_stop:
            break
            
        print()

    end = time.time()

    print('Training completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

    test_datatset_ = train_dataset(test_path, 
                             size_w, size_h, 
                             flip = 0, 
                             batch_size = 1,        #tenere 1
                             transform = val_trans)
    start = time.time()
    test_iter = test_datatset_.data_iter_index(index = test_index)
    if os.path.exists('%s/' % out_file + 'netG.pth'):
        net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))
        print("Checkpoints correctly loaded: ", out_file)

    net.eval()

    hist = torch.zeros((class_num, class_num)).to(device=device, dtype=torch.long)

    for initial_image, semantic_image in tqdm(test_iter, desc='test'):
        initial_image = initial_image.cuda()
        semantic_image = semantic_image.cuda()

        semantic_image_pred = net(initial_image).detach()
        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)

        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

        hist += fast_hist(semantic_image.flatten().type(torch.LongTensor), 
                          semantic_image_pred.flatten().type(torch.LongTensor), 
                          class_num)


    m_arr = eval_metrics(hist, verbose = True)

    end = time.time()
    print('Test completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    print("OA:", m_arr[0], '\nACC_per_Class:', m_arr[1], "\nmIoU:", m_arr[2], "\nF1:", m_arr[3])
    
    track_metrics['train_loss'] = track_loss
    track_metrics['train_val_OA'] = track_OA
    track_metrics['train_val_acc_per_class'] = track_acc
    track_metrics['train_val_miou'] = track_miou
    track_metrics['train_val_f1'] = track_f1
    # track_metrics['lr'] = track_lr
    # track_metrics['test'] = m_arr

    with open(out_file+'/track_metrics.json', 'w') as outfile:
      json.dump(track_metrics, outfile)
