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

from tqdm import tqdm, trange

from dataset import train_dataset
from MACUNet import MACUNet
from MAResUNet import MAResUNet
from early_stopping import EarlyStopping
from cp import *
from sce import *
from eval.metrics import *

pretrained = 'imagenet'
CHECKPOINTS = '../U-Net_08_03/imagenet_weights'
batch_size = 128
niter = 20
class_num = 6
learning_rate = 0.0001 * 3
beta1 = 0.5
cuda = True
num_workers = 1
size_h = 256
size_w = 256
flip = 0
band = 3
train_path = '../Vaihingen/dataset/training/'
val_path = '../Vaihingen/dataset/validation/'
test_path = '../Vaihingen/test'

encoder, pretrained, _ = pretrain_strategy(pretrained, CHECKPOINTS)
net = MAResUNet(band, class_num, base_model= encoder)
net.name = datetime.today().strftime('%Y_%m_%d') + '_' + pretrained

out_file = './checkpoint/' + net.name
num_GPU = 1
index = 6000
torch.cuda.set_device(0)

try:
    import os
    if os.path.exists(out_file):
      net.load_state_dict(torch.load(out_file+'/netG.pth'))
      print('Checkpoints successfully loaded!')
    else:
      os.makedirs(out_file)
except OSError:
    pass

manual_seed = np.random.randint(1, 10000)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
cudnn.benchmark = True

train_trans = T.Compose([
  T.ToPILImage(),
  T.RandomApply([T.ColorJitter((0.4, 1.5), (0.2, 2), (0.2, 2), (-0.3, 0.3))], p=0.5),
  T.RandomGrayscale(p = 0.3),
  T.GaussianBlur(3),
  T.ToTensor(),
  T.Normalize([0.3206467284225958, 0.3230442137290628, 0.4726621112943237], 
  [0.1507140017014572, 0.15590631427820206, 0.21559699547025785])
])

print(train_trans)

val_trans = T.Compose([
  T.ToTensor(),
  T.Normalize([0.3156378902694481, 0.31905504344848745, 0.4772593356486166],
  [0.14555519829539584, 0.15285215419466583, 0.21367692564739332])
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
# criterion = nn.CrossEntropyLoss(ignore_index=255)
criterion = SoftCrossEntropyLoss(smooth_factor= 0.1, n_classes = class_num, ignore_index=255)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=10, verbose=True)

if __name__ == '__main__':
    start = time.time()
    track_metrics = {}
    net.train()
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=learning_rate * 0.01, last_epoch=-1)
    for epoch in range(1, niter + 1):
        for iter_num in trange(index//batch_size, desc='train, epoch:%s' % epoch):
            train_iter = train_dataset_.data_iter_index(index=index)
            for initial_image, semantic_image in train_iter:
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image)

                loss = criterion(semantic_image_pred, semantic_image.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        lr_adjust.step()

        with torch.no_grad():
            net.eval()
            # val_iter = val_dataset_.data_iter_index(index=index)
            val_iter = val_dataset_.data_iter()
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
        track_metrics[epoch] = list(m_arr)
        print("OA:", m_arr[0], '\tACC_per_Class:', m_arr[1], "\tmIoU:", m_arr[2], "\tF1:", m_arr[3])

        net.train()

        early_stopping(1 - m_arr[2], net, '%s/' % out_file + 'netG.pth')

        if early_stopping.early_stop:
            break

    end = time.time()

    with open(out_file+'/val_metrics.json', 'w') as outfile:
      json.dump(track_metrics, outfile)

    print('Training completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

    test_datatset_ = train_dataset(test_path, time_series=band, transform =val_trans)
    start = time.time()
    test_iter = test_datatset_.data_iter()
    # test_iter = test_datatset_.data_iter_index(index = index)
    if os.path.exists('%s/' % out_file + 'netG.pth'):
        net.load_state_dict(torch.load('%s/' % out_file + 'netG.pth'))

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
    # print(metric)