import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from PIL import ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pathlib

from eval.metrics import *

def plot_grad_flow(named_parameters, epoch, graph_directory):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    plt.figure(figsize=(25,10))
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=2, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=2, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=3, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=6),
                Line2D([0], [0], color="b", lw=6),
                Line2D([0], [0], color="k", lw=6)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(pathlib.Path(graph_directory) / str(epoch))
    plt.close()


#### CUSTOM GAUSSIAN BLUR FOR PYTORCH COMPATIBILITY
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


#### CUSTOM LOSSES

class CustomLoss():
    def __init__(self, class_num = 6, loss_name = 'jaccard'):
      self.class_num = class_num
      self.loss_name = loss_name

    def __call__(self, semantic_image_pred, semantic_image):
        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)
        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
        _, _, jacc, dice = eval_metrics(semantic_image_pred.long(), semantic_image.long(), 
                              self.class_num, learnable = True)
        if self.loss_name == 'jaccard':
            return nn.Parameter(jacc, requires_grad = True)
        elif self.loss_name == 'dice':
            return nn.Parameter(dice, requires_grad = True)
        else:
        	print("Loss not implemented yet. Dice selected.")
        	return nn.Parameter(dice, requires_grad = True)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)