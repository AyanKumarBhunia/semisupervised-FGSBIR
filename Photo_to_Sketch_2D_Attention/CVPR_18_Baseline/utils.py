from matplotlib import pyplot as plt
import torch
use_cuda = True
from IPython.display import SVG, display
import numpy as np
import svgwrite
from six.moves import xrange
import math
import torch.nn as nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import torch
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""

  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)-1
  result = np.zeros((l+1, 3))
  result[:, 0:2] = big_stroke[0:l+1, 0:2]
  result[:, 2] = big_stroke[0:l+1, 3]
  result[-1, -1] = 1.
  return result

def get_bounds(data, factor=10):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)



def transfer_ImageNomralization(x, type='to_Gen'):
    #  https://discuss.pytorch.org/t/how-to-normalize-multidimensional-tensor/65304
    #to_Gen (-1, 1) vs to_Recog (ImageNet Normalize)
    if type == 'to_Gen':
        # First Unnormalize
        mean = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).to(device)
        std = torch.tensor([1/0.229, 1/0.224, 1/0.225]).to(device)
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        # Then Normalize Again
        mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).to(device)
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

    elif type == 'to_Recog':
        # First Unnormalize
        mean = torch.tensor([-1.0, -1.0, -1.0]).to(device)
        std = torch.tensor([1/0.5, 1/0.5, 1/0.5]).to(device)
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        # Then Normalize Again
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return x

def sample_next_state(output, hp, temperature =0.01):

    def adjust_temp(pi_pdf):
        pi_pdf = np.log(pi_pdf)/temperature
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = output
    # get mixture indices:
    o_pi = o_pi.data[0,:].cpu().numpy()
    o_pi = adjust_temp(o_pi)
    pi_idx = np.random.choice(hp.num_mixture, p=o_pi)
    # get pen state:
    o_pen = F.softmax(o_pen_logits, dim=-1)
    o_pen = o_pen.data[0,:].cpu().numpy()
    pen = adjust_temp(o_pen)
    pen_idx = np.random.choice(3, p=pen)
    # get mixture params:
    o_mu1 = o_mu1.data[0,pi_idx].item()
    o_mu2 = o_mu2.data[0,pi_idx].item()
    o_sigma1 = o_sigma1.data[0,pi_idx].item()
    o_sigma2 = o_sigma2.data[0,pi_idx].item()
    o_corr = o_corr.data[0,pi_idx].item()
    x,y = sample_bivariate_normal(o_mu1,o_mu2,o_sigma1,o_sigma2,o_corr, temperature = temperature, greedy=False)
    next_state = torch.zeros(5)
    next_state[0] = x
    next_state[1] = y
    next_state[pen_idx+2] = 1
    return next_state.to(device).view(1,1,-1), next_state


def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, temperature = 0.2, greedy=False):
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(temperature) #confusion
    sigma_y *= np.sqrt(temperature) #confusion
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], \
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

class Visualizer:
    def __init__(self, name = 'Photo2Sketch'):

        if os.path.exists('Tensorboard_' + name):
            shutil.rmtree('Tensorboard_' + name)

        self.writer = SummaryWriter('Tensorboard_' + name)

        self.mean = torch.tensor([-1.0, -1.0, -1.0]).to(device)
        self.std = torch.tensor([1 / 0.5, 1 / 0.5, 1 / 0.5]).to(device)

    def vis_image(self, visularize, step):
        for keys, value in visularize.items():
            #print(keys,value.size())
            value.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
            visularize[keys] = torchvision.utils.make_grid(value)
            self.writer.add_image('{}'.format(keys), visularize[keys], step)


    def plot_scalars(self, scalars, step):

        for keys, value in scalars.items():
            #print(keys,value.size())
            self.writer.add_scalar('{}'.format(keys), scalars[keys], step)