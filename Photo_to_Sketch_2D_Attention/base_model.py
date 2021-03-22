import torch.nn as nn
from Image_Networks import *
from Sketch_Networks import *
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
import torchvision
from dataset import get_sketchOnly_dataloader, get_dataloader
from rasterize import rasterize_relative, to_stroke_list
import math
from rasterize import batch_rasterize_relative



class Photo2Sketch_Base(nn.Module):

    def __init__(self, hp):
        super(Photo2Sketch_Base, self).__init__()
        self.Image_Encoder = EncoderCNN()
        # self.Image_Decoder = DecoderCNN()
        # self.Sketch_Encoder = EncoderRNN(hp)
        self.Sketch_Decoder = DecoderRNN2D(hp)
        self.hp = hp
        # self.apply(weights_init_normal)

    def freeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = False


    def Unfreeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = True





