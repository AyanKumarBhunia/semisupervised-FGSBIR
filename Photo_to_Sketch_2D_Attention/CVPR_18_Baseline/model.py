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
from dataset import get_imageOnly_dataloader, get_sketchOnly_dataloader
from rasterize import rasterize_relative, to_stroke_list
import math
from rasterize import batch_rasterize_relative
from base_model import Photo2Sketch_Base
from torchvision.utils import save_image

class Photo2Sketch(Photo2Sketch_Base):
    def __init__(self, hp):

        Photo2Sketch_Base.__init__(self, hp)
        self.train_params = self.parameters()
        self.main_optimizer = optim.Adam(self.train_params, hp.learning_rate)


    def Image2Sketch_Train(self, rgb_image, sketch_vector, length_sketch, step):

        self.train()
        self.main_optimizer.zero_grad()

        """ Encoding the Input """
        sketch_encoded_dist = self.Sketch_Encoder(sketch_vector, length_sketch)
        sketch_encoded_z_vector = sketch_encoded_dist.rsample()

        rgb_encoded_dist = self.Image_Encoder(rgb_image)
        rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

        """ Ditribution Matching Loss """
        prior_distribution = torch.distributions.Normal(torch.zeros_like(sketch_encoded_dist.mean),
                                                        torch.ones_like(sketch_encoded_dist.stddev))
        kl_cost_1 = torch.distributions.kl_divergence(sketch_encoded_dist, prior_distribution).sum()
        kl_cost_2 = torch.distributions.kl_divergence(rgb_encoded_dist, prior_distribution).sum()


        ##############################################################
        """ Cross Modal the Decoding """
        ##############################################################

        """ a) Photo to Sketch """
        start_token = torch.stack([torch.tensor([0, 0, 1, 0, 0])] *rgb_image.shape[0]).unsqueeze(0).float().to(device)
        batch_init = torch.cat([start_token, sketch_vector], 0)
        z_stack = torch.stack([rgb_encoded_dist_z_vector] * (self.hp.max_seq_len + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        photo2sketch_output, _ = self.Sketch_Decoder(inputs, rgb_encoded_dist_z_vector, length_sketch + 1)

        end_token = torch.stack([torch.tensor([0, 0, 0, 0, 1])] * rgb_image.shape[0]).unsqueeze(0).to(device).float()
        batch = torch.cat([sketch_vector, end_token], 0)
        x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim

        sup_p2s_loss = sketch_reconstruction_loss(photo2sketch_output, x_target)  #TODO: Photo to Sketch Loss


        """ b)  Sketch to Photo """
        cross_recons_photo = self.Image_Decoder(sketch_encoded_z_vector)
        # sup_s2p_loss = F.mse_loss(rgb_image, cross_recons_photo, reduction='sum')/rgb_image.shape[0] #TODO: Sketch 2 Photo Loss
        sup_s2p_loss = F.mse_loss(rgb_image, cross_recons_photo)

        ##############################################################
        """ Self Modal the Decoding """
        ##############################################################
        """ a) Photo to photo """
        self_recons_photo = self.Image_Decoder(rgb_encoded_dist_z_vector)
        # short_p2p_loss = F.mse_loss(rgb_image, self_recons_photo, reduction='sum')/rgb_image.shape[0]
        short_p2p_loss = F.mse_loss(rgb_image, self_recons_photo)

        """ a) Sketch to Sketch """
        start_token = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * rgb_image.shape[0]).unsqueeze(0).to(device).float()
        batch_init = torch.cat([start_token, sketch_vector], 0)
        z_stack = torch.stack([sketch_encoded_z_vector] * (self.hp.max_seq_len + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        sketch2sketch_output, _ = self.Sketch_Decoder(inputs, sketch_encoded_z_vector, length_sketch + 1)

        end_token = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * rgb_image.shape[0]).unsqueeze(0).to(device).float()
        batch = torch.cat([sketch_vector, end_token], 0)
        x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim

        short_s2s_loss = sketch_reconstruction_loss(sketch2sketch_output, x_target)  # TODO: Photo to Sketch Loss

        loss = sup_p2s_loss + sup_s2p_loss + short_p2p_loss + short_s2s_loss + 0.01*(kl_cost_1 + kl_cost_2)

        loss.backward()
        nn.utils.clip_grad_norm(self.train_params, self.hp.grad_clip)
        self.main_optimizer.step()


        if step%1000 == 0:

            """ Draw Photo to Sketch """
            start_token = torch.Tensor([0, 0, 1, 0, 0]).view(-1, 5).to(device)
            start_token = torch.stack([start_token] * rgb_encoded_dist_z_vector.shape[0], dim=1)
            state = start_token
            hidden_cell = None

            batch_gen_strokes = []
            for i_seq in range(self.hp.max_seq_len):
                input = torch.cat([state, rgb_encoded_dist_z_vector.unsqueeze(0)], 2)
                state, hidden_cell = self.Sketch_Decoder(input, rgb_encoded_dist_z_vector, hidden_cell=hidden_cell, isTrain=False,
                                               get_deterministic=True)
                batch_gen_strokes.append(state.squeeze(0))
            photo2sketch_gen = torch.stack(batch_gen_strokes, dim=1)

            """ Draw Sketch to Sketch """
            start_token = torch.Tensor([0, 0, 1, 0, 0]).view(-1, 5).to(device)
            start_token = torch.stack([start_token] * sketch_encoded_z_vector.shape[0], dim=1)
            state = start_token
            hidden_cell = None

            batch_gen_strokes = []
            for i_seq in range(self.hp.max_seq_len):
                input = torch.cat([state, sketch_encoded_z_vector.unsqueeze(0)], 2)
                state, hidden_cell = self.Sketch_Decoder(input, sketch_encoded_z_vector, hidden_cell=hidden_cell, isTrain=False,
                                               get_deterministic=True)
                batch_gen_strokes.append(state.squeeze(0))
            sketch2sketch_gen = torch.stack(batch_gen_strokes, dim=1)

            sketch_vector_gt = sketch_vector.permute(1, 0, 2)

            sketch_vector_gt_draw = batch_rasterize_relative(sketch_vector_gt).to(device)
            photo2sketch_gen_draw = batch_rasterize_relative(photo2sketch_gen).to(device)
            sketch2sketch_gen_draw = batch_rasterize_relative(sketch2sketch_gen).to(device)

            batch_redraw = []
            for a, b, c, d, e ,f in zip(sketch_vector_gt_draw, rgb_image, photo2sketch_gen_draw, sketch2sketch_gen_draw, self_recons_photo, cross_recons_photo):
                batch_redraw.append(torch.cat((1. - a, b, 1. - c, 1. - d, e, f), dim=-1))

            torchvision.utils.save_image(torch.stack(batch_redraw), './Redraw_Photo2Sketch/redraw_{}.jpg'.format(step),
                                         nrow=6)


        return sup_p2s_loss, sup_s2p_loss, short_p2p_loss, short_s2s_loss, kl_cost_1, kl_cost_2, loss



