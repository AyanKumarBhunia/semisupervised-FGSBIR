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
from dataset import get_sketchOnly_dataloader
from rasterize import rasterize_relative, to_stroke_list
import math
from rasterize import batch_rasterize_relative
from base_model import Photo2Sketch_Base
from torchvision.utils import save_image
import os


class Photo2Sketch(Photo2Sketch_Base):
    def __init__(self, hp):

        Photo2Sketch_Base.__init__(self, hp)
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate, betas=(0.5, 0.999))
        # self.visualizer = Visualizer()

    def Image2Sketch_Train(self, rgb_image, sketch_vector, length_sketch, step, sketch_name):

        self.train()
        self.optimizer.zero_grad()

        curr_learning_rate = ((self.hp.learning_rate - self.hp.min_learning_rate) *
                              (self.hp.decay_rate) ** step + self.hp.min_learning_rate)
        curr_kl_weight = (self.hp.kl_weight - (self.hp.kl_weight - self.hp.kl_weight_start) *
                          (self.hp.kl_decay_rate) ** step)


        """ Encoding the Input """
        backbone_feature, rgb_encoded_dist = self.Image_Encoder(rgb_image)
        rgb_encoded_dist_z_vector = rgb_encoded_dist.rsample()

        """ Ditribution Matching Loss """
        prior_distribution = torch.distributions.Normal(torch.zeros_like(rgb_encoded_dist.mean),
                                                        torch.ones_like(rgb_encoded_dist.stddev))
        
        kl_cost_rgb = torch.max(torch.distributions.kl_divergence(rgb_encoded_dist, prior_distribution).mean(), torch.tensor(self.hp.kl_tolerance).to(device))
        
        ##############################################################
        ##############################################################
        """ Cross Modal the Decoding """
        ##############################################################
        ##############################################################
        
        photo2sketch_output = self.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1)
        
        end_token = torch.stack([torch.tensor([0, 0, 0, 0, 1])] * rgb_image.shape[0]).unsqueeze(0).to(device).float()
        batch = torch.cat([sketch_vector, end_token], 0)
        x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim
        
        sup_p2s_loss = sketch_reconstruction_loss(photo2sketch_output, x_target)  #TODO: Photo to Sketch Loss
        
        loss = sup_p2s_loss + curr_kl_weight*kl_cost_rgb
        
        set_learninRate(self.optimizer, curr_learning_rate)
        loss.backward()
        nn.utils.clip_grad_norm_(self.train_params, self.hp.grad_clip)
        self.optimizer.step()
        
        print('Step:{} ** sup_p2s_loss:{} ** kl_cost_rgb:{} ** Total_loss:{}'.format(step, sup_p2s_loss,
                                                                               kl_cost_rgb, loss))


        if step%5 == 0:
        
            data = {}
            data['Reconstrcution_Loss'] = sup_p2s_loss
            data['KL_Loss'] = kl_cost_rgb
            data['Total Loss'] = loss
        
            self.visualizer.plot_scalars(data, step)


        if step%1 == 0:

            folder_name = os.path.join('./CVPR_SSL/' + '_'.join(sketch_name.split('/')[-1].split('_')[:-1]))
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            sketch_vector_gt = sketch_vector.permute(1, 0, 2)

            save_sketch(sketch_vector_gt[0], sketch_name)


            with torch.no_grad():
                photo2sketch_gen, attention_plot  = \
                    self.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch+1, isTrain=False)

            sketch_vector_gt = sketch_vector.permute(1, 0, 2)


            for num, len in enumerate(length_sketch):
                photo2sketch_gen[num, len:, 4 ] = 1.0
                photo2sketch_gen[num, len:, 2:4] = 0.0

            save_sketch_gen(photo2sketch_gen[0], sketch_name)

            sketch_vector_gt_draw = batch_rasterize_relative(sketch_vector_gt)
            photo2sketch_gen_draw = batch_rasterize_relative(photo2sketch_gen)

            batch_redraw = []
            plot_attention = showAttention(attention_plot, rgb_image, sketch_vector_gt_draw, photo2sketch_gen_draw, sketch_name)
            # max_image = 5
            # for a, b, c, d in zip(sketch_vector_gt_draw[:max_image], rgb_image.cpu()[:max_image],
            #                       photo2sketch_gen_draw[:max_image], plot_attention[:max_image]):
            #     batch_redraw.append(torch.cat((1. - a, b, 1. - c,  d), dim=-1))
            #
            # torchvision.utils.save_image(torch.stack(batch_redraw), './Redraw_Photo2Sketch_'
            #                              + self.hp.setup + '/redraw_{}.jpg'.format(step),
            #                              nrow=1, normalize=False)

            # data = {'attention_1': [], 'attention_2':[]}
            # for x in attention_plot:
            #     data['attention_1'].append(x[0])
            #     data['attention_2'].append(x[2])
            #
            # data['attention_1'] = torch.stack(data['attention_1'])
            # data['attention_2'] = torch.stack(data['attention_2'])
            #
            # self.visualizer.vis_image(data, step)



        # return sup_p2s_loss, kl_cost_rgb, loss

        return 0, 0, 0



