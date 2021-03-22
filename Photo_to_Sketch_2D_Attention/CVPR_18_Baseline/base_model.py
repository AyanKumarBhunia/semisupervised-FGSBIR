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
from dataset import get_imageOnly_dataloader, get_sketchOnly_dataloader, get_dataloader
from rasterize import rasterize_relative, to_stroke_list
import math
from rasterize import batch_rasterize_relative



class Photo2Sketch_Base(nn.Module):

    def __init__(self, hp):
        super(Photo2Sketch_Base, self).__init__()
        self.Image_Encoder = EncoderCNN()
        self.Image_Decoder = DecoderCNN()
        self.Sketch_Encoder = EncoderRNN(hp)
        self.Sketch_Decoder = DecoderRNN(hp)
        self.hp = hp
        self.apply(weights_init_normal)

    def pretrain_SketchBranch(self, iteration = 100000):

        dataloader = get_sketchOnly_dataloader(self.hp)
        self.hp.max_seq_len = self.hp.sketch_rnn_max_seq_len
        self.Sketch_Encoder.train()
        self.Sketch_Decoder.train()
        self.train_sketch_params = list(self.Sketch_Encoder.parameters()) + list(self.Sketch_Decoder.parameters())
        self.sketch_optimizer = optim.Adam(self.train_sketch_params, self.hp.learning_rate)
        self.visalizer = Visualizer()

        for step in range(iteration):

            batch, lengths = dataloader.train_batch()

            self.sketch_optimizer.zero_grad()

            curr_learning_rate = ((self.hp.learning_rate - self.hp.min_learning_rate) *
                                  (self.hp.decay_rate) ** step + self.hp.min_learning_rate)
            curr_kl_weight = (self.hp.kl_weight - (self.hp.kl_weight - self.hp.kl_weight_start) *
                              (self.hp.kl_decay_rate) ** step)

            post_dist = self.Sketch_Encoder(batch, lengths)

            z_vector = post_dist.rsample()
            start_token = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.hp.batch_size_sketch_rnn).unsqueeze(0).to(
                device)
            batch_init = torch.cat([start_token, batch], 0)
            z_stack = torch.stack([z_vector] * (self.hp.sketch_rnn_max_seq_len + 1))
            inputs = torch.cat([batch_init, z_stack], 2)

            output, _ = self.Sketch_Decoder(inputs, z_vector, lengths + 1)

            end_token = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.shape[1]).unsqueeze(0).to(device)
            batch = torch.cat([batch, end_token], 0)
            x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim

            #################### Loss Calculation ########################################
            ##############################################################################
            recons_loss = sketch_reconstruction_loss(output, x_target)

            prior_distribution = torch.distributions.Normal(torch.zeros_like(post_dist.mean),
                                                            torch.ones_like(post_dist.stddev))
            kl_cost = torch.max(torch.distributions.kl_divergence(post_dist, prior_distribution).mean(),
                                torch.tensor(self.hp.kl_tolerance).to(device))
            loss = recons_loss + curr_kl_weight * kl_cost

            #################### Update Gradient ########################################
            #############################################################################
            set_learninRate(self.sketch_optimizer, curr_learning_rate)
            loss.backward()
            nn.utils.clip_grad_norm(self.train_sketch_params, self.hp.grad_clip)
            self.sketch_optimizer.step()

            if (step + 1) % 5 == 0:
                print('Step:{} ** KL_Loss:{} '
                      '** Recons_Loss:{} ** Total_loss:{}'.format(step, kl_cost.item(),
                                                                  recons_loss.item(), loss.item()))

                data = {}
                data['Reconstrcution_Loss'] = recons_loss
                data['KL_Loss'] = kl_cost
                data['Total Loss'] = loss
                self.visalizer.plot_scalars(data, step)

            if (step + 1) % self.hp.eval_freq_iter == 0:

                batch_input, batch_gen_strokes = self.sketch_generation_deterministic(dataloader)
                # batch_input, batch_gen_strokes = self.sketch_generation_sample(dataloader)

                batch_redraw =  batch_rasterize_relative(batch_gen_strokes)

                if batch_input is not None:
                    batch_input_redraw = batch_rasterize_relative(batch_input)
                    batch = []
                    for a, b in zip(batch_input_redraw, batch_redraw):
                        batch.append(torch.cat((a, 1. - b), dim=-1))
                    batch = torch.stack(batch).float()
                else:
                    batch = batch_redraw.float()

                torchvision.utils.save_image(batch, './pretrain_sketch_Viz/deterministic/batch_rceonstruction_' + str(step) + '_.jpg',
                                             nrow=round(math.sqrt(len(batch))))

                torch.save(self.Sketch_Encoder.state_dict(), './pretrain_models/Sketch_Encoder.pth')
                torch.save(self.Sketch_Decoder.state_dict(), './pretrain_models/Sketch_Decoder.pth')

                self.Sketch_Encoder.train()
                self.Sketch_Decoder.train()



    def sketch_generation_deterministic(self, dataloader, number_of_sample=64, condition = True):

        self.Sketch_Encoder.eval()
        self.Sketch_Decoder.eval()

        batch, lengths = dataloader.valid_batch(number_of_sample)
        if condition:
            post_dist = self.Sketch_Encoder(batch, lengths)
            z_vector = post_dist.sample()
        else:
            z_vector = torch.randn(number_of_sample, 128).to(device)

        start_token = torch.Tensor([0, 0, 1, 0, 0]).view(-1, 5).to(device)
        start_token = torch.stack([start_token] * number_of_sample, dim=1)
        state = start_token
        hidden_cell = None

        batch_gen_strokes = []
        for i_seq in range(self.hp.average_seq_len):
            input = torch.cat([state, z_vector.unsqueeze(0)], 2)
            state, hidden_cell = self.Sketch_Decoder(input, z_vector, hidden_cell=hidden_cell, isTrain=False, get_deterministic=True)
            batch_gen_strokes.append(state.squeeze(0))

        batch_gen_strokes = torch.stack(batch_gen_strokes, dim=1)

        if condition:
            return batch.permute(1, 0, 2), batch_gen_strokes
        else:
            return None, batch_gen_strokes


    def sketch_generation_sample(self, dataloader, number_of_sample=64, condition = True):

        self.Sketch_Encoder.eval()
        self.Sketch_Decoder.eval()

        batch_gen_strokes = []
        batch_input = []

        for i_x in range(number_of_sample):
            batch, lengths = dataloader.valid_batch(1)

            if condition:
                post_dist = self.Sketch_Encoder(batch, lengths)
                z_vector = post_dist.sample()
            else:
                z_vector = torch.randn(1,128).to(device)

            start_token = torch.Tensor([0,0,1,0,0]).view(1, 1, -1).to(device)
            state = start_token
            hidden_cell = None
            gen_strokes = []
            for i in range(self.hp.sketch_rnn_max_seq_len):
                input = torch.cat([state, z_vector.unsqueeze(0)],2)
                output, hidden_cell = self.Sketch_Decoder(input, z_vector, hidden_cell = hidden_cell, isTrain = False, get_deterministic=False)
                state, next_state = sample_next_state(output, self.hp)
                gen_strokes.append(next_state)

            gen_strokes = torch.stack(gen_strokes)
            batch_gen_strokes.append(gen_strokes)
            batch_input.append(batch.squeeze(1))

        batch_gen_strokes = torch.stack(batch_gen_strokes, dim=1)
        batch_input = torch.stack(batch_input, dim=1)

        if condition:
            return batch_input.permute(1, 0, 2), batch_gen_strokes.permute(1, 0, 2)
        else:
            return None, batch_gen_strokes.permute(1, 0, 2)


    def pretrain_ImageBranch(self, epoch = 200):

        image_dataloader = get_imageOnly_dataloader()
        self.Image_Encoder.train()
        self.Image_Decoder.train()
        self.train_image_params = list(self.Image_Encoder.parameters()) + list(self.Image_Decoder.parameters())
        self.image_optimizer = optim.Adam(self.train_image_params, self.hp.learning_rate)
        step = 0
        self.visalizer = Visualizer()

        for i_epoch in range(epoch):

            for _, batch_sample in enumerate(image_dataloader, 0):

                step = step + 1
                self.image_optimizer.zero_grad()

                batch_image = batch_sample[0].to(device)
                post_dist = self.Image_Encoder(batch_image)
                z_vector = post_dist.rsample()
                recons_batch_image = self.Image_Decoder(z_vector)

                # batch_image_normalized = transfer_ImageNomralization(batch_image, 'to_Gen')
                batch_image_normalized = batch_image
                recons_loss = F.mse_loss(batch_image_normalized, recons_batch_image, reduction='sum')/batch_image.shape[0]
                # recons_loss = F.mse_loss(batch_image_normalized, recons_batch_image)

                prior_distribution = torch.distributions.Normal(torch.zeros_like(post_dist.mean), torch.ones_like(post_dist.stddev))
                kl_cost = torch.distributions.kl_divergence(post_dist, prior_distribution).sum(1).mean()

                loss = recons_loss + kl_cost

                # log_var = torch.log(post_dist.stddev**2)
                # loss_matrx = 1 + log_var - post_dist.loc ** 2 - log_var.exp()
                # loss_matrx_sum = torch.sum(loss_matrx,  dim=1)
                # kld_loss = torch.mean(-0.5 * loss_matrx_sum, dim=0)

                loss.backward()
                nn.utils.clip_grad_norm(self.train_image_params, self.hp.grad_clip)
                self.image_optimizer.step()



                if (step + 1) % 20 == 0:
                    print('Step:{} ** KL_Loss:{} '
                          '** Recons_Loss:{} ** Total_loss:{}'.format(step, kl_cost.item(),
                                                                      recons_loss.item(), loss.item()))

                    data = {}
                    data['Reconstrcution_Loss'] = recons_loss
                    data['KL_Loss'] = kl_cost
                    data['Total Loss'] = loss
                    self.visalizer.plot_scalars(data, step)

                    data = {}
                    data['Input_Image'] = batch_image
                    data['Recons_Image'] = recons_batch_image
                    sample_z = torch.randn_like(z_vector)
                    data['Sampled_Image'] = self.Image_Decoder(sample_z)
                    self.visalizer.vis_image(data, step)


                if (step + 1) % self.hp.eval_freq_iter == 0:
                    saved_tensor = torch.cat([batch_image_normalized, recons_batch_image], dim=0)
                    torchvision.utils.save_image(saved_tensor, './pretrain_image_Viz/'+ str(step) + '.jpg', normalize=True)
                    torch.save(self.Image_Encoder.state_dict(), './pretrain_models/Image_Encoder' + str(step) + '.pth')
                    torch.save(self.Image_Decoder.state_dict(), './pretrain_models/Image_Decoder' + str(step) + '.pth')

    def pretrain_SketchBranch_ShoeV2(self, iteration = 10000):

        self.hp.batchsize = 100
        dataloader_Train, dataloader_Test = get_dataloader(self.hp)

        self.Sketch_Encoder.train()
        self.Sketch_Decoder.train()

        self.train_sketch_params = list(self.Sketch_Encoder.parameters()) + list(self.Sketch_Decoder.parameters())
        self.sketch_optimizer = optim.Adam(self.train_sketch_params, self.hp.learning_rate)

        self.visalizer = Visualizer()

        step =0

        for i_epoch in range(2000):

            for batch_data in dataloader_Train:

                batch = batch_data['relative_fivePoint'].to(device).permute(1, 0, 2).float()  # Seq_Len, Batch, Feature
                lengths = batch_data['sketch_length'].to(device) - 1  # TODO: Relative coord has one less
                step += 1
                # batch, lengths = dataloader.train_batch()

                self.sketch_optimizer.zero_grad()

                curr_learning_rate = ((self.hp.learning_rate - self.hp.min_learning_rate) *
                                      (self.hp.decay_rate) ** step + self.hp.min_learning_rate)
                curr_kl_weight = (self.hp.kl_weight - (self.hp.kl_weight - self.hp.kl_weight_start) *
                                  (self.hp.kl_decay_rate) ** step)

                post_dist = self.Sketch_Encoder(batch, lengths)

                z_vector = post_dist.rsample()
                start_token = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * batch.shape[1]).unsqueeze(0).to(device)
                batch_init = torch.cat([start_token, batch], 0)
                z_stack = torch.stack([z_vector] * (self.hp.max_seq_len + 1))
                inputs = torch.cat([batch_init, z_stack], 2)

                output, _ = self.Sketch_Decoder(inputs, z_vector, lengths + 1)

                end_token = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.shape[1]).unsqueeze(0).to(device)
                batch = torch.cat([batch, end_token], 0)
                x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim

                #################### Loss Calculation ########################################
                ##############################################################################
                recons_loss = sketch_reconstruction_loss(output, x_target)

                prior_distribution = torch.distributions.Normal(torch.zeros_like(post_dist.mean),
                                                                torch.ones_like(post_dist.stddev))
                kl_cost = torch.max(torch.distributions.kl_divergence(post_dist, prior_distribution).mean(),
                                    torch.tensor(self.hp.kl_tolerance).to(device))
                loss = recons_loss + curr_kl_weight * kl_cost

                #################### Update Gradient ########################################
                #############################################################################
                set_learninRate(self.sketch_optimizer, curr_learning_rate)
                loss.backward()
                nn.utils.clip_grad_norm(self.train_sketch_params, self.hp.grad_clip)
                self.sketch_optimizer.step()

                if (step + 1) % 5 == 0:
                    print('Step:{} ** KL_Loss:{} '
                          '** Recons_Loss:{} ** Total_loss:{}'.format(step, kl_cost.item(),
                                                                      recons_loss.item(), loss.item()))
                    data = {}
                    data['Reconstrcution_Loss'] = recons_loss
                    data['KL_Loss'] = kl_cost
                    data['Total Loss'] = loss
                    self.visalizer.plot_scalars(data, step)

                if (step  -1) % 1000 == 0:

                    """ Draw Sketch to Sketch """
                    start_token = torch.Tensor([0, 0, 1, 0, 0]).view(-1, 5).to(device)
                    start_token = torch.stack([start_token] * z_vector.shape[0], dim=1)
                    state = start_token
                    hidden_cell = None

                    batch_gen_strokes = []
                    for i_seq in range(self.hp.average_seq_len):
                        input = torch.cat([state, z_vector.unsqueeze(0)], 2)
                        state, hidden_cell = self.Sketch_Decoder(input, z_vector, hidden_cell=hidden_cell,
                                                                 isTrain=False,
                                                                 get_deterministic=True)
                        batch_gen_strokes.append(state.squeeze(0))

                    sketch2sketch_gen = torch.stack(batch_gen_strokes, dim=1)
                    sketch_vector_gt = batch.permute(1, 0, 2)

                    sketch_vector_gt_draw = batch_rasterize_relative(sketch_vector_gt).to(device)
                    sketch2sketch_gen_draw = batch_rasterize_relative(sketch2sketch_gen).to(device)

                    batch_redraw = []
                    for a, b in zip(sketch_vector_gt_draw, sketch2sketch_gen_draw):
                        batch_redraw.append(torch.cat((a, 1.- b), dim=-1))

                    torchvision.utils.save_image(torch.stack(batch_redraw),
                                                 './pretrain_sketch_Viz/ShoeV2/redraw_{}.jpg'.format(step),
                                                 nrow=8)

                    torch.save(self.Sketch_Encoder.state_dict(), './pretrain_models/ShoeV2/Sketch_Encoder.pth')
                    torch.save(self.Sketch_Decoder.state_dict(), './pretrain_models/ShoeV2/Sketch_Decoder.pth')

                    self.Sketch_Encoder.train()
                    self.Sketch_Decoder.train()



    def freeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = False


    def Unfreeze_weights(self):
        for name, x in self.named_parameters():
            x.requires_grad = True





