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
# from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import cv2
import imageio

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

        # if os.path.exists('Tensorboard_' + name):
        #     shutil.rmtree('Tensorboard_' + name)

        self.writer = SummaryWriter()

        self.mean = torch.tensor([-1.0, -1.0, -1.0]).to(device)
        self.std = torch.tensor([1 / 0.5, 1 / 0.5, 1 / 0.5]).to(device)

    def vis_image(self, visularize, step, normalize=False):
        for keys, value in visularize.items():
            #print(keys,value.size())
            if normalize:
                value.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
            visularize[keys] = torchvision.utils.make_grid(value)
            self.writer.add_image('{}'.format(keys), visularize[keys], step)


    def plot_scalars(self, scalars, step):

        for keys, value in scalars.items():
            #print(keys,value.size())
            self.writer.add_scalar('{}'.format(keys), scalars[keys], step)


def showAttention(attention_plot, sketch_img, sketch_vector_gt_draw, photo2sketch_gen_draw, sketch_name):
    # Set up figure with colorbar
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).to('cpu')
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).to('cpu')


    folder_name = os.path.join('./CVPR_SSL/' + '_'.join(sketch_name.split('/')[-1].split('_')[:-1]))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # sketch_vector_gt_draw = sketch_vector_gt_draw.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    sketch_vector_gt_draw = sketch_vector_gt_draw.squeeze(0).permute(1, 2, 0).numpy()
    sketch_vector_gt_draw = cv2.resize(np.float32(np.uint8(255. * (1. - sketch_vector_gt_draw))), (256, 256))

    # photo2sketch_gen_draw = photo2sketch_gen_draw.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    photo2sketch_gen_draw = photo2sketch_gen_draw.squeeze(0).permute(1, 2, 0).numpy()
    photo2sketch_gen_draw = cv2.resize(np.float32(np.uint8(255. * (1. -photo2sketch_gen_draw))), (256, 256))

    imageio.imwrite(folder_name + '/sketch_' + 'GT.jpg', sketch_vector_gt_draw)
    imageio.imwrite(folder_name + '/sketch_' + 'Gen.jpg', photo2sketch_gen_draw)

    attention_dictionary = {}
    for num, val in enumerate(sketch_img):
        attention_dictionary[num] = []
        val = val.cpu()
        x = val.unsqueeze(0)
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        x = x.squeeze(0)
        attention_dictionary[num].append(x)


    alpha = 0.5
    for atten_num, x_data in enumerate(attention_plot):
        for num, per_image_x in enumerate(x_data):

            attention = per_image_x.squeeze(0).cpu().numpy()

            # attention[attention < 0.01] = 0
            # attention = attention / attention.sum()
            # attention = np.clip(attention / attention.max() * 255., 0, 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
            heatmap = cv2.resize(np.float32(heatmap), (256, 256))

            # heatmap = heatmap**2


            # image = 255. - attention_dictionary[num][0].permute(1, 2, 0).numpy()

            # mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).to('cpu')
            # std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).to('cpu')
            # x = attention_dictionary[num][0].unsqueeze(0)
            # x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            # x = x.squeeze(0)
            image = attention_dictionary[num][0].permute(1, 2, 0).numpy()
            image = cv2.resize(np.float32(np.uint8(255. * image)), (256, 256))

            # image preprocess

            imageio.imwrite(folder_name  + '/RGB.jpg', image)

            heat_map_overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
            imageio.imwrite(folder_name + '/' + sketch_name.split('/')[-1] +  '_' + str(atten_num) + '.jpg', heat_map_overlay)


            heat_map_tensor = torch.from_numpy(heat_map_overlay).permute(2, 0, 1)
            # heat_map_tensor = attention_dictionary[num][0] + torch.from_numpy(heatmap).permute(2, 0, 1)
            # heat_map_tensor = heat_map_tensor / heat_map_tensor.max()
            attention_dictionary[num].append(heat_map_tensor)

    # plot_attention = []
    # for num, val in enumerate(sketch_img):
    #     image = torch.stack(attention_dictionary[num][1:], dim=0).permute(1, 2, 0, 3).reshape(3, 256, -1)
    #     image.add_(-image.min()).div_(image.max() - image.min()+ 1e-5)
    #     plot_attention.append(image)

    # return torch.stack(plot_attention)
    return None

# def showAttention(attention_plot, sketch_img):
#     # Set up figure with colorbar
#
#     attention_dictionary = {}
#     for num, val in enumerate(sketch_img):
#         attention_dictionary[num] = []
#         attention_dictionary[num].append(val.cpu())
#     alpha = 0.3
#     for x_data in attention_plot:
#         for num, per_image_x in enumerate(x_data):
#             attention = per_image_x.squeeze(0).cpu().numpy()
#             heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
#             heatmap = cv2.resize(np.float32(heatmap), (256, 256))
#
#             image = 255. - attention_dictionary[num][0].permute(1, 2, 0).numpy()
#             # mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).to(device)
#             # std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).to(device)
#             # x = attention_dictionary[num][0].unsqueeze(0)
#             # x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
#             # x = x.squeeze(0)
#             # image = x.permute(1, 2, 0).numpy()
#             # image = cv2.resize(np.float32(np.uint8(255. * image)), (256, 256))
#
#             heat_map_overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
#             heat_map_tensor = torch.from_numpy(heat_map_overlay).permute(2, 0, 1)
#             # heat_map_tensor = attention_dictionary[num][0] + torch.from_numpy(heatmap).permute(2, 0, 1)
#             # heat_map_tensor = heat_map_tensor / heat_map_tensor.max()
#             attention_dictionary[num].append(heat_map_tensor)
#
#     plot_attention = []
#     for num, val in enumerate(sketch_img):
#         image = torch.stack(attention_dictionary[num][1:], dim=0).permute(1, 2, 0, 3).reshape(3, 256, -1)
#         image.add_(-image.min()).div_(image.max() - image.min()+ 1e-5)
#         plot_attention.append(image)
#
#     return torch.stack(plot_attention)


    #
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(attentions.numpy(), cmap='bone')
    # fig.colorbar(cax)
    #
    # # Set up axes
    # ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                    ['<EOS>'], rotation=90)
    # ax.set_yticklabels([''] + output_words)
    #
    # # Show label at every tick
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

def to_stroke_list(sketch):
    ## sketch: an `.npz` style sketch from QuickDraw
    sketch = np.vstack((np.array([0, 0, 0]), sketch))
    sketch[:, :2] = np.cumsum(sketch[:, :2], axis=0)

    # range normalization
    xmin, xmax = sketch[:, 0].min(), sketch[:, 0].max()
    ymin, ymax = sketch[:, 1].min(), sketch[:, 1].max()

    sketch[:, 0] = ((sketch[:, 0] - xmin) / float(xmax - xmin)) * (255. - 60.) + 30.
    sketch[:, 1] = ((sketch[:, 1] - ymin) / float(ymax - ymin)) * (255. - 60.) + 30.
    sketch = sketch.astype(np.int64)

    stroke_list = np.split(sketch[:, :2], np.where(sketch[:, 2])[0] + 1, axis=0)

    if stroke_list[-1].size == 0:
        stroke_list = stroke_list[:-1]

    if len(stroke_list) == 0:
        stroke_list = [sketch[:, :2]]
        # print('error')
    return stroke_list

def save_sketch(sketch_vector, sketch_name):
    stroke_list = to_stroke_list(to_normal_strokes(sketch_vector.cpu().numpy()))

    points = np.sum([len(x) for x in stroke_list])
    point_list = [len(x) for x in stroke_list]

    folder_name = os.path.join('./CVPR_SSL/' + '_'.join(sketch_name.split('/')[-1].split('_')[:-1]), sketch_name.split('/')[-1])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    fig = plt.figure(frameon=False, figsize=(2.56, 2.56))
    xlim = [0, 255]
    ylim = [0, 255]
    x_count = 0
    count = 0
    for stroke in stroke_list:
        stroke_buffer = np.array(stroke[0])
        for x_num in range(len(stroke)):
            x_count = x_count + 1
            stroke_buffer = np.vstack((stroke_buffer, stroke[x_num, :2]))
            if x_count % 5 == 0:

                plt.plot(stroke_buffer[:, 0], stroke_buffer[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
                plt.gca().invert_yaxis();
                plt.axis('off')

                plt.savefig(folder_name + '/sketch_' + str(count) + 'points_.jpg', bbox_inches='tight',
                            pad_inches=0, dpi=1200)
                count = count + 1
                plt.gca().invert_yaxis();


        plt.plot(stroke_buffer[:, 0], stroke_buffer[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)






def save_sketch_gen(sketch_vector, sketch_name):
    stroke_list = to_stroke_list(to_normal_strokes(sketch_vector.cpu().numpy()))

    folder_name = os.path.join('./CVPR_SSL/' + '_'.join(sketch_name.split('/')[-1].split('_')[:-1]), sketch_name.split('/')[-1]+'GEN')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    fig = plt.figure(frameon=False, figsize=(2.56, 2.56))
    xlim = [0, 255]
    ylim = [0, 255]
    x_count = 0
    count = 0
    for stroke in stroke_list:
        stroke_buffer = np.array(stroke[0])
        for x_num in range(len(stroke)):
            x_count = x_count + 1
            stroke_buffer = np.vstack((stroke_buffer, stroke[x_num, :2]))
            if x_count % 5 == 0:

                plt.plot(stroke_buffer[:, 0], stroke_buffer[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
                plt.gca().invert_yaxis();
                plt.axis('off')

                plt.savefig(folder_name + '/sketch_' + str(count) + 'points_.jpg', bbox_inches='tight',
                            pad_inches=0, dpi=1200)
                count = count + 1
                plt.gca().invert_yaxis();


        plt.plot(stroke_buffer[:, 0], stroke_buffer[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)