import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
random.seed(9001)

from rasterize import *
import argparse
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from utils import *
import time
import torchvision.transforms.functional as TF

class Photo2Sketch_Dataset(data.Dataset):
    def __init__(self, hp, mode):
        super(Photo2Sketch_Dataset, self).__init__()

        self.hp = hp
        self.mode = mode
        hp.root_dir = '/home/media/On_the_Fly/Code_ALL/Final_Dataset'
        hp.dataset_name = 'ShoeV2'
        hp.seq_len_threshold = 251

        # coordinate_path = os.path.join(hp.root_dir, hp.dataset_name , hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.root_dir, hp.dataset_name)

        with open('./preprocess/ShoeV2_RDP_3', 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        coordinate_refine = {}
        seq_len = []
        for key in self.Coordinate.keys():
            if len(self.Coordinate[key]) < 81:
                coordinate_refine[key] = self.Coordinate[key]
                seq_len.append(len(self.Coordinate[key]))
        self.Coordinate = coordinate_refine
        hp.max_seq_len = max(seq_len)
        hp.average_seq_len = int(np.round(np.mean(seq_len) + np.std(seq_len)))

        # greater_than_average = 0
        # for seq in seq_len:
        #     if seq > self.hp.average_len:
        #         greater_than_average +=1

        self.Train_Sketch = [x for x in self.Coordinate if ('train' in x) and (len(self.Coordinate[x]) <130)]  # separating trains
        self.Test_Sketch = [x for x in self.Coordinate if ('test' in x) and (len(self.Coordinate[x]) <130)]    # separating tests

        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')

        # # seq_len = []
        # # for key in self.Coordinate.keys():
        # #     seq_len += [len(self.Coordinate[key])]
        # # plt.hist(seq_len)
        # # plt.savefig('histogram of number of Coordinate Points.png')
        # # plt.close()
        # # hp.max_seq_len = max(seq_len)
        # hp.max_seq_len = 130


        """" Preprocess offset coordinates """
        self.Offset_Coordinate = {}
        for key in self.Coordinate.keys():
            self.Offset_Coordinate[key] = to_delXY(self.Coordinate[key])
        data = []
        for sample in self.Offset_Coordinate.values():
            data.extend(sample[:, 0])
            data.extend(sample[:, 1])
        data = np.array(data)
        scale_factor = np.std(data)

        for key in self.Coordinate.keys():
            self.Offset_Coordinate[key][:, :2] /= scale_factor

        """" <<< Preprocess offset coordinates >>> """
        """" <<<           Done                >>> """



    def __getitem__(self, item):

        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            positive_sample = '_'.join(self.Train_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = Image.open(positive_path).convert('RGB')

            sketch_abs = self.Coordinate[sketch_path]
            sketch_delta = self.Offset_Coordinate[sketch_path]

            sketch_img = rasterize_Sketch(sketch_abs)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            sketch_img = TF.hflip(sketch_img)
            positive_img = TF.hflip(positive_img)
            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)

            #################################### #################################### ####################################

            absolute_coordinate = np.zeros((self.hp.max_seq_len, 3))
            relative_coordinate = np.zeros((self.hp.max_seq_len, 3))
            absolute_coordinate[:sketch_abs.shape[0], :] = sketch_abs
            relative_coordinate[:sketch_delta.shape[0], :] = sketch_delta
            #################################### #################################### ####################################

            sample = {'sketch_img': sketch_img,
                      'sketch_path': sketch_path,
                      'absolute_coordinate':absolute_coordinate,
                      'relative_coordinate': relative_coordinate,
                      'sketch_length': int(len(sketch_abs)),
                      'absolute_fivePoint': to_FivePoint(sketch_abs, self.hp.max_seq_len),
                      'relative_fivePoint': to_FivePoint(sketch_delta, self.hp.max_seq_len),
                      'positive_img': positive_img,
                      'positive_path': positive_sample}


        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')

            sketch_abs = self.Coordinate[sketch_path]
            sketch_delta = self.Offset_Coordinate[sketch_path]

            sketch_img = rasterize_Sketch(sketch_abs)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            sketch_img = self.test_transform(sketch_img)
            positive_img = self.test_transform(Image.open(positive_path).convert('RGB'))

            #################################### #################################### ####################################

            absolute_coordinate = np.zeros((self.hp.max_seq_len, 3))
            relative_coordinate = np.zeros((self.hp.max_seq_len, 3))
            absolute_coordinate[:sketch_abs.shape[0], :] = sketch_abs
            relative_coordinate[:sketch_delta.shape[0], :] = sketch_delta
            #################################### #################################### ####################################

            sample = {'sketch_img': sketch_img,
                      'sketch_path': sketch_path,
                      'absolute_coordinate':absolute_coordinate,
                      'relative_coordinate': relative_coordinate,
                      'sketch_length': int(len(sketch_abs)),
                      'absolute_fivePoint': to_FivePoint(sketch_abs),
                      'relative_fivePoint': to_FivePoint(sketch_delta),
                      'positive_img': positive_img,
                      'positive_path': positive_sample}

        return sample



    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)



def get_dataloader(hp):

    dataset_Train  = Photo2Sketch_Dataset(hp, mode = 'Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=False,
                                         num_workers=int(hp.nThreads))

    dataset_Test  = Photo2Sketch_Dataset(hp, mode = 'Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
                                         num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test


def get_transform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(256)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(256)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transforms.Compose(transform_list)


def get_imageOnly_dataloader(dataroot = '/home/media/CVPR_2021/Sketch_SelfSupervised/ut-zap50k-images-square'):
    dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(260),
                                       transforms.CenterCrop(256),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)),
                                   ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True,
                                             num_workers=8)
    return dataloader


class get_sketchOnly_dataloader(object):

    def __init__(self, hp):
        dataset = np.load('sketchrnn_shoe.npz', encoding='latin1', allow_pickle=True)
        # /home/media/Siggraph/pytorch-sketchRNN/cat.npz  sketchrnn_shoe.npz
        self.hp = hp
        data_train = dataset['train']
        data_valid = dataset['valid']


        # hp.sketch_rnn_max_seq_len = self.max_size(np.concatenate((data_train, data_valid)))
        sizes = [len(seq) for seq in np.concatenate((data_train, data_valid))]
        hp.sketch_rnn_max_seq_len = max(sizes)
        hp.average_seq_len = int(np.round(np.mean(sizes) + np.std(sizes)))



        data_train = self.purify(data_train)
        self.data_train = self.normalize(data_train)


        data_valid = self.purify(data_valid)
        self.data_valid = self.normalize(data_valid)


    def purify(self, strokes):
        """removes to small or too long sequences + removes large gaps"""
        data = []
        for seq in strokes:
            if seq.shape[0] <= self.hp.sketch_rnn_max_seq_len and seq.shape[0] > 10:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)
        return data

    def max_size(self, data):
        """larger sequence length in the data set"""
        sizes = [len(seq) for seq in data]
        self.hp.average_len = np.round(np.mean(sizes) + np.std(sizes))

        # greater_than_average = 0
        # for seq in data:
        #     if len(seq) > self.hp.average_len:
        #         greater_than_average +=1

        return max(sizes)


    def calculate_normalizing_scale_factor(self, strokes):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(strokes)):
            for j in range(len(strokes[i])):
                data.append(strokes[i][j, 0])
                data.append(strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, strokes):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(strokes)
        for seq in strokes:
            seq[:, 0:2] /= scale_factor
            data.append(seq)
        return data

    def train_batch(self, batch_size_sketch_rnn=100):
        batch_idx = np.random.choice(len(self.data_train), batch_size_sketch_rnn)
        batch_sequences = [self.data_train[idx] for idx in batch_idx]
        strokes = []
        lengths = []
        indice = 0
        for seq in batch_sequences:
            len_seq = len(seq[:, 0])
            new_seq = np.zeros((self.hp.sketch_rnn_max_seq_len, 5))
            new_seq[0:len_seq, :2] = seq[:, :2]
            new_seq[0:len_seq, 3] = seq[:, 2]
            new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
            new_seq[(len_seq-1):, 4] = 1
            new_seq[(len_seq - 1), 2:4] = 0
            lengths.append(len(seq[:, 0]))
            strokes.append(new_seq)
            indice += 1

        batch = torch.from_numpy(np.stack(strokes, 1)).to(device).float()
        return batch, torch.tensor(lengths).type(torch.int64).to(device)


    def valid_batch(self, batch_size_sketch_rnn=100):
        batch_idx = np.random.choice(len(self.data_valid), batch_size_sketch_rnn)
        batch_sequences = [self.data_valid[idx] for idx in batch_idx]
        strokes = []
        lengths = []
        indice = 0
        for seq in batch_sequences:
            len_seq = len(seq[:, 0])
            new_seq = np.zeros((self.hp.sketch_rnn_max_seq_len, 5))
            new_seq[0:len_seq, :2] = seq[:, :2]
            new_seq[0:len_seq, 3] = seq[:, 2]
            new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
            new_seq[len_seq:, 4] = 1
            lengths.append(len(seq[:, 0]))
            strokes.append(new_seq)
            indice += 1

        batch = torch.from_numpy(np.stack(strokes, 1)).to(device).float()
        return batch, torch.tensor(lengths).type(torch.int64).to(device)


if __name__ == '__main__':
    pass
    # ########## Module Testing ####################
    # parser = argparse.ArgumentParser(description='rgb2sketch')
    # parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    # hp = parser.parse_args()
    # hp.batchsize = 64
    # hp.nThreads = 8
    # hp.splitTrain = 0.7
    # dataset_Train = Photo2Sketch_Dataset(hp, mode='Train')
    # dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=False,
    #                                    num_workers=int(hp.nThreads))
    #
    # canvas = plt.figure(frameon=False, figsize=(2.56, 2.56))
    #
    # for data in dataloader_Train:
    #     print(data['sketch_img'].shape)
    #     print(data['absolute_coordinate'].shape)
    #     print(data['relative_coordinate'].shape)
    #     print(data['absolute_fivePoint'].shape)
    #     print(data['relative_fivePoint'].shape)
    #     torchvision.utils.save_image(data['sketch_img'], 'sketch_img.jpg', normalize=True)
    #     torchvision.utils.save_image(data['positive_img'], 'positive_img.jpg', normalize=True)
    #
    #
    #     batch_redraw = []
    #
    #     for data in data['relative_fivePoint']:
    #         image = rasterize_relative(to_stroke_list(to_normal_strokes(data)), canvas)
    #         batch_redraw.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    #
    #     torchvision.utils.save_image(torch.stack(batch_redraw).float(), 'batch_redraw.jpg', normalize=True)
    #
    #
    #     # image = rasterize_relative(to_stroke_list(to_normal_strokes(data['relative_fivePoint'][7])), canvas)
    #     # image.save('a.jpg')
    #     # image.show()

