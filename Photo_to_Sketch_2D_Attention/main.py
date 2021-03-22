import torch
from model import Photo2Sketch
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import random
from matplotlib import pyplot as plt
from rasterize import batch_rasterize_relative
from torchvision.utils import save_image
import time
from dataset import get_sketchOnly_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Photo2Sketch')

    parser.add_argument('--setup', type=str, default='QMUL', help='QuickDraw vs QMUL')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--nThreads', type=int, default=8)

    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)


    parser.add_argument('--enc_rnn_size', default=256)
    parser.add_argument('--dec_rnn_size', default=512)
    parser.add_argument('--z_size', default=128)

    parser.add_argument('--num_mixture', default=20)
    parser.add_argument('--input_dropout_prob', default=0.9)
    parser.add_argument('--output_dropout_prob', default=0.9)
    parser.add_argument('--batch_size_sketch_rnn', default=100)

    parser.add_argument('--kl_weight_start', default=0.01)
    parser.add_argument('--kl_decay_rate', default=0.99995)
    parser.add_argument('--kl_tolerance', default=0.2)
    parser.add_argument('--kl_weight', default=1.0)

    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--decay_rate', default=0.9999)
    parser.add_argument('--min_learning_rate', default=0.00001)
    parser.add_argument('--grad_clip', default=1.)

    hp = parser.parse_args()


    print(hp)
    model = Photo2Sketch(hp)
    model.to(device)
    model.load_state_dict(torch.load('./modelCVPR21/QMUL/model_photo2Sketch_QMUL_2Dattention_8000_.pth'))


    step = 0
    current_loss = 1e+10

    """ Model Training Image2Sketch """

    if hp.setup == 'QuickDraw':

        dataloader = get_sketchOnly_dataloader(hp)

        for step in range(100000*2):
            sample = dataloader.train_batch()

            rgb_image = sample['photo'].to(device)
            sketch_vector = sample['sketch_vector'].to(device)  # Seq_Len, Batch, Feature
            length_sketch = sample['length'].to(device)

            sup_p2s_loss, kl_cost_rgb, total_loss = model.Image2Sketch_Train(rgb_image, sketch_vector, length_sketch, step)

            if total_loss.item() < current_loss:
                torch.save(model.state_dict(), './modelCVPR21/model_photo2Sketch_Pretraining2D_K.pth')
                current_loss = total_loss.item()

    elif  hp.setup == 'QMUL':

        dataloader_Train, dataloader_Test = get_dataloader(hp)

        for i_epoch in range(hp.max_epoch):
            for z_num, batch_data in enumerate(dataloader_Train):

                rgb_image = batch_data['photo'].to(device)
                sketch_vector = batch_data['sketch_vector'].to(device).permute(1, 0, 2).float()  # Seq_Len, Batch, Feature
                length_sketch = batch_data['length'].to(device) - 1  # TODO: Relative coord has one less
                sketch_name = batch_data['sketch_path'][0]

                sup_p2s_loss, kl_cost_rgb, total_loss = model.Image2Sketch_Train(rgb_image, sketch_vector,
                                                                                 length_sketch, step, sketch_name)
                step += 1

                print(z_num)
                # if total_loss.item() < current_loss:
                #     torch.save(model.state_dict(), './modelCVPR21/model_photo2Sketch_QMUL_2Dattention_Pre.pth')
                #     current_loss = total_loss.item()

                # if step % 1000 == 0:
                #     torch.save(model.state_dict(), './modelCVPR21/QMUL/model_photo2Sketch_QMUL_2Dattention_' + str(step) + '_.pth')
                #     current_loss = total_loss.item()