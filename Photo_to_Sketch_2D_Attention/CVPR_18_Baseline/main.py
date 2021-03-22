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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Photo2Sketch')
    # parser.add_argument('--backbone_name', type=str, default='Resnet', help='VGG / InceptionV3/ Resnet')
    # parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d', help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--nThreads', type=int, default=8)

    # parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
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

    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--decay_rate', default=0.9999)
    parser.add_argument('--min_learning_rate', default=0.00001)
    parser.add_argument('--grad_clip', default=1.)

    # parser.add_argument('--sketch_rnn_max_seq_len', default=200)

    hp = parser.parse_args()

    print(hp)
    model = Photo2Sketch(hp)
    model.to(device)

    # """ Load Pretrained Model """
    # model.Sketch_Encoder.load_state_dict(torch.load('./pretrain_models/Sketch_Encoder.pth', map_location=device))
    # model.Sketch_Decoder.load_state_dict(torch.load('./pretrain_models/Sketch_Decoder.pth', map_location=device))

    """ Model Pretraining """
    # model.pretrain_SketchBranch(iteration=100000)
    # model.pretrain_ImageBranch()
    model.pretrain_SketchBranch_ShoeV2()

    """ Load Pretrained Model """
    model.Image_Encoder.load_state_dict(torch.load('./pretrain_models/Image_Encoder.pth', map_location=device))
    model.Image_Decoder.load_state_dict(torch.load('./pretrain_models/Image_Decoder.pth', map_location=device))
    model.Sketch_Encoder.load_state_dict(torch.load('./pretrain_models/Sketch_Encoder.pth', map_location=device))
    model.Sketch_Decoder.load_state_dict(torch.load('./pretrain_models/Sketch_Decoder.pth', map_location=device))

    """ Model Training Image2Sketch """
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    step = 0
    loss_best = 0


    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            rgb_image = batch_data['positive_img'].to(device)
            sketch_vector = batch_data['relative_fivePoint'].to(device).permute(1, 0, 2).float() # Seq_Len, Batch, Feature
            length_sketch = batch_data['sketch_length'].to(device) -1 #TODO: Relative coord has one less

            sup_p2s_loss, sup_s2p_loss, KL_1, KL_2, \
                short_p2p, short_s2s, total_loss = model.Image2Sketch_Train(rgb_image, sketch_vector, length_sketch, step)

            print('Step:{} ** sup_p2s_loss:{} ** sup_s2p_loss:{} ** KL_1:{} ** KL_2:{} '
                  '** short_p2p:{} ** short_s2s:{} ** Total_loss:{}'.format(step, sup_p2s_loss, sup_s2p_loss, KL_1, KL_2,
                                                              short_p2p, short_s2s, total_loss))

            # print(batch_data['sketch_img'].shape)
            #
            # start_time = time.time()
            # save_image(1. - batch_rasterize_relative(batch_data['relative_fivePoint']), 'a.jpg')
            # print('Time:{}'.format(time.time() - start_time))
            #
            # start_time = time.time()
            # save_image(1. - batch_rasterize_relative(batch_data['relative_coordinate']), 'b.jpg')
            # print('Time:{}'.format(time.time() - start_time))
            #
            #
            # save_image(batch_data['sketch_img'], 'c.jpg')




    # for i_epoch in range(hp.max_epoch):
    #     for batch_data in dataloader_Train:
    #         kl_cost, recons_loss, loss, curr_kl_weight = model.train_model(batch_data, step)
    #         step = step + 1
    #         print('Step:{} ** Current_KL:{} ** KL_Loss:{} '
    #               '** Recons_Loss:{} ** Total_loss:{}'.format(step, curr_kl_weight,
    #                                                           kl_cost, recons_loss, loss))
    #         if (step + 1) % hp.eval_freq_iter == 0:
    #             rand_int = random.randint(1, 21)
    #             for i_num, batch_data in enumerate(dataloader_Test):
    #                 if i_num > rand_int:
    #                     break
    #
    #             kl_cost, recons_loss, loss, curr_kl_weight = \
    #                 model.test_model(batch_data, step)
    #
    #             print('### Evaluation ### \n Step:{} ** Current_KL:{} ** KL_Loss:{} '
    #                   '** Recons_Loss:{} ** Total_loss:{}'.format(step, curr_kl_weight,
    #                                                               kl_cost, recons_loss, loss))
    #             if loss < loss_best:
    #                 loss_best = loss
    #                 print('### Model Updated ###')
    #                 torch.save(model.state_dict(), 'model_best.pth')

