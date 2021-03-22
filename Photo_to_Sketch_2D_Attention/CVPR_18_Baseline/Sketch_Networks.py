import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from utils import *
import torch.nn as nn
import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self, hp):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(5, hp.enc_rnn_size, dropout=hp.input_dropout_prob, bidirectional=True)
        self.fc_mu = nn.Linear(2*hp.enc_rnn_size, hp.z_size)
        self.fc_sigma = nn.Linear(2*hp.enc_rnn_size, hp.z_size)

    def forward(self, x, Seq_Len=None):
        x = pack_padded_sequence(x, Seq_Len, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x.float())
        h_n = h_n.permute(1,0,2).reshape(h_n.shape[1], -1)
        mean = self.fc_mu(h_n)
        log_var = self.fc_sigma(h_n)
        posterior_dist = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        return posterior_dist



class DecoderRNN(nn.Module):
    def __init__(self, hp):
        super(DecoderRNN, self).__init__()
        self.fc_hc = nn.Linear(hp.z_size, 2 * hp.dec_rnn_size)
        self.lstm = nn.LSTM(hp.z_size + 5, hp.dec_rnn_size, dropout=hp.output_dropout_prob)
        self.fc_params = nn.Linear(hp.dec_rnn_size, 6 * hp.num_mixture + 3)
        self.hp = hp

    def forward(self, inputs, z_vector, seq_len = None, hidden_cell=None, isTrain = True, get_deterministic = True):

        self.training = isTrain
        if hidden_cell is None:
            hidden, cell = torch.split(F.tanh(self.fc_hc(z_vector)), self.hp.dec_rnn_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())

        if seq_len is None:
            # seq_len = torch.tensor([1]).type(torch.int64).to(device)
            seq_len = torch.ones(inputs.shape[1]).type(torch.int64).to(device)

        inputs = pack_padded_sequence(inputs, seq_len, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        outputs, _ = pad_packed_sequence(outputs)

        if self.training:
            if outputs.shape[0] != (self.hp.max_seq_len + 1):
                pad = torch.zeros(outputs.shape[-1]).repeat(self.hp.max_seq_len + 1 - outputs.shape[0], outputs.shape[1], 1).cuda()
                outputs = torch.cat((outputs, pad), dim=0)
            y_output = self.fc_params(outputs.permute(1,0,2))
        else:
            y_output = self.fc_params(hidden.permute(1,0,2))

        z_pen_logits = y_output[:, :, 0:3]
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(y_output[:, :, 3:], 6, 2)
        z_pi = F.softmax(z_pi, dim=-1)
        z_sigma1 = torch.exp(z_sigma1)
        z_sigma2 = torch.exp(z_sigma2)
        z_corr = torch.tanh(z_corr)


        if (not self.training) and get_deterministic:
            batch_size = z_pi.shape[0]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits = z_pi.reshape(-1, 20), z_mu1.reshape(-1, 20), z_mu2.reshape(-1, 20), \
            z_sigma1.reshape(-1, 20), z_sigma2.reshape(-1, 20), z_corr.reshape(-1, 20), z_pen_logits.reshape(-1, 3)

            recons_output = torch.zeros(batch_size, 5).to(device)
            z_pi_idx = z_pi.argmax(dim=-1)
            z_pen_idx = z_pen_logits.argmax(-1)
            recons_output[:, 0] = z_mu1[range(z_mu1.shape[0]), z_pi_idx]
            recons_output[:, 1] = z_mu2[range(z_mu2.shape[0]), z_pi_idx]

            recons_output[range(z_mu1.shape[0]), z_pen_idx + 2] = 1.

            return  recons_output.unsqueeze(0).data, (hidden, cell)

        return [z_pi.reshape(-1, 20), z_mu1.reshape(-1, 20), z_mu2.reshape(-1, 20), \
               z_sigma1.reshape(-1, 20), z_sigma2.reshape(-1, 20), z_corr.reshape(-1, 20), z_pen_logits.reshape(-1, 3)], (hidden, cell)


def torch_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2

    z_1 = (norm1 / s1) ** 2
    z_2 = (norm2 / s2) ** 2
    z1_z2 = (norm1 * norm2) / s1s2

    z = z_1 + z_2 - 2 * rho * z1_z2
    neg_rho = 1 - rho ** 2
    result = torch.exp(-z / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    return result / denom


def sketch_reconstruction_loss(output, x_input):
    # x_input =
    # Ouput = Predicted 123 parameters from decoder = Batch*Max_seq_len, 20
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = output
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = torch.chunk(x_input.reshape(-1, 5), 5, 1)
    pen_data = torch.cat([eos_data, eoc_data, cont_data], 1)
    mask = 1.0 - pen_data[:, 2]  # use training data for this

    result0 = torch_2d_normal(x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2,
                                   o_corr)
    epsilon = 1e-6

    result1 = torch.sum(result0 * o_pi, dim=1)  # ? unsqueeae(-1) ??
    result1 = -torch.log(result1 + epsilon)  # avoid log(0)

    result2 = F.cross_entropy(o_pen_logits, pen_data.argmax(1), reduction='none')

    result = mask * result1 + mask * result2
    # result = result1 + result2

    return result.mean()

def set_learninRate(optimizer, curr_learning_rate):
    for g in optimizer.param_groups:
        g['lr'] = curr_learning_rate
