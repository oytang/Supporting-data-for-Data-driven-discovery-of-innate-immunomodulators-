import os, sys, time
import json
import numpy as np
import torch
import pandas as pd
import selfies
from torch import nn
torch.backends.cudnn.benchmark = True

from utils import timer, printw, save_models, model_dir
from VAE import VAE_encode, VAE_decode

# ----------------------------------------------------------------------------------

def train_model(models, idx, path, device, warm_epoch, batch_size, latent_dimension, KLD_alpha, lr_enc, lr_dec, num_epochs):
    
    # data preprocessing
    num_train = int(len(idx))
    idx_train = np.array(idx)
    num_batches_train = int(num_train / batch_size) # drop_last = True

    model_encode, model_decode = models[0], models[1]
    optimizer_encoder = torch.optim.Adam(model_encode.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(model_decode.parameters(), lr=lr_dec)

    # iterables collecting training profile
    epoch_list = []
    loss_list = [[]]
    recon_list = [[]]
    kld_list = [[]]
    excon_list = [[]]

    all_start = time.time()

    if warm_epoch is None:
        epoch = 0
        end = num_epochs
    else:
        epoch = int(warm_epoch)
        end = warm_epoch + num_epochs
    
    while True:
        
        save_models(model_encode, model_decode, epoch)

        np.random.shuffle(idx_train)

        epoch_list.append(epoch)
        
        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator
            
            loss, recon_loss, kld = 0., 0., 0.

            # manual batch iterations
            current_idx_start = batch_iteration * batch_size
            current_idx_stop = (batch_iteration + 1) * batch_size
            idx_smile_hot = idx_train[current_idx_start: current_idx_stop]
            
            # read in the training data
            inp_smile_hot = np.array([np.load(path + '%d.npy' % idx) for idx in idx_smile_hot])
            inp_smile_hot = torch.tensor(inp_smile_hot, dtype=torch.float).to(device)
            
            # reshaping for efficient parallelization
            inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0],
                                                     inp_smile_hot.shape[1] * inp_smile_hot.shape[2])
            
            # encoding molecular one-hots into latent space
            latent_points, mus, log_vars = model_encode(inp_smile_encode)
            latent_points = latent_points.reshape(1, batch_size, latent_points.shape[1])

            # standard Kullback–Leibler divergence
            kld += -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

            # initialization hidden internal state of RNN (RNN has two inputs and two outputs:)
            #    input: latent space & hidden state
            #    output: onehot encoding of one character of molecule & hidden state
            #    the hidden state acts as the internal memory
            hidden = model_decode.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest molecule (all molecules are padded)
            decoded_one_hot = torch.zeros(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2]).to(device)
            for seq_index in range(inp_smile_hot.shape[1]):
                decoded_one_hot_line, hidden = model_decode(latent_points, hidden)
                decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
            
            decoded_one_hot = decoded_one_hot.reshape(batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2])
            _, label_atoms = inp_smile_hot.max(2)
            label_atoms = label_atoms.reshape(batch_size * inp_smile_hot.shape[1])

            # we use cross entropy of expected symbols and decoded one-hot
            criterion = torch.nn.CrossEntropyLoss()
            recon_loss += criterion(decoded_one_hot, label_atoms)

            loss += recon_loss + KLD_alpha * kld

            if (batch_iteration + 1) % 10 == 0:
                loss_list[-1].append(float(loss))
                recon_list[-1].append(float(recon_loss))
                kld_list[-1].append(float(kld))

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model_decode.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()
            
            # each epoch prints around 10 rows whatever the batch size
            table_rows = (int(num_batches_train / 50) + 1) * 5
            if (batch_iteration + 1) % table_rows == 0 or batch_iteration + 1 == num_batches_train:
                # assess reconstruction quality
                _, decoded_max_indices = decoded_one_hot.max(1)
                _, input_max_indices = inp_smile_hot.reshape(batch_size * inp_smile_hot.shape[1],
                                                             inp_smile_hot.shape[2]).max(1)

                differences = torch.abs(decoded_max_indices - input_max_indices)
                # differences = torch.clamp(differences, min=0., max=1.).double()
                # quality = 100. * (1.0 - torch.mean(differences))
                # quality = quality.detach().cpu().numpy()
                differences = differences.reshape(batch_size, -1)
                differences = torch.sum(differences, dim=1)
                differences = torch.clamp(differences, min=0., max=1.).double()
                excon = 1.0 - torch.sum(differences) / batch_size
                excon_list[-1].append(float(excon))
                
                new_line = 'Epoch: %3d, Batch: %3d / %3d,\tLoss: %6.4f | Recon: %6.4f | KLD: %6.4f | Excon: %6.4f | Time: %3d' % (
                    epoch, 
                    (batch_iteration + 1), 
                    num_batches_train, 
                    loss.item(), 
                    recon_loss.item(),
                    kld.item(),
                    excon, 
                    time.time() - start)
                printw(new_line)

        timer(all_start)
        
        # update local training profiles
        loss_array = np.array(loss_list)
        recon_array = np.array(recon_list)
        kld_array = np.array(kld_list)
        excon_array = np.array(excon_list)
        epoch_array = np.array(epoch_list)

        np.save(os.path.join(model_dir , 'loss_array'), loss_array)
        np.save(os.path.join(model_dir , 'recon_array'), recon_array)
        np.save(os.path.join(model_dir , 'kld_array'), kld_array)
        np.save(os.path.join(model_dir , 'excon_array'), excon_array)
        np.save(os.path.join(model_dir , 'epoch_array'), epoch_array)

        loss_list.append([])
        recon_list.append([])
        kld_list.append([])
        excon_list.append([])

        epoch += 1

        # epoch end

