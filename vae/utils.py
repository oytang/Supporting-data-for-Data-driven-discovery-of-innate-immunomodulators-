# Useful helper functions

# function for benchmark of time using

import os
import time
import torch
from datetime import datetime

# time stamp
now = datetime.now()
now = str(now)[:19]
model_dir = './Models/{}'.format(str(now)).replace(':','-')

def timer(start):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    printw("-"*60 + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def save_models(encoder, decoder, epoch, model_dir=model_dir):
    save_dir = os.path.join(model_dir, 'Epochs:', '{}'.format(epoch) )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({'state_dict': encoder.state_dict()}, save_dir + '/encode.tar')
    torch.save({'state_dict': decoder.state_dict()}, save_dir + '/decode.tar')

def smiles_to_file(smiles_list, filename):
    with open(filename,'w') as content:
        content.write('smiles')
        content.write('\n')
        for smiles in smiles_list:
            content.write(smiles)
            content.write('\n')

def printw(line):
    with open(model_dir + '/training_log.txt', 'a') as content:
        content.write(line)
        content.write('\n')
    print(line)

selfies_alphabet = ['[epsilon]', '[C]', '[=C]', '[C@Hexpl]' '[N]', '[Ring1]', '[Branch1_1]', '[C@@Hexpl]', '[Branch1_3]', '[=O]', '[O]', '[Branch2_3]', '[Branch2_1]', '[=N]', '[c]', '[F]', '[Ring2]', '[Branch1_2]', '[o]', '[n]', '[S]', '[Branch2_2]', '[#N]', '[#C]', '[=S]', '[s]', '[-c]', '[Cl]', '[=c]', '[/C]', '[/c]', '[Br]', '[-n]', '[nHexpl]', '[=N+expl]', '[O-expl]', '[N+expl]', '[C@expl]', '[C@@expl]', '[/N]', '[\\S]', '[\\c]', '[.]', '[n+expl]', '[/O]', '[Expl-Ring1]', '[\\C]', '[P]', '[/S]' '[\\N]', '[\\O]', '[I]', '[Siexpl]', '[S@@expl]', '[S@expl]']
