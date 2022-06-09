
import time
import numpy as np
import torch.optim as optim
import torch
from tqdm import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='WSD Evaluation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', default=1, type=int)
    parser.add_argument('-norm', default=False)
    parser.add_argument('-step', default=1)
    # parser.add_argument('-noun',default=False)
    args = parser.parse_args()
    return args

def load_source_embeddings(sources:list,emb_dim =2048):
    """
    Load all source embeddings and compute the intersection of all vocabularies.
    """
    embeddings = []
    vocab = set(sources[0]['labels'])

    #get the intersection of the vocabs
    for np_loader in sources:

        vocab = vocab.intersection(np_loader['labels'])

    # sort vocab intersection, create sense2idx, idx2sense
    embs = []
    vocab = list(vocab)
    vocab = sorted(vocab)
    sense_to_ix = {}
    for sense in vocab:
        sense_to_ix[sense] = len(sense_to_ix)
    ix_to_sense = {value: key for (key, value) in sense_to_ix.items()}

    # Get matrices for training with the same sense order
    for np_loader in sources:
        src = np.zeros((len(vocab), emb_dim))
        vecs = {k:v for k,v in zip(np_loader['labels'],np_loader['vectors'])}
        for sense in vocab:
            src[sense_to_ix[sense], :] = vecs[sense]
        embs.append(torch.tensor(src,device=device,dtype=torch.float32))

        del vecs
    return embs,sense_to_ix,ix_to_sense

def get_res(key_list,mat_list):
    vocab = set(key_list[0])
    for skset in key_list:
        vocab = vocab.union(skset)
    print('final vocab size',len(vocab),vocab[1])
    sense_to_ix = {}
    for sense in vocab:
        sense_to_ix[sense] = len(sense_to_ix)
    res = np.zeros((len(vocab), emb_dim))
    for i in range(len(key_list)):
        vecs = {k: v for k, v in zip(key_list[i], mat_list[i])}
        for sense in vocab:
            res[sense_to_ix[sense], :] += vecs[sense]
    return res

if __name__ == '__main__':
    #
    args = get_args()
    if torch.cuda.is_available():
        print('device num', torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print('current idx', idx)
        print(torch.cuda.get_device_name(idx))
    device = torch.device('cuda')

    proj_mat1 = torch.eye(2048)
    proj_mat1 = proj_mat1.to(device)
    proj_mat1.requires_grad = True

    proj_mat2 = torch.eye(2048)
    proj_mat2 = proj_mat2.to(device)
    proj_mat2.requires_grad = True

    optmizer = optim.Adam([proj_mat1,proj_mat2], lr=0.001)
    res = torch.zeros(1)
    res.to(device)
    ares_name = ''
    if args.norm:
        ares = np.load('../emb_npz/ares.npz')
        ares_name = 'n_ares'
    else:
        ares = np.load('../emb_npz/ares_org.npz')
        ares_name = 'unares'
    smbert = np.load('../sensembert/sensembert_norm.npz')
    sources = [smbert,ares]
    embs,sense_to_ix,ix_to_sense = load_source_embeddings(sources)
    emb1 = embs[0]
    emb2 = embs[1]


    batch_sz = 10000
    step = args.step
    loss = 0
    naive_loss = 0
    epoch = 3 # args.e
    loss_list = []
    vocab_size = len(sense_to_ix.keys())
    for _ in range(epoch):
        epoch_loss = 0
        for i in range(0, vocab_size// batch_sz):
            src1 = emb1[i * batch_sz:(i + 1) * batch_sz, :]

            src2 = emb2[i * batch_sz:(i + 1) * batch_sz, :]

            meta =  torch.matmul(src1,proj_mat1) + \
                       torch.matmul(src2, proj_mat2)
            meta_pip = torch.matmul(meta, meta.T)
            naive = src1+src2
            naive_pip = torch.matmul(naive, naive.T)

            naive_loss = naive_loss + torch.norm(naive_pip - torch.matmul(src1, src1.T)) \
                   + torch.norm(naive_pip - torch.matmul(src2, src2.T))
            loss = loss + torch.norm(meta_pip - torch.matmul(src1, src1.T)) \
                   + torch.norm(meta_pip - torch.matmul(src2, src2.T))
            print("loss %.3f" % loss.item(),"naive %.3f" % naive_loss.item(),
                              _, epoch_loss)

            epoch_loss = epoch_loss+loss.item()
            loss.backward()
            optmizer.step()
            optmizer.zero_grad()
            loss = 0
            naive_loss=0
        print(epoch_loss)
        loss_list.append(epoch_loss)


    print(loss_list)




    mat1 = proj_mat1.cpu().detach().numpy()
    mat2 = proj_mat2.cpu().detach().numpy()

    src1_vec = sources[0]['vectors']
    src2_vec = sources[1]['vectors']

    src1_vec = np.matmul(src1_vec, mat1)
    src2_vec = np.matmul(src2_vec, mat2)

    res = get_res([i['labels'] for i in sources],[src1_vec,src2_vec])
    np.savez(f'pip_{ares_name}_sensem{step}_e{epoch}.npz', vectors=res, labels=list(ares_dict_src.keys())
             , mat1=mat1, mat2=mat2,loss = loss_list)
    end = time.time()
    print(f"it took {end - start}")
