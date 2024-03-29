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
    parser.add_argument('-norm', type=bool,default=True)
    parser.add_argument('-step', default=1)
    parser.add_argument("-path", nargs="+", help="source embedding files")
    # parser.add_argument('-noun',default=False)
    args = parser.parse_args()
    return args

def cos_dis(a:torch.Tensor,b):
    assert a.shape == b.shape
    return -torch.dot(a,b)/(torch.norm(a,2)*torch.norm(b,2))

def load_source_embeddings(sources: list, emb_dim=2048):
    """
    Load all source embeddings and compute the intersection of all vocabularies.
    """
    embeddings = []
    vocab = set(sources[0]['labels'])

    # get the intersection of the vocabs
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
        vecs = {k: v for k, v in zip(np_loader['labels'], np_loader['vectors'])}
        for sense in vocab:
            src[sense_to_ix[sense], :] = vecs[sense]
        embs.append(torch.tensor(src, device=device, dtype=torch.float32))

        del vecs
    return embs, sense_to_ix, ix_to_sense


def get_res(key_list, mat_list, emb_dim=2048):
    vocab = set(key_list[0])
    for skset in key_list:
        vocab = vocab.union(skset)
    print('final vocab size', len(vocab))
    sense_to_ix = {}
    for sense in vocab:
        sense_to_ix[sense] = len(sense_to_ix)
    meta_emb = np.zeros((len(vocab), emb_dim))
    for i in range(len(key_list)):
        vecs = {k: v for k, v in zip(key_list[i], mat_list[i])}
        for sense in vecs:
            meta_emb[sense_to_ix[sense], :] += vecs[sense]
    return meta_emb, list(vocab)


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

    sup_loader = np.load("../train.npz")
    vect3 = torch.tensor(sup_loader['vectors'], requires_grad=False, device=device)
    dict3 = {k: v for k, v in zip(sup_loader['labels'], vect3)}

    optmizer = optim.Adam([proj_mat1, proj_mat2], lr=0.001)
    res = torch.zeros(1)
    res.to(device)
    source_path = args.path
    possible_emb = {'ares','lmms','sensem'}
    for i in source_path:
        assert i in possible_emb, 'please choose source embedding from ares,lmms,sensem'
    path_dict = {
        'ares':np.load('../emb_npz/ares.npz'),
        'lmms':np.load('../emb_npz/lmms2048.npz'),
        'sensem':np.load('../sensembert/sensembert_norm.npz')
    }
    emb_name_dict = {
        'ares': 'n_ares',
        'lmms': 'lmms',
        'sensem': 'sensem'
    }
    emb1_name = emb_name_dict[source_path[0]]
    emb2_name = emb_name_dict[source_path[1]]

    sources = [path_dict[source_path[0]], path_dict[source_path[1]]]
    embs, sense_to_ix, ix_to_sense = load_source_embeddings(sources)
    emb1 = embs[0]
    emb2 = embs[1]

    batch_sz = 10000
    loss = 0
    naive_loss = 0
    epoch = args.e
    loss_list = []
    vocab_size = len(sense_to_ix.keys())
    for _ in range(epoch):
        epoch_loss = 0

        for key in dict3.keys():
            if key not in sense_to_ix.keys():
                continue
            vec1 = emb1[sense_to_ix[key]]
            vec2 = emb2[sense_to_ix[key]]
            # print(vec1.shape) torch.Size([2048])
            meta_vec = torch.matmul(vec1, proj_mat1) + \
                   torch.matmul(vec2, proj_mat2)
            loss = loss+cos_dis(meta_vec,dict3[key])



        epoch_loss =  loss.item()
        loss.backward()
        optmizer.step()
        optmizer.zero_grad()
        loss = 0

        print(epoch_loss)
        loss_list.append(epoch_loss)

    print(loss_list)

    start = time.time()
    mat1 = proj_mat1.cpu().detach().numpy()
    mat2 = proj_mat2.cpu().detach().numpy()

    src1_vec = sources[0]['vectors']
    src2_vec = sources[1]['vectors']

    src1_vec = np.matmul(src1_vec, mat1)
    src2_vec = np.matmul(src2_vec, mat2)

    meta_emb, keys = get_res([i['labels'] for i in sources], [src1_vec, src2_vec])
    np.savez(f'suponly_{emb1_name}_{emb2_name}_e{epoch}.npz', vectors=meta_emb, labels=keys
             , mat1=mat1, mat2=mat2, loss=loss_list)
    end = time.time()
    print(f"it took {end - start}")
