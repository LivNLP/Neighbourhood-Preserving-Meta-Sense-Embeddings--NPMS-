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
    parser.add_argument("-path", nargs="+", help="source embedding files")
    parser.add_argument('-root',default='/LOCAL3/robert/')
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

    proj_mat3 = torch.eye(2048)
    proj_mat3 = proj_mat3.to(device)
    proj_mat3.requires_grad = True



    optmizer = optim.Adam([proj_mat1, proj_mat2,proj_mat3], lr=0.001)
    res = torch.zeros(1)
    res.to(device)
    source_path = args.path
    possible_emb = {'ares', 'lmms', 'sensem'}
    for i in source_path:
        assert i in possible_emb, 'please choose source embedding from ares,lmms,sensem'
    path_dict = {
        'ares': np.load('../emb_npz/ares.npz'),
        'lmms': np.load('../emb_npz/lmms2048.npz'),
        'sensem': np.load('../sensembert/sensembert_norm.npz')
    }

    emb1_name = source_path[0]
    emb2_name = source_path[1]
    emb3_name = source_path[2]

    sources = [path_dict[source_path[0]], path_dict[source_path[1]],path_dict[source_path[2]]]
    embs, sense_to_ix, ix_to_sense = load_source_embeddings(sources)
    emb1 = embs[0]
    emb2 = embs[1]
    emb3 = embs[2]
    batch_sz = 10000

    loss = 0
    naive_loss = 0
    epoch = args.e
    loss_list = []
    vocab_size = len(sense_to_ix.keys())
    for _ in range(epoch):
        epoch_loss = 0


        for i in range(0, vocab_size // batch_sz):
            src1 = emb1[i * batch_sz:(i + 1) * batch_sz, :]

            src2 = emb2[i * batch_sz:(i + 1) * batch_sz, :]

            src3 = emb3[i * batch_sz:(i + 1) * batch_sz, :]

            meta = torch.matmul(src1, proj_mat1) + \
                   torch.matmul(src2, proj_mat2)+\
                   torch.matmul(src3,proj_mat3)
            meta_pip = torch.matmul(meta, meta.T)

            loss = loss + torch.norm(meta_pip - torch.matmul(src1, src1.T)) \
                   + torch.norm(meta_pip - torch.matmul(src2, src2.T))+\
                   torch.norm(meta_pip - torch.matmul(src3, src3.T))
            print("loss %.3f" % loss.item(), _, epoch_loss)



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
    mat3 = proj_mat3.cpu().detach().numpy()

    src1_vec = sources[0]['vectors']
    src2_vec = sources[1]['vectors']
    src3_vec = sources[2]['vectors']

    src1_vec = np.matmul(src1_vec, mat1)
    src2_vec = np.matmul(src2_vec, mat2)
    src3_vec = np.matmul(src3_vec, mat3)

    meta_emb, keys = get_res([i['labels'] for i in sources], [src1_vec, src2_vec,src3_vec])
    np.savez(f'/LOCAL3/robert/piponly_{emb1_name}_{emb2_name}_{emb3_name}_e{epoch}.npz',  vectors=meta_emb, labels=keys
             , mat1=mat1, mat2=mat2,mat3=mat3, loss=loss_list)
    end = time.time()
    print(f"it took {end - start}")
