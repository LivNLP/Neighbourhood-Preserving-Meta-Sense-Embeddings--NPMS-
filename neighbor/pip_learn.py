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


if __name__ == '__main__':
    #
    args = get_args()
    if torch.cuda.is_available():
        print('device num', torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print('current idx', idx)
        print(torch.cuda.get_device_name(idx))
    device = torch.device('cuda')

    smbert_mat = torch.eye(2048)
    smbert_mat = smbert_mat.to(device)
    smbert_mat.requires_grad = True

    ares_mat = torch.eye(2048)
    ares_mat = ares_mat.to(device)
    ares_mat.requires_grad = True

    optmizer = optim.Adam([ares_mat, smbert_mat], lr=0.001)
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

    smbert_dict_src = {k: v for k, v in zip(smbert['labels'],
                                            torch.tensor(smbert['vectors'], device=device))}
    ares_dict_src = {k: v for k, v in zip(ares['labels'],
                                          torch.tensor(ares['vectors'], device=device))}

    smbert_dict = dict(sorted(smbert_dict_src.items(), key=lambda pair: pair[0]))
    del smbert_dict_src
    ares_dict = {k: ares_dict_src[k] for k in smbert_dict.keys()}

    batch_sz = 10000
    step = args.step
    loss = 0
    naive_loss = 0
    epoch = args.e
    loss_list = []
    for _ in range(epoch):
        epoch_loss = 0
        for i in range(0, len(smbert_dict.keys()) // batch_sz):
            smbert_vec = torch.stack(
                list(smbert_dict.values())[i * batch_sz:(i + 1) * batch_sz])
            ares_vec = torch.stack(
                list(ares_dict.values())[i * batch_sz:(i + 1) * batch_sz])
            # print(smbert_vec.shape)
            tpoch = tqdm(enumerate(range(i, len(smbert_dict.keys()) // batch_sz, step)))
            for idx, j in tpoch:
                smbert_vec_T = torch.stack(
                    list(smbert_dict.values())[j * batch_sz:(j + 1) * batch_sz])
                ares_vec_T = torch.stack(
                    list(ares_dict.values())[j * batch_sz:(j + 1) * batch_sz])
                # print(ares_vec.shape)
                smbert_pip = torch.matmul(smbert_vec, smbert_vec_T.T)
                ares_pip = torch.matmul(ares_vec, ares_vec_T.T)
                meta =  torch.matmul(smbert_vec, smbert_mat) + \
                       torch.matmul(ares_vec, ares_mat)
                meta_T = torch.matmul(smbert_vec_T, smbert_mat) + \
                       torch.matmul(ares_vec_T, ares_mat)
                meta_pip = torch.matmul(meta, meta_T.T)
                naive = smbert_vec + ares_vec
                naive_T = smbert_vec_T+ares_vec_T
                naive_pip = torch.matmul(naive,naive_T.T)
                # print(meta_pip.shape)
                loss = loss + torch.norm(meta_pip - ares_pip) + torch.norm(meta_pip - smbert_pip)
                naive_loss = naive_loss + torch.norm(naive_pip - ares_pip) + torch.norm(naive_pip - smbert_pip)

                tpoch.set_postfix(loss="%.3f" % loss.item(),naive="%.3f" % naive_loss.item(),
                                  ep=_,eploss = epoch_loss)
            # print(ares_mat)
            # print("step back")
            epoch_loss = epoch_loss+loss.item()
            loss.backward()
            optmizer.step()
            optmizer.zero_grad()

            loss = 0
            naive_loss=0
        print(epoch_loss)
        loss_list.append(epoch_loss)

        # if _>=5:
    start = time.time()
    out = []
    for key in ares_dict_src.keys():
        if key in smbert_dict.keys():
            temp = torch.matmul(ares_dict[key], ares_mat) + torch.matmul(smbert_dict[key], smbert_mat)
            out.append(temp.cpu().detach().tolist())
        else:
            temp = torch.matmul(ares_dict_src[key], ares_mat)
            out.append(temp.cpu().detach().tolist())

    np.savez(f'pip_{ares_name}_sensem{step}_e{epoch}.npz', vectors=np.array(out), labels=list(ares_dict_src.keys())
             , mat1=smbert_mat.cpu().detach().numpy(), mat2=ares_mat.cpu().detach().numpy(),loss = loss_list)
    end = time.time()
    print(f"it took {end - start}")
