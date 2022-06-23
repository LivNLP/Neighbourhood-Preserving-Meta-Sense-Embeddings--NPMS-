import argparse
import numpy as np
import time
import sys
from wordreps import WordReps
from sklearn.utils.extmath import randomized_svd


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs="+", help="source embedding files")
    parser.add_argument("-o", type=str, default=None,help="output file name.")
    parser.add_argument("-k", default=2048,type=int, help="Dimensionality for SVD")
    parser.add_argument("-m", choices=['svd', 'avg', 'cat'], required=True,
                        help="use svd or avg respectively to perform SVD on the concatenated sources or to use their average, or concatenation")
    args = parser.parse_args()
    if args.m.lower() == "avg":
        avg_baseline(args.i,args.o)
    elif args.m.lower() == "svd":
        svd_baseline(args.i,args.k,args.o)
    elif args.m.lower() == "cat":
        cat_baseline(args.i,args.o)
    else:
        raise "Invalid argument %s" % args.m


def cat_baseline(sources_paths,save_path=None):
    paths = []
    for i in sources_paths:
        paths.append(i)
    loader1 = np.load(paths[0])
    loader2 = np.load(paths[1])
    dict1 = {k: v for k, v in zip(loader1['labels'], loader1['vectors'])}
    dict2 = {k: v for k, v in zip(loader2['labels'], loader2['vectors'])}
    vec_dim = len(list(dict1.values())[0])
    # update dict 1 use dict 2 by concatenate common sense id
    for k, v in dict2.items():
        if k in dict1.keys():
            dict1[k] = np.concatenate((dict1[k], dict2[k]))
        else:
            dict1[k] = np.concatenate((dict2[k], np.zeros_like(dict2[k])))

    # for the keys which only dict1 contains, padding corresponding sense id
    for k in dict1.keys():
        if len(dict1[k])==vec_dim:
            dict1[k] = np.concatenate((dict1[k], np.zeros_like(dict1[k])))

    # If not just used for a intermediate result for SVD, save the embedding
    if save_path!=None:
        np.savez(save_path, vectors=np.array(list(dict1.keys())),
             labels=np.array(list(dict1.keys())))
#     else:
#         np.savez("../part/cat" + "_".join(paths[0].split('_')[2:4]), vectors=np.array(list(dict1.keys())),
#                  labels=np.array(list(dict1.keys())))

    return dict1


def svd_baseline(sources,k = 2048,save_path=None):
    """
    Concatenate all sources and apply SVD to reduce dimensionality to k.
    """
    dict1 = cat_baseline(sources)
    paths = []
    for i in sources:
        paths.append(i)
    print("concatenated M = ",list(dict1.values())[0],len(list(dict1.keys())))
    U, A, VT = randomized_svd(np.vstack(list(dict1.values())), n_components=k, random_state=None)
    # print(U.shape)
    # print(A.shape)
    # print(VT.shape)
    if save_path!=None:
        np.savez(save_path, vectors=U[:, :k],
                 labels=np.array(list(dict1.keys())))
    else:
        np.savez("../part/svd" + "_".join(paths[0].split('_')[2:4]), vectors=U[:, :k],
             labels=np.array(list(dict1.keys())))
    return U[:, :k]  # @ np.diag(A[:k])


def avg_baseline(sources,save_path=None):
    """
    Average the source embeddings.
    """
    paths = []
    for i in sources:
        paths.append(i)
    loader1 = np.load(paths[0])
    loader2 = np.load(paths[1])
    dict1 = {k: v for k, v in zip(loader1['labels'], loader1['vectors'])}
    dict2 = {k: v for k, v in zip(loader2['labels'], loader2['vectors'])}
    for k, v in dict2.items():
        if k in dict1.keys():
            dict1[k] = (dict1[k]+dict2[k])/2
        else:
            dict1[k] = dict2[k]
            
    if save_path!=None:
        np.savez(save_path, vectors=np.array(list(dict1.keys())),
             labels=np.array(list(dict1.keys())))
    else:
        np.savez("../part/avg" + "_".join(paths[0].split('_')[2:4]), vectors=np.array(list(dict1.keys())),
                 labels=np.array(list(dict1.values())))



if __name__ == "__main__":
    command_line()
