import time

from tqdm import tqdm

import torch
import json

import xml.etree.ElementTree as ET
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch.optim as optim
import random
import logging
import argparse
import lxml.etree
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel, BertForMaskedLM

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
model.eval()
if not torch.cuda.is_available():
    print("Switching to CPU because no GPU !!")
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name())


def load_instances(train_path, keys_path):
    """Parse XML of split set and return list of instances (dict)."""
    train_instances = []
    sense_mapping = get_sense_mapping(keys_path)
    text = read_xml_sents(train_path)
    for sent_idx, sentence in enumerate(text):
        inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': [], 'id': []}
        for e in sentence:
            inst['tokens_mw'].append(e.text)
            inst['lemmas'].append(e.get('lemma'))
            inst['id'].append(e.get('id'))
            inst['pos'].append(e.get('pos'))
            if 'id' in e.attrib.keys():
                inst['senses'].append(sense_mapping[e.get('id')])
            else:
                inst['senses'].append(None)
        inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

        """handling multi-word expressions, mapping allows matching tokens with mw features"""
        idx_map_abs = []
        idx_map_rel = [(i, list(range(len(t.split()))))
                       for i, t in enumerate(inst['tokens_mw'])]
        token_counter = 0
        """converting relative token positions to absolute"""
        for idx_group, idx_tokens in idx_map_rel:
            idx_tokens = [i + token_counter for i in idx_tokens]
            token_counter += len(idx_tokens)
            idx_map_abs.append([idx_group, idx_tokens])
        inst['tokenized_sentence'] = ' '.join(inst['tokens'])
        inst['idx_map_abs'] = idx_map_abs
        inst['idx'] = sent_idx
        train_instances.append(inst)

    return train_instances


def chunks(l, n):
    """Yield successive n-sized chunks from given list."""
    for i in range(0, len(l), n):
        yield l[i:min(i + n, len(l))]


def get_sense_mapping(keys_path):
    sensekey_mapping = {}
    sense2id = {}
    with open(keys_path) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            sensekey_mapping[id_] = keys
    return sensekey_mapping


def load_senseMatrices_npz(path):
    logging.info("Loading Pre-trained Sense Matrices ...")
    A = np.load(path, allow_pickle=True)  # A is loaded a 0d array
    A = np.atleast_1d(A.f.arr_0)  # convert it to a 1d array with 1 element
    A = A[0]  # a dictionary, key is sense id and value is sense matrix
    logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(A))
    return A


def load_weight(path):
    logging.info("Loading Model Parameters W ...")
    weight = np.load(path)
    weight = weight.f.arr_0
    logging.info('Loaded Model Parameters W')
    return weight


def read_xml_sents(xml_path):
    with open(xml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('<sentence '):
                sent_elems = [line]
            elif line.startswith('<wf ') or line.startswith('<instance '):
                sent_elems.append(line)
            elif line.startswith('</sentence>'):
                sent_elems.append(line)
                yield lxml.etree.fromstring(''.join(sent_elems))


def write_to_file(path, mat):
    with open(path, 'w') as f:
        for sen_str, matrix in mat.items():
            matrix_str = ' '.join([str(v) for v in matrix])
            f.write('%s %s\n' % (sen_str, matrix_str))
    logging.info('Written %s' % path)


def get_sk_lemma(sensekey):
    return sensekey.split('%')[0]


def load_gloss_embeddings(path):
    logging.info("Loading Pre-trained Sense Matrices ...")
    loader = np.load(path, allow_pickle=True)  # gloss_embeddings is loaded a 0d array
    loader = np.atleast_1d(loader.f.arr_0)  # convert it to a 1d array with 1 element
    gloss_embeddings = loader[0]  # a dictionary, key is sense id and value is embeddings
    logging.info("Done. Loaded %d gloss embeddings" % len(gloss_embeddings))
    return gloss_embeddings


def get_synonyms(sensekey, word):
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                synonyms_list = synset.lemma_names()
    return synonyms_list


def get_args(
        num_epochs=30,
        emb_dim=300,
        batch_size=64,
        diag=False,
        lr=1e-4
):
    parser = argparse.ArgumentParser(description='Word Sense Mapping')
    parser.add_argument('--glove_embedding_path', default='external/glove/glove.840B.300d.txt')
    parser.add_argument('-epoch', default=5, type=int)
    parser.add_argument('--gloss_embedding_path', default='data/vectors/gloss_embeddings.npz')
    parser.add_argument('-sv_path',
                        help='Path to sense vectors')  # default='data/vectors/senseMatrix.semcor_diagonal_linear_large_bertlast4layers_multiword_{}_20.npz'.format(emb_dim))
    parser.add_argument('-emb_dir', default='part')
    parser.add_argument('-pretrain', default=None)
    # parser.add_argument('--load_weight_path', default='data/vectors/weight.semcor_diagonal_linear_bertlast4layers_multiword_1024_{}_20.npz'.format(emb_dim))
    parser.add_argument('--num_epochs', default=num_epochs, type=int)
    parser.add_argument('--loss', default='standard', type=str, choices=['standard'])
    parser.add_argument('--emb_dim', default=emb_dim, type=int)
    parser.add_argument('-func', default='cos')
    parser.add_argument('--diagonalize', default=diag, type=bool)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--bert', default='large', type=str)
    parser.add_argument('--wsd_fw_path', help='Path to Semcor', required=False,
                        default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('--dataset', default='semcor', help='Name of dataset', required=False,
                        choices=['semcor', 'semcor_omsti'])
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size', required=False)
    parser.add_argument('--lr', type=float, default=lr, help='Learning rate', required=False)
    parser.add_argument('--merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy',
                        required=False,
                        choices=['mean', 'first', 'sum'])
    args = parser.parse_args()
    return args


args = get_args()


# Get embeddings from files
def load_glove_embeddings(fn):
    embeddings = {}
    with open(fn, 'r') as gfile:
        for line in gfile:
            splitLine = line.split(' ')
            word = splitLine[0]
            vec = np.array(splitLine[1:], dtype='float32')
            vec = torch.from_numpy(vec)
            embeddings[word] = vec
    return embeddings


def get_bert_embedding(sent):
    """
    input: a sentence
    output: word embeddigns for the words apprearing in the sentence
    """
    tokenized_text = tokenizer.tokenize("[CLS] {0} [SEP]".format(sent))
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(indexed_tokens))]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    model.to(device)
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    """[1:-1] is used to get rid of CLS] and [SEP]"""
    # res = list(zip(tokenized_text[1:-1], outputs[0].cpu().detach().numpy()[0][1:-1]))
    layers_vecs = np.sum([outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]],
                         axis=0)  ### use the last 4 layers
    res = list(zip(tokenized_text[1:-1], layers_vecs.cpu().detach().numpy()[0][1:-1]))

    """merge subtokens"""
    sent_tokens_vecs = []
    for token in sent.split():
        token_vecs = []
        sub = []
        for subtoken in tokenizer.tokenize(token):
            encoded_token, encoded_vec = res.pop(0)
            sub.append(encoded_token)
            token_vecs.append(encoded_vec)
            merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0)
            if args.bert == 'large':
                merged_vec = torch.from_numpy(merged_vec.reshape(1024, 1)).to(device)
            elif args.bert == 'base':
                merged_vec = torch.from_numpy(merged_vec.reshape(768, 1)).to(device)
        sent_tokens_vecs.append((token, merged_vec))
    return sent_tokens_vecs


def get_id2sks(wsd_eval_keys):
    """Maps ids of split set to sensekeys, just for in-code evaluation."""
    id2sks = {}
    with open(wsd_eval_keys) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            id2sks[id_] = keys
    return id2sks


def load_wsd_fw_set(wsd_fw_set_path):
    """Parse XML of split set and return list of instances (dict)."""
    eval_instances = []
    tree = ET.parse(wsd_fw_set_path)
    for text in tree.getroot():
        for sent_idx, sentence in enumerate(text):
            inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
            for e in sentence:
                inst['tokens_mw'].append(e.text)
                inst['lemmas'].append(e.get('lemma'))
                inst['senses'].append(e.get('id'))
                inst['pos'].append(e.get('pos'))

            inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])
            # handling multi-word expressions, mapping allows matching tokens with mw features
            idx_map_abs = []
            idx_map_rel = [(i, list(range(len(t.split()))))
                           for i, t in enumerate(inst['tokens_mw'])]
            token_counter = 0
            for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
                idx_tokens = [i + token_counter for i in idx_tokens]
                token_counter += len(idx_tokens)
                idx_map_abs.append([idx_group, idx_tokens])

            inst['tokenized_sentence'] = ' '.join(inst['tokens'])
            inst['idx_map_abs'] = idx_map_abs
            inst['idx'] = sent_idx
            eval_instances.append(inst)
    return eval_instances


def create_sup_embedding():
    args = get_args()



    pos_confusion = {}
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        pos_confusion[pos] = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}

    train_path, keys_path = None, None
    if args.dataset == 'semcor':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
        keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
    elif args.dataset == 'semcor_omsti':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
        keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'

    instances = load_instances(train_path, keys_path)

    logging.info('Finish formating training data')


    train_data = {}
    multi = []
    tpoch = tqdm(enumerate(chunks(instances, args.batch_size)))
    for batch_idx, batch in tpoch:

        for sent_info in batch:
            idx_map_abs = sent_info['idx_map_abs']
            sent_bert = get_bert_embedding(sent_info['tokenized_sentence'])

            for mw_idx, tok_idxs in idx_map_abs:
                curr_sense = sent_info['senses'][mw_idx]
                '''check if a word contains sense id'''
                if curr_sense is None:
                    continue
                '''
                for the case of taking multiple words as a instance
                for example, obtaining the embedding for 'too much' instead of two embeddings for 'too' and 'much'
                we use mean to compute the averaged vec for a multiple words expression
                '''
                currVec_c = torch.mean(torch.stack([sent_bert[i][1] for i in tok_idxs]), dim=0).to(device)
                currVec_c = torch.cat((currVec_c, currVec_c), dim=0)
                # print('curr_sense[0]',curr_sense[0])
                # print('curr_sense',curr_sense)
                if len(curr_sense)>=2:
                    print('----------------------------------')
                    multi.append(curr_sense)
                train_data[curr_sense[0]] = currVec_c

    # print(train_data)
    vectors = torch.vstack(list(train_data.values()))
    print(vectors.shape)
    print(multi)
    np.savez('semcor.npz', labels=list(train_data.keys()), vectors=vectors)



create_sup_embedding()
