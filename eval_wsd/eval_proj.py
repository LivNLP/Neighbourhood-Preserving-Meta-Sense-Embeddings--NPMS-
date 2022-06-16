

import os
import logging
import argparse
from time import time
from collections import defaultdict
import xml.etree.ElementTree as ET
from functools import lru_cache
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.corpus import wordnet as wn

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s - %(levelname)s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')

def mse(a: torch.Tensor, b):
	assert a.shape == b.shape
	return torch.sum((a - b) ** 2)

def get_args(
		emb_dim=300,
		batch_size=64,
		diag=False
):
	parser = argparse.ArgumentParser(description='WSD Evaluation.',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-glove_embedding_path', default='external/glove/glove.840B.300d.txt')
	parser.add_argument('--gloss_embedding_path', default='data/vectors/gloss_embeddings.npz')
	parser.add_argument('--lmms_embedding_path', default='../bias-sense/data/lmms_2048.bert-large-cased.npz')
	# parser.add_argument('--lmms_embedding_path', default='external/lmms/lmms_1024.bert-large-cased.npz')
	parser.add_argument('--ares_embedding_path', default='external/ares/ares_bert_large.txt')
	parser.add_argument('-sv_path', help='Path to sense vectors', required=False,
						default='data/vectors/senseMatrix.semcor_diagonal_linear_large_bertlast4layers_multiword_{}_50.npz'.format(
							emb_dim))
	parser.add_argument('-load_weight_path',
						default='data/vectors/weight.semcor_diagonal_linear_bertlast4layers_multiword_1024_{}_50.npz'.format(
							emb_dim))
	parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
						default='external/wsd_eval/WSD_Evaluation_Framework/')
	parser.add_argument('-tran',default = 'trans_0.npz')
	parser.add_argument('-test_set', default='senseval2', help='Name of test set', required=False,
						choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'])
	parser.add_argument('-batch_size', type=int, default=batch_size, help='Batch size', required=False)
	parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy',
						required=False)
	parser.add_argument('-ignore_pos', dest='use_pos', action='store_false', help='Ignore POS features', required=False)
	parser.add_argument('-thresh', type=float, default=-1, help='Similarity threshold', required=False)
	parser.add_argument('-k', type=int, default=1, help='Number of Neighbors to accept', required=False)
	parser.add_argument('-quiet', dest='debug', action='store_false', help='Less verbose (debug=False)', required=False)
	parser.add_argument('-device', default='cuda', type=str)
	parser.set_defaults(use_lemma=True)
	parser.set_defaults(use_pos=True)
	parser.set_defaults(debug=False)
	args = parser.parse_args()
	return args


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


def get_id2sks(wsd_eval_keys):
	"""Maps ids of split set to sensekeys, just for in-code evaluation."""
	id2sks = {}
	with open(wsd_eval_keys) as keys_f:
		for line in keys_f:
			id_ = line.split()[0]
			keys = line.split()[1:]
			id2sks[id_] = keys
	return id2sks


def run_scorer(wsd_fw_path, test_set, results_path):
	"""Runs the official java-based scorer of the WSD Evaluation Framework."""
	cmd = 'cd %s && java Scorer %s %s' % (wsd_fw_path + 'Evaluation_Datasets/',
										  '%s/%s.gold.key.txt' % (test_set, test_set),
										  '../../../../' + results_path)
	print(cmd)
	os.system(cmd)


def chunks(l, n):
	"""Yield successive n-sized chunks from given list."""
	for i in range(0, len(l), n):
		yield l[i:i + n]


def str_scores(scores, n=3, r=5):  ###
	"""Convert scores list to a more readable string."""
	return str([(l, round(s, r)) for l, s in scores[:n]])


@lru_cache()
def wn_first_sense(lemma, postag=None):
	pos_map = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}
	first_synset = wn.synsets(lemma, pos=pos_map[postag])[0]
	found = False
	for lem in first_synset.lemmas():
		key = lem.key()
		if key.startswith('{}%'.format(lemma)):
			found = True
			break
	assert found
	return key


def gelu(x):
	""" Original Implementation of the gelu activation function in Google Bert repo when initialy created.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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


def load_lmms(npz_vecs_path):
	lmms = {}
	loader = np.load(npz_vecs_path)
	labels = loader['labels'].tolist()
	vectors = loader['vectors']
	for label, vector in list(zip(labels, vectors)):
		lmms[label] = vector
	return lmms


def get_synonyms_sk(sensekey, word):
	synonyms_sk = []
	for synset in wn.synsets(word):
		for lemma in synset.lemmas():
			if lemma.key() == sensekey:
				for lemma2 in synset.lemmas():
					synonyms_sk.append(lemma2.key())
	return synonyms_sk


def get_sk_pos(sk, tagtype='long'):
	# merges ADJ with ADJ_SAT
	if tagtype == 'long':
		type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
		return type2pos[get_sk_type(sk)]
	elif tagtype == 'short':
		type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
		return type2pos[get_sk_type(sk)]


def get_sk_type(sensekey):
	return int(sensekey.split('%')[1].split(':')[0])


def get_sk_lemma(sensekey):
	return sensekey.split('%')[0]


def get_synonyms(sensekey, word):
	for synset in wn.synsets(word):
		for lemma in synset.lemmas():
			# print('lemma.key', lemma.key())
			if lemma.key() == sensekey:
				synonyms_list = synset.lemma_names()
	return synonyms_list


def get_bert_embedding(sent):
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
	# res = list(zip(tokenized_text[1:-1], outputs[0].cpu().detach().numpy()[0][1:-1])) ## [1:-1] is used to get rid of CLS] and [SEP]
	layers_vecs = np.sum([outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]],
						 axis=0)  ### use the last 4 layers
	res = list(zip(tokenized_text[1:-1], layers_vecs.cpu().detach().numpy()[0][1:-1]))

	## merge subtokens
	sent_tokens_vecs = []
	for token in sent.split():
		token_vecs = []
		sub = []
		for subtoken in tokenizer.tokenize(token):
			encoded_token, encoded_vec = res.pop(0)
			sub.append(encoded_token)
			token_vecs.append(encoded_vec)
			merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0)
			merged_vec = torch.from_numpy(merged_vec)
		sent_tokens_vecs.append((token, merged_vec))

	return sent_tokens_vecs


def load_ares_txt(path):
	sense_vecs = {}
	with open(path, 'r') as sfile:
		for idx, line in enumerate(sfile):
			if idx == 0:
				continue
			splitLine = line.split(' ')
			label = splitLine[0]
			vec = np.array(splitLine[1:], dtype='float32')
			dim = vec.shape[0]
			sense_vecs[label] = vec
	return sense_vecs


class SensesVSM(object):

	def __init__(self, vecs_path: str, normalize=False):
		self.vecs_path = vecs_path
		self.labels = []
		self.matrix = []
		self.indices = {}
		self.ndims = 0

		if self.vecs_path.endswith('.txt'):
			self.load_txt(self.vecs_path)
		elif self.vecs_path.endswith('.npz'):
			print('loading npz file:',self.vecs_path)
			self.my_load_npz(self.vecs_path)
		self.load_aux_senses()

	def my_load_npz(self, npz_vecs_path):
		self.vectors = []
		loader = np.load(npz_vecs_path,allow_pickle=True)
		if False: #npz_vecs_path.find('svd')!=-1 or npz_vecs_path.find('avg_')!=-1:
			self.labels = loader['vocabs'].tolist()
		else:
			self.labels = loader['labels'].tolist()
		self.vectors = np.array(loader['vectors'],dtype = np.float32)
		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}
		self.ndims = self.vectors.shape[1]
		print('The dim of the npz file is',self.ndims)

	def load_txt(self, txt_vecs_path):
		self.vectors = []
		with open(txt_vecs_path, encoding='utf-8') as vecs_f:
			for line_idx, line in enumerate(vecs_f):
				if len(line.split()) == 2:
					print("ignoring the dimension info",line)
					continue
				elems = line.split()
				self.labels.append(elems[0])
				self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))
		self.vectors = np.vstack(self.vectors)
		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}

	def load_npz(self, path):
		self.matrices = []
		logging.info("Loading Pre-trained Sense Matrices ...")
		loader = np.load(path, allow_pickle=True)  # A is loaded a 0d array
		loader = np.atleast_1d(loader.f.arr_0)  # convert it to a 1d array with 1 element
		self.A = loader[0]  # a dictionary, key is sense id and value is sense matrix
		# self.labels = list(ares_embeddings.keys())
		# self.labels = list(lmms_embeddings.keys())

		self.labels_set = set(self.labels)
		self.indices = {l: i for i, l in enumerate(self.labels)}
		logging.info("Done. Loaded %d matrices from Pre-trained Sense Matrices" % len(self.A))

	def load_aux_senses(self):

		self.sk_lemmas = {sk: get_sk_lemma(sk) for sk in self.labels}
		self.sk_postags = {sk: get_sk_pos(sk) for sk in self.labels}

		self.lemma_sks = defaultdict(list)
		for sk, lemma in self.sk_lemmas.items():
			self.lemma_sks[lemma].append(sk)
		self.known_lemmas = set(self.lemma_sks.keys())

		self.sks_by_pos = defaultdict(list)
		for s in self.labels:
			self.sks_by_pos[self.sk_postags[s]].append(s)
		self.known_postags = set(self.sks_by_pos.keys())

	def match_senses(self, context_vec, lemma=None, postag=None, topn=100):
		relevant_sks = []
		sense_scores = []
		for sk in self.labels:
			if (lemma is None) or (self.sk_lemmas[sk] == lemma):
				if (postag is None) or (self.sk_postags[sk] == postag):
					relevant_sks.append(sk)
					sense_vec = self.vectors[self.indices[sk]]
					sense_vec = torch.from_numpy(sense_vec)
					# print('sk',sk)
					# print('sense_vec length',len(sense_vec))
					context_vec = context_vec.to(device)
					sense_vec = sense_vec.to(device)
					
					#print('trans shape',trans.shape,'vec shape',sense_vec.shape)

					sense_vec = (torch.mm(sense_vec.unsqueeze(0),trans).squeeze(0)+bias).squeeze(0)
					#print(sense_vec.shape)
					sim = torch.dot(context_vec, sense_vec)/(context_vec.norm() * sense_vec.norm())
					sense_scores.append(sim)

		matches = list(zip(relevant_sks, sense_scores))
		matches = sorted(matches, key=lambda x: x[1], reverse=True)
		return matches[:topn]
@lru_cache()
def wn_first_sense(lemma, postag=None):
    pos_map = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}
    first_synset = wn.synsets(lemma, pos=pos_map[postag])[0]
    found = False
    for lem in first_synset.lemmas():
        key = lem.key()
        if key.startswith('{}%'.format(lemma)):
            found = True
            break
    assert found
    return key

if __name__ == '__main__':

	args = get_args()

	if torch.cuda.is_available() is False and args.device == 'cuda':
		print("Switching to CPU because no GPU !!")
		args.device = 'cpu'
	else:
		print(torch.cuda.get_device_name())
	device = torch.device(args.device)
	tran_pth = args.tran
	print(tran_pth)
	loader = np.load(tran_pth)
	trans = loader['arr']
	if 'bias' in loader.keys():
		bias = loader['bias']
		print('there is bias parameter')
	else:
		bias = 0
		print('there is no bias parameter')
	trans = torch.tensor(trans).to(device)
	bias = torch.tensor(bias).to(device)
	if 'bert' in loader.keys():
		bert_mat = loader['bert']
		print('there is bert matrix')
		bert_mat = torch.tensor(bert_mat).to(device)
	else:
		bert_mat = None
		print('there is no bert')


	'''
	Load pre-trianed sense embeddings for evaluation.
	Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
	Load fastText static embeddings if required.
	'''
	relu = nn.ReLU(inplace=True)

	# ares_embeddings = load_ares_txt(args.ares_embedding_path)
	# lmms_embeddings = load_lmms(args.lmms_embedding_path)
	senses_vsm = SensesVSM(args.sv_path)
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
	model = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)
	model.eval()

	# gloss_vecs = load_gloss_embeddings(args.gloss_embedding_path)

	'''
	Initialize various counters for calculating supplementary metrics.
	'''
	n_instances, n_correct, n_unk_lemmas, acc_sum = 0, 0, 0, 0
	n_incorrect = 0
	num_options = []
	correct_idxs = []
	failed_by_pos = defaultdict(list)

	pos_confusion = {}
	for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
		pos_confusion[pos] = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}

	'''
	Load evaluation instances and gold labels.
	Gold labels (sensekeys) only used for reporting accuracy during evaluation.
	'''
	wsd_fw_set_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (args.test_set, args.test_set)
	wsd_fw_gold_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (args.test_set, args.test_set)
	id2senses = get_id2sks(wsd_fw_gold_path)
	logging.info('Formating testing data')
	eval_instances = load_wsd_fw_set(wsd_fw_set_path)
	logging.info('Finish formating testing data')

	'''
	Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
	File with predictions is processed by the official scorer after iterating over all instances.
	'''
	count = 0
	results_path = 'data/results/%d.%s.%s.key' % (int(time()), args.test_set, args.merge_strategy)
	with open(results_path, 'w') as results_f:
		for batch_idx, batch in enumerate(chunks(eval_instances, args.batch_size)):

			for sent_info in batch:
				idx_map_abs = sent_info['idx_map_abs']
				sent_bert = get_bert_embedding(sent_info['tokenized_sentence'])

				for mw_idx, tok_idxs in idx_map_abs:
					curr_sense = sent_info['senses'][mw_idx]
					'''check if a word contains sense id'''
					if curr_sense is None:
						continue

					curr_lemma = sent_info['lemmas'][mw_idx]
					curr_postag = sent_info['pos'][mw_idx]

					multi_words = []

					'''
					for the case of taking multiple words as a instance
					for example, obtaining the embedding for 'too much' instead of two embeddings for 'too' and 'much'
					we use mean to compute the averaged vec for a multiple words expression
					'''
					currVec_c = torch.mean(torch.stack([sent_bert[i][1] for i in tok_idxs]), dim=0).to(device)

					if bert_mat is not None:
						aa = currVec_c.reshape(1, -1).to(device)
					# print(aa.shape)
						currVec_c = torch.mm(aa, bert_mat).squeeze(0)
					currVec_c = currVec_c / torch.norm(currVec_c, 2)
					#print(senses_vsm.ndims)
					#if senses_vsm.ndims == 0 or senses_vsm.ndims == 2048:
				         #       pass
                                                #t(senses_vsm.ndims)
						#print('no current concate this time')
						#currVec_c = torch.cat((currVec_c, currVec_c), dim=0)
					#elif senses_vsm.ndims == 4096:
						#currVec_c = torch.cat((currVec_c, currVec_c), dim=0)
					#	currVec_c = torch.cat((currVec_c, currVec_c), dim=0)
					#elif senses_vsm.ndims == 3072:
					#	currVec_c = torch.cat((currVec_c, currVec_c, currVec_c), dim=0)
					#print("currVec_c.shape is",currVec_c.shape)
					matches = senses_vsm.match_senses(currVec_c, lemma=curr_lemma, postag=curr_postag)
					num_options.append(len(matches))
					predict = [sk for sk, sim in matches if sim > args.thresh][:args.k]
					#print('predict', predict)
					if len(predict)==0:
						print('empty')
						predict = [(wn_first_sense(curr_lemma, curr_postag), 1)[0]]
						print(predict)
					if len(predict) > 0:
						results_f.write('{} {}\n'.format(curr_sense, predict[0]))
					# for sense_key in predict:
					# 	results_f.write('%s %s' % (curr_sense, sense_key))
					# results_f.write('\n')

					'''check if our prediction(s) was correct, register POS of mistakes'''
					n_instances += 1
					wsd_correct = False
					gold_sensekeys = id2senses[curr_sense]

					#print('gold_sensekeys', gold_sensekeys)

					if len(set(predict).intersection(set(gold_sensekeys))) > 0:
						n_correct += 1
						wsd_correct = True
					elif len(predict) > 0:
						n_incorrect += 1
					if len(predict) > 0:
						failed_by_pos[curr_postag].append((predict, gold_sensekeys))
					else:
						failed_by_pos[curr_postag].append((None, gold_sensekeys))

					'''register if our prediction belonged to a different POS than gold'''
					if len(predict) > 0:
						pred_sk_pos = get_sk_pos(predict[0])
						gold_sk_pos = get_sk_pos(gold_sensekeys[0])
						pos_confusion[gold_sk_pos][pred_sk_pos] += 1

					# register how far the correct prediction was from the top of our matches
					correct_idx = None
					for idx, (matched_sensekey, matched_score) in enumerate(matches):
						if matched_sensekey in gold_sensekeys:
							correct_idx = idx
							correct_idxs.append(idx)
							break

					acc = n_correct / n_instances
					logging.info('ACC: %.3f (%d %d/%d)' % (
						acc, n_instances, sent_info['idx'], len(eval_instances)))

	precision = (n_correct / (n_correct + n_incorrect)) * 100
	recall = (n_correct / len(id2senses)) * 100
	if (precision + recall) == 0.0:
		f_score = 0.0
	else:
		f_score = 2 * precision * recall / (precision + recall)

	if args.debug:
		logging.info('Supplementary Metrics:')
		logging.info('Avg. correct idx: %.6f' % np.mean(np.array(correct_idxs)))
		logging.info('Avg. correct idx (failed): %.6f' % np.mean(np.array([i for i in correct_idxs if i > 0])))
		logging.info('Avg. num options: %.6f' % np.mean(num_options))
		logging.info('Num. unknown lemmas: %d' % n_unk_lemmas)

		logging.info('POS Failures:')
		for pos, fails in failed_by_pos.items():
			logging.info('%s fails: %d' % (pos, len(fails)))

		logging.info('POS Confusion:')
		for pos in pos_confusion:
			logging.info('%s - %s' % (pos, str(pos_confusion[pos])))

		logging.info('precision: %.1f' % precision)
		logging.info('recall: %.1f' % recall)
		logging.info('f_score: %.1f' % f_score)

	logging.info('Running official scorer ...')
	run_scorer(args.wsd_fw_path, args.test_set, results_path)

	with open("ostiresults/"+str(args.tran).split('/')[-1][:-16]+".txt",'a+') as f:
		f.write(str(args.tran).split('/')[-1][:-4]+" ")
		f.write(str(args.test_set)+" "+str(precision))
		f.write('\n')

