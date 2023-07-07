# Together We make Sense–Learning of Meta-Sense Embeddings

This code in this repository is related to the ACL 2023 Findings [Paper](https://arxiv.org/abs/2305.19092)

## Preparation

To reproduce our results, you may use the following command to create a conda environment.
```
conda update conda
conda env create -f environment.yml
conda activate npms
```

## Word Sense Disambiguation(WSD)
### Obtain meta embedding
- AVG
```
python3 base_method.py -i emb1 emb2 -o outfile -m avg
```
- CONC
```
python3 base_method.py -i emb1 emb2 -o outfile -m cat
```
- SVD

```
python3 base_method.py -m svd -k 2048 -i emb1 emb2 -o outfile
```
- AEME

The AEME is adopted from the open source for the COLING paper: _Learning Word Meta-Embeddings by Autoencoding_  
```
git clone https://github.com/CongBao/AutoencodedMetaEmbedding.git
```
Follow the instruction in the code and use the following command to generate AEME meta embedding.

```
python run.py -m AAEME -i emb1 emb2 -d emb1_dim emb2_dim -o outfile --embed-dim 2048
```
- NPMS

First go to the neighbor directory.
```
cd neighbor
```
The alpha can be a fixed hyper-parameter specified using following command
```
python3  npms.py -i emb1 emb2  -alpha a 
```
To tune the value of alpha, we need to set the argument hyper to True
```
python3  npms.py -i -path emb1 emb2 -hyper True
```
### Evaluate meta embedding

First switch to the eval_wsd directory
```
cd eval_wsd
```
For AEME and SVD, please run 
```
python3 eval_proj.py -sv_path emb_path -test_set test_set_name -tran projection_matrix_path
```
For any other meta embedding methods
```
python3 eval.py -sv_path emb_path -test_set test_set_name
```


### Word in Context(WiC)
This paper tackle WiC by training a classifier and give prediction
using this model.

```
cd wic
python3 train_classifier.py -sv_path emb_path -out_path model_path -tran projection_matrix_path
python3 eval_classifier_wic.py -sv_path emb_path -clf_path model_path -tran projection_matrix_path
```
