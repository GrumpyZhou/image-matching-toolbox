#!/bin/bash

# CAPS:
mkdir caps
gdown --id 1UVjtuhTDmlvvVuUlEq_M5oJVImQl6z1f -O caps/caps-pretrained.pth

# D2Net
mkdir d2net
wget https://dsmn.ml/files/d2-net/d2_tf.pth -O d2net/d2_tf.pth
wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O d2net/d2_tf_no_phototourism.pth

# R2D2 Symbolic  links
ln -s ../third_party/r2d2/models  r2d2

# SparseNCNet
mkdir sparsencnet
wget https://www.di.ens.fr/willow/research/sparse-ncnet/models/sparsencnet_k10.pth.tar  -O sparsencnet/sparsencnet_k10.pth.tar 

# Patch2Pix Symbolic links
ln -s ../third_party/patch2pix/pretrained patch2pix
cd patch2pix
bash download.sh
cd ..

# LoFTR
mkdir loftr
gdown --id 1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY -O loftr/outdoor_ds.ckpt
gdown --id 19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS -O loftr/indoor_ds_new.ckpt