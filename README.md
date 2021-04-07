

# A Toolbox for Image Feature Matching and Evaluations 
In this repository, we provide **easy interfaces** for several exisiting SoA methods to match image feature correspondences between image pairs.
We provide **scripts to evaluate** their predicted correspondences on common benchmarks for the tasks of image matching, homography estimation and visual localization.

### Regarding Patch2Pix
With this reprository, one can **reproduce** the tables reported in our  paper accepted at CVPR2021: [Patch2Pix: Epipolar-Guided Pixel-Level Correspondences](https://arxiv.org/abs/2012.01909). 
Check [our patch2pix repository](https://github.com/GrumpyZhou/patch2pix) for its training code.


###  Clarification
 All of the supported methods and evaluations are **not implemented from scratch**  by us.  Instead, we modularize their original code to define unified interfaces.
 If you are using the results of a method, **remember to cite the corresponding paper**.
 All credits of the implemetation of those methods  belong to their authors .

## Supported Methods & Evaluations 
Currently **supported methods** :
- Local Feature:
[CAPS](https://arxiv.org/abs/2004.13324), [D2Net](https://arxiv.org/abs/1905.03561),  [R2D2](https://arxiv.org/abs/1906.06195), [SuperPoint](https://arxiv.org/abs/1712.07629)
- Matcher: [SuperGlue](https://arxiv.org/abs/1911.11763)
- Correspondence Network:   [NCNet](https://arxiv.org/abs/1810.10510),  [SparseNCNet](https://arxiv.org/pdf/2004.10566.pdf)
- Local Refinement: [Patch2Pix](https://arxiv.org/abs/2012.01909)

Currently **supported evaluations** :
- Image feature matching on HPatches
- Homography estimation on HPatches
- Visual localization benchmarks: InLoc.

**TODO:  add Aachen evaluation**

## Repository Overview

The repository is structured as follows:
 - **configs/**: Each method has its own .yaml file to configure its testing parameters. 
 - **data/**: All datasets should be placed under this folder following our instructions described in **Data Preparation**.
 - **immatch/**: It contains implementations of method wrappers  and evaluation interfaces.
 - **outputs/**: All evaluation results are supposed to be saved here. One folder per benchmark.
 - **pretrained/**: It contains the pretrained models of the supported methods. 
 - **third_party/**: The real implementation of the supported methods from their original repositories, as git submodules.

##  Installation
To use our repository, first run:
```
git clone git@github.com:GrumpyZhou/image-matching-toolbox.git

# Install submodules non-recursively
git submodule update --init
```
### Setup Running Environment
The code has been tested on Ubuntu 18.04 with Python 3.7 + Pytorch 1.7.0  + CUDA 10.2.  We recommend to use *Anaconda* to manage packages and reproduce the paper results. Run the following lines to automatically setup a ready environment for our code.
```
conda env create -f environment.yml
conda activte immatch

# Install  pycolmap 
git clone --recursive git@github.com:mihaidusmanu/pycolmap.git
pip install ./
```
Otherwise, one can try to download all required packages seperately according to their offical documentation.

**About SparseNCNet environment**:
The **immatch** conda env allows to run all supported methods expect for SparseNCNet. In order to use it, please install another environment according to its official [INSTALL.md](https://github.com/ignacio-rocco/sparse-ncnet/blob/master/INSTALL.md). And run evaluations of SparseNCNet using our code under that env.

### Download Pretrained Models
The following command will download the pretained models and place them to the correct places.
```
cd pretrained/
bash download.sh
```


## Data Preparation
One needs to prepare the target dataset(s) to evaluate on in advance.  All datasets should be placed under  **data/**. Here, we provide brief data instructions of our supported benchmarks.

 - **HPatches** : We recommend to follow [D2Net's instructions](https://github.com/mihaidusmanu/d2-net/tree/master/hpatches_sequences) as we did. The dataset root folder needs to be named as **hpatches-sequences-release/** and contains 52 illumination sequences and 56 viewpoint sequences.
 - **InLoc**: Simply download the [dataset](http://www.ok.sc.e.titech.ac.jp/INLOC/) and place them under a folder named **InLoc/**.

Finally, the **data/** folder should contain look like:
```
data/
|-- datasets/
	|-- hpatches-sequences-release/
	     |-- i_ajuntament/ 
	     |-- ...
	|-- InLoc/
	     |-- database/
	     |-- query/
```


## Evaluations
The following example commands are supposed to be executed under **the repository root**.

### HPatches
Available tasks are image feature matching and homography estimation. One can specify to run on either or both at one time by setting  `--task `  with one of ['matching' , 'homography', 'both']. 

For example, the following command evaluates **SuperPoint** and **NCNet** on **both** tasks using their settings defined under **configs/**:
```
python -m immatch.eval_hpatches --gpu 0 \
    --config  'superpoint' 'ncnet' \
    --task 'both' --save_npy \
    --root_dir . 
```
The following command evaluates **Patch2Pix** on under 3 matching thresholds:
```
python -m immatch.eval_hpatches --gpu 0 \
    --config 'patch2pix' --match_thres 0.25 0.5 0.9  \
    --task 'both' --save_npy \
    --root_dir .     
```

### InLoc
We adopt the public implementation [Hierarchical Localization](https://github.com/cvg/Hierarchical-Localization) to evaluate matches on InLoc benchmark.
The following command evaluates  **Patch2Pix** using its setting defined inside [patch2pix.yml](configs/patch2pix.yml) :
```
# python -m immatch.eval_inloc --gpu 0 \
	--config 'patch2pix' 
```

### Aachen Day and Night 
**Coming later**

