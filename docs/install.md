##  Installation
To use our repository, first run:
```bash
git clone git@github.com:GrumpyZhou/image-matching-toolbox.git

# Install submodules non-recursively
git submodule update --init
```
### Setup Running Environment
The code has been tested on Ubuntu 18.04 with Python 3.7 + Pytorch 1.7.0  + CUDA 10.2.  We recommend to use *Anaconda* to manage packages and reproduce the paper results. Run the following lines to automatically setup a ready environment for our code.
```bash
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
```bash
cd pretrained/
bash download.sh
```