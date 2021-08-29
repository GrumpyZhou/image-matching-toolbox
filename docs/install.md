#  Installation

To use our code, first download this repository and initialize the submodules:
```bash
git clone git@github.com:GrumpyZhou/image-matching-toolbox.git

# Install submodules non-recursively
cd image-matching-toolbox/
git submodule update --init
```

Next, download the pretained models and place them to the correct places by running the followings:
```bash
cd pretrained/
bash download.sh
```

## Setup Running Environment
Following the steps to setup the ready environment to run the matching toolbox. The code has been tested on Ubuntu 18.04 with Python 3.7 + Pytorch 1.7.0  + CUDA 10.2.  
### 1. Create the immatch virtual environment
```bash
conda env create -f environment.yml
conda activte immatch
```
Notice, the **immatch** conda env allows to run all supported methods **expect for SparseNCNet**. In order to use it, please install its required dependencies according to its official [installation](https://github.com/ignacio-rocco/sparse-ncnet/blob/master/INSTALL.md),

### 2. Install the immatch toolbox as a python package
```bash
# Install immatch toolbox
cd image-matching-toolbox/
python setup.py develop
```
The developing mode allows you to change the code **without re-installing** it in the environment.  You can also install the matching toolbox to any environment to use it **for your other projects**. 
To **uninstall** it from an environment:
```
pip uninstall immatch
```

### 3.  Install pycolmap 
This package is essential for evaluations on localization benchmarks.
```bash
# Install  pycolmap 
pip install git+https://github.com/mihaidusmanu/pycolmap
```

### 4. Update immatch environment when needed
Incase more packages are needed for new features, one can update your created immatch environment:
#### Option 1: add new libs into [setup.py](../setup.py) (Recommended & Faster)
```bash
# Update immatch toolbox
cd image-matching-toolbox/
python setup.py develop
```

#### Option 2: add new libs into [environment.yml](../environment.yml)
```
conda activate immatch
conda env update --file environment.yml --prune
```
