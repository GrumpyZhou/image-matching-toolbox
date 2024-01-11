## Data Preparation
###  Prepare Datasets
One needs to prepare the target dataset(s) to evaluate on in advance.  
All datasets should be placed under  **data/datasets**. Here, we provide brief data instructions of our supported benchmarks.

 - **HPatches** : We recommend to follow [D2Net's instructions](https://github.com/mihaidusmanu/d2-net/tree/master/hpatches_sequences) as we did. The dataset root folder needs to be named as **hpatches-sequences-release/** and contains 52 illumination sequences and 56 viewpoint sequences.

 - **InLoc**: Simply download the [InLoc](http://www.ok.sc.e.titech.ac.jp/INLOC/) dataset and place them under a folder named **InLoc/**.

 - **Aachen Day and Night**: We follow the instructions provided in [Local Feature Evaluation](https://github.com/tsattler/visuallocalizationbenchmark/tree/master/local_feature_evaluation)  to prepare the data folder. The dataset folder is name as **AachenDayNight/**.  If you want to work on **Aachen v1.1**, you also need to place it under the original version as described in [README_Aachen-Day-Night_v1_1.md](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/README_Aachen-Day-Night_v1_1.md).

- **RobotCar Seasons**: Download the [RobotCar Seasons](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/) dataset contents and place them under a folder named **RobotCar/**. 

 - **MegaDepth** and **ScanNet** for relative pose estimation: we followed [LoFTR](https://github.com/zju3dv/LoFTR?tab=readme-ov-file) to setup both datasets for testing, please refer to their repo for details.

### Prepare Image Pairs
To evaluate on visual localization benchmarks, one needs to prepare image pairs that are required by HLoc pipeline in advance.
For convenience, we cached the pairs that are extracted by [HLoc](https://github.com/cvg/Hierarchical-Localization) author Paul-Edouard Sarlin.
You can download them by running from the repository root:
```bash
cd data
bash download.sh
```
Otherwise, feel free to use your own database pairs and query pairs.

### RobotCar Season Queries
The above _download.sh_ also downloads the zip file containing RobotCar Season queries which was pre-extracted using [scripts from the original HLoc](https://github.com/cvg/Hierarchical-Localization/blob/robotcar/pipelines/RobotCar/robotcar_generate_query_list.py).
You need to unzip it and place the folder under _data/RobotCar/queries_.

Finally, the **data/** folder should contain look like:
```
data/
|--	datasets/
	|-- hpatches-sequences-release/
	     |-- i_ajuntament/ 
	     |-- ...
	|-- AachenDayNight/
	     |-- 3D-models/
		     |-- aachen_v_1_1/
	     |-- images/
	     |-- queries/
	|-- InLoc/
	     |-- dataset/
	     	|-- alignments/
		|-- cutouts/
	     |-- query/
    |-- Robotcar/
	     |-- 3D-models/
	     |-- images/
	     |-- intrinsics/
	     |-- queries/
		 ...
|-- pairs/
	|-- aachen
	|-- aachen_v1.1
	|-- inloc
	|-- robotcar
	|-- readme.txt
```

## Evaluations
The following example commands are supposed to be executed under **the repository root**.

### HPatches
Available tasks are image feature matching and homography estimation. One can specify to run on either or both at one time by setting  `--task `  with one of ['matching' , 'homography', 'both']. 

For example, the following command evaluates **SuperPoint** and **NCNet** on **both** tasks using their settings defined under **configs/**:
```python
python -m immatch.eval_hpatches --gpu 0 \
    --config  'superpoint' 'ncnet' \
    --task 'both' --save_npy \
    --root_dir . 
```
The following command evaluates **Patch2Pix** on under 3 matching thresholds:
```python
python -m immatch.eval_hpatches --gpu 0 \
    --config 'patch2pix' --match_thres 0.25 0.5 0.9  \
    --task 'both' --save_npy \
    --root_dir .     
```
### Relative Pose Estimation
To reproduce [AspanFormer results](https://github.com/apple/ml-aspanformer/tree/main?tab=readme-ov-file#evaluation):
```
# MegaDepth

python -m immatch.eval_relapose --config 'aspanformer' --benchmark 'megadepth'

# ScanNet
python -m immatch.eval_relapose --config 'aspanformer' --benchmark 'scannet'
```


### Long-term Visual Localization
We adopt the public implementation [Hierarchical Localization](https://github.com/cvg/Hierarchical-Localization) to evaluate matches on several long-term visual localization benchmarks, including:
-  InLoc
-  Aachen Day and Night  (original + v1.1)
- RobotCar Seasons (v1 + v2) 

Notice, our released scripts are tested on at least one of the following methods:
- **Patch2Pix**
- **Patch2Pix+SuperGlue**
- **SuperGlue**

to verify that they can reproduce their released results on the visual localization benchmark [leader board](https://www.visuallocalization.net/benchmark/).

In the following, we give examples to evaluate  **Patch2Pix** using its setting defined inside [patch2pix.yml](configs/patch2pix.yml).
To use another method, simply replace `--config 'patch2pix'` with the name of its config file name, for example `--config 'patch2pix_superglue'` or `--config 'superglue'`.
**Notice**, one needs to prepare datasets following the previous section before running the following evaluations.

#### InLoc
```python
python -m immatch.eval_inloc --gpu 0\
	--config 'patch2pix' 
```
#### Aachen Day and Night
```python
# Original version
python -m immatch.eval_aachen --gpu 0 \
	--config 'patch2pix_superglue' \
	--colmap $COLMAP_PATH  \
	--benchmark_name 'aachen'

# Version 1.1
python -m immatch.eval_aachen --gpu 0 \
	--config 'patch2pix' \
	--colmap $COLMAP_PATH  \
	--benchmark_name 'aachen_v1.1'
```

###  RobotCar Season
By running the following commant, you will get qurey poses for both RobotCar v1 & v2.  We output those results at the same time, since they only differ in the format rather than the pose results.
Notice, running RobotCar takes rather long time (~days).
For debugging, we suggest using only a subset of the provided pairs for both db &  query pairs.
 
```python
#  Version 1 + 2
python -m immatch.eval_robotcar --gpu 0 \
	--config 'superglue' \
	--colmap  $COLMAP_PATH \
	--benchmark_name 'robotcar'	
```
