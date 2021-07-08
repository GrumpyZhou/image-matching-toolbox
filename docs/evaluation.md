## Data Preparation
One needs to prepare the target dataset(s) to evaluate on in advance.  All datasets should be placed under  **data/**. Here, we provide brief data instructions of our supported benchmarks.

 - **HPatches** : We recommend to follow [D2Net's instructions](https://github.com/mihaidusmanu/d2-net/tree/master/hpatches_sequences) as we did. The dataset root folder needs to be named as **hpatches-sequences-release/** and contains 52 illumination sequences and 56 viewpoint sequences.

 - **InLoc**: Simply download the [InLoc](http://www.ok.sc.e.titech.ac.jp/INLOC/) dataset and place them under a folder named **InLoc/**.

 - **Aachen Day and Night**: We follow the instructions provided in [Local Feature Evaluation](https://github.com/tsattler/visuallocalizationbenchmark/tree/master/local_feature_evaluation)  to prepare the data folder. The dataset folder is name as **AachenDayNight/**.  If you want to work on **Aachen v1.1**, you also need to place it under the original version as described in [README_Aachen-Day-Night_v1_1.md](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/README_Aachen-Day-Night_v1_1.md).
 
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

### Long-term Visual Localization
We adopt the public implementation [Hierarchical Localization](https://github.com/cvg/Hierarchical-Localization) to evaluate matches on several long-term visual localization benchmarks, including:
-  InLoc
-  Aachen Day and Night  (original + v1.1)

In the following, we give examples to evaluate  **Patch2Pix** using its setting defined inside [patch2pix.yml](configs/patch2pix.yml).
To use another method, simply replace `--config 'patch2pix'` with the name of its config file name, for example `--config 'patch2pix_superglue'`.
**Notice**, one needs to prepare datasets following the previous section before running the following evaluations.

#### InLoc
```
python -m immatch.eval_inloc --gpu 0\
	--config 'patch2pix' 
```
#### Aachen Day and Night
```
# Original version
python -m immatch.eval_aachen --gpu 0 \
	--config 'patch2pix_superglue' \
	--colmap $COLMAP_PATH$  \
	--benchmark_name 'aachen'

# Version 1.1
python -m immatch.eval_aachen --gpu 0 \
	--config 'patch2pix' \
	--colmap $COLMAP_PATH$  \
	--benchmark_name 'aachen_v1.1'
```

###  RobotCar Season
**Coming later**
