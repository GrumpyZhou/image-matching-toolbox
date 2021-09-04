
## Add IMC Support Dev Log
###  Dataset Folder Structure
```
image-matching-toolbox/
|-- data/
	|--datasets/
		|--imc-2021/
			|--phototourism/
			    |--british_museum/
			    |-- ...
			|--pragueparks/
			|--googleurban/			
```

### Results Folder Structure
```
outputs/
|--imc/
	|-- Method_Name
		|-- im{imsize}
			# Dataset
			|--phototourism/
				# Raw matches (without any filtering) per scene
				|--{scene}-matches_raw.h5 
					# Match filtering config
					|--sc{sc_thres}gv{gv_thres}/
						# Submission contents
						|--matches.h5
						|--keypoints.h5
						|--descriptors.h5
						|--config.json
			|--pragueparks/
			|--googleurban/
			
```

### To check and test:
- [ ] Predict and save matches 
- [ ] Extract keypoints and dummy descriptors
- [ ] Geometric verification for custom matches
- [ ] the entry script eval_imc.py 

### To implement:
- [ ] Parse submission json config  from configs/imc/{method}.yml or we have another template per method, e.g., configs/imc/{method}.json. 

### Target running command:
```
python -m immatch.eval_imc --gpu 0 \
    --config 'localize/superglue' \
    --dataset 'phototourism' \
    --prefix 'debug'
	
```