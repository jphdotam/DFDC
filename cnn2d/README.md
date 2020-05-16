# 2D CNN for Deepfake Detection

This contains code to train the 2D SE-ResNeXt50 frame-by-frame deepfake detection model used as part of our solution. 


## Setup 
A working conda environment can be setup through: `skp/setup.sh`


## Data Preprocessing
Mini videos are generated from the original video dataset using face ROI predictions from MT-CNN, which can be found in `../cnn3d/data/rois/`. Change the paths `ORIGINAL` and `SAVEDIR` in `skp/etl/3_extract_face_rois.py`, then run

```
cd skp/etl ; python 3_extract_face_rois.py 
```

to create the videos. This part can take up to several days. 

## Model Training
Run the following: 

```
cd skp ; python run.py configs/experiments/experiment045.yaml train --gpu 0 --num-workers 4
```

You may need to change the `--gpu` and `--num-workers` arguments to fit your personal configuration. You may also need to change `data_dir` under `dataset` in the YAML file, as well as modify where model checkpoints are saved. This model was able to obtain a loss of 0.2504 on our local validation set. 