# DFDC

![DFDC Pipeline](https://james.dev/img/approach_large.PNG)
 
Dear Kaggle Team,

We hope this code is clear. As we are doctors without formal training our systems are probably a little bit spicy
(esoteric?) but hopefully they are clear.

This is the system for training the SEVEN 3D CNNs we used are trained from.

All the scripts needing to be run are in the '_run' directory, and fall into 2 stages, each in their own directory
with a more indepth readme.

1) Export - These scripts will take us from the original deepfake dataset to ~250,000 MP4 files which are closely
cropped to individual people's faces, tracking a bounding box temporally through time.

2) Train models - self-explanatory. There is 1 python file for each of the 7 models we used in the final submission.

The final models, as well as being attached to the submission notebook, are also available in /data/saved_models
