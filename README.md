# DFDC

![DFDC Pipeline](https://james.dev/img/approach_large.PNG)
 
Dear Kaggle Team,

We hope this code is clear. As we are doctors without formal training in programming our systems are probably a little bit idiosyncratic but hopefully they are clear.

We worked on the 2 different pipelines used by our system separately.

The 3D CNN pathway (the top blue pathway on the figure) is trained using the system outlined in the 'cnn3d' folder.
* Training of the different models was performed across two machines using Pytorch which are evident based on whether the model uses a DataParellel sturcture or not, and whether the paths to the data are in Linux or Windows formatting.
   * A Windows machine with a single NVidia RTX Titan
   * A Linux machine with two NVidia RTX Titans
* Dependencies for training these networks are:
   * tqdm (https://pypi.org/project/tqdm/)
   * facenet_pytorch (https://pypi.org/project/facenet-pytorch/)
   * scikit-video (https://pypi.org/project/scikit-video/)
   * scikit-image (https://pypi.org/project/scikit-image/)
   * PIL/pillow (https://pypi.org/project/Pillow/2.2.1/)
   * numpy (https://pypi.org/project/numpy/)
   * opencv & opencv-python >2 (https://anaconda.org/conda-forge/opencv)
   * pandas (https://pypi.org/project/pandas/)
   * pytorch (https://pytorch.org/)
   * albumentations (https://pypi.org/project/albumentations/)

The 2D CNN pathway (the bottom green pathway on the figure) is trained using the system outlines in the 'cnn2d' folder.
