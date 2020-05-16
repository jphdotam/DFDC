conda create -y -n pytorch_p37 python=3.7 pip
source activate pytorch_p37
conda install -y pytorch torchvision cudatoolkit -c pytorch
conda install -y -c conda-forge opencv
conda install -y scikit-learn scikit-image pandas pyyaml tqdm

pip install pretrainedmodels albumentations kaggle decord iterative-stratification
