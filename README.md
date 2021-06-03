
## SSIM assisted-CycleGAN
Training and testing script for Deep-champ. 
Virtually stain the label-free autofluoroscence image to Hematoxylin and Eosin images

#### Run Testing with pretrained models
The testing data is stored in Data/Fixed/Test/A.
The pretrained model has to be downloaded from the following link, then put the downloaded models into save_models/Fixed/

$ cd Deep_CHAMP

$ python test.py --dataset_name Fixed



#### Run Example for training
$ cd Deep_CHAMP

$ python train.py --dataset_name Fixed
