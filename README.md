
## SSIM assisted-CycleGAN
Training and testing script for Deep-champ. 
Virtually stain the label-free autofluoroscence image to Hematoxylin and Eosin images

#### Run Testing with pretrained models
The testing data is stored in Data/Fixed/Test/A.

The pretrained model has to be downloaded from the following google drive link.
https://drive.google.com/drive/folders/19K7tGo4eQ0cQUACzc-VnRhUTOwjGAcRL?usp=sharing

Click on the link, then right click on the Folder "Fixed" to download the folder,then unzip the folder and put the unzipped folder "Fixed" under "saved_checkpionts"

Now you can run the tetsing.

Testing results will be saved into test_results/Fixed/

$ cd Deep_CHAMP

$ python test.py --dataset_name Fixed



#### Run Example for training
$ cd Deep_CHAMP

$ python train.py --dataset_name Fixed
