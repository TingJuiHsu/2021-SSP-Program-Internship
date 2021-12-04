# 2021-SSP-Program-Internship
Machine learning for classifying images by feature engineering in python.

Advisor : Prof. YAMADA Seiji (National Institute of Informatics, http://www.ymd.nii.ac.jp/lab/seiji/en)

Student : Ting-Jui Hsu (National Tsing Hua University, NTHU)

# Dataset : ImageNet (https://image-net.org/index.php)

* I use tiny imagenet 200 dataset. (https://paperswithcode.com/dataset/tiny-imagenet)
* I choose random 5 images to show each class.

# Requirement

* python version > 3.7.3

* library

List in the requirements.

* Install XGBOOST

Follow the link. (https://xgboost.readthedocs.io/en/latest/install.html)

 ------------------------------------------------------

# Phase 1 : Feature Engineering
* TWO image classes 

n01443537	goldfish, Carassius auratus

n01629819	European fire salamander, Salamandra salamandra

* FOUR image classes 

n01855672	goose

n01698640	American alligator, Alligator mississipiensis

n02123394	Persian cat

n03837869	obelisk

* Data augmentation

Random rotation and scale on original data.

After data augmentation, I also apply SIFT and bof to generate feature vectors.

--> The final Data of data augmentation I got is too large to upload... > <

Usage :

python aug.py

* SIFT and bag-of-features 

Usage : (Get the feature vectors, including the image with keypoints and only descriptors.)

python get_bof_feature_vectors.py

* Notice

There are some results I got in the bof_data folder.

-------------------------------------------------------

# Phase 2 : Classification Learning Experiments
* 2-classes classification learning by XGBoost

data for training : In training_data folder

trained model : In model folder

* Usage

data preprocessing > python pre.py (if needed)

main training or testing XGBOOST >>>> python original.py

* 4-classes classification learning by XGBoost

--> I can't have good results on original training manner and I use xgb.train instead of xgb.fit to train the model.

* Usage:

python train_version2.py


* Notice

All the results of model (such as parameters, training results) and trained model are in the '' all_results_during_training '' folder.

 ------------------------------------------------------

# Summary of Phase 2

In XGBOOST, I find that data augmentation can’t enhance the accuracy.

It will be very hard to train when the number of class increased.

The best accuracy is 1 / 0.88 on 2 classes,  while on 4 classes is 1 / 0.605.

It still has severe overfitting on both cases.

 ------------------------------------------------------

# Conclusion: 

In phase 0, I learn how to maintain the github.

In phase 1, I obtain the technique to apply data augmentation, SIFT and BoF on ImageNet.

In phase 2, I acquire how to train and tune the model and compare the results of all parameters.











