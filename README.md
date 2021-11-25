# 2021-SSP-Program-Internship
Machine learning for classifying images by feature engineering in python.

Advisor : Prof. YAMADA Seiji (National Institute of Informatics, http://www.ymd.nii.ac.jp/lab/seiji/en)

Student : Ting-Jui Hsu (National Tsing Hua University, NTHU)

# Dataset : ImageNet (https://image-net.org/index.php)

* I use tiny imagenet 200 dataset. (https://paperswithcode.com/dataset/tiny-imagenet)
* I choose random 5 images to show each class.

# Requirement

* python version > 3.7.3

* library:

List in the requirements.

* Install XGBOOST

Follow the link. (https://xgboost.readthedocs.io/en/latest/install.html)

# Phase 1 : Feature Engineering
* TWO image classes 

n01443537	goldfish, Carassius auratus

n01629819	European fire salamander, Salamandra salamandra

* FOUR image classes 

n01855672	goose

n01698640	American alligator, Alligator mississipiensis

n02123394	Persian cat

n03837869	obelisk

* SIFT and bag-of-features 

Usage : (Get the feature vectors, including the image with keypoints and only descriptors.)

python get_bof_feature_vectors.py

* Notice!!

There are some results I got in the bof_data folder.


# Phase 2 : Classification Learning Experiments
* 2-classes classification learning by XGBoost

data for training : In training_data folder

trained model : In model folder

* Usage:

data preprocessing > python pre.py (if needed)

main training or testing XGBOOST >>>> python original.py

* 4-classes classification learning by XGBoost

Didn't finish yet. > <







