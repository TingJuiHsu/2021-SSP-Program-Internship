import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from Bagoffeature import bag_of_features ## import the function I will use from Bagoffeature.py

## Get feature vectors, by Ting-Jui Hsu, NTHU, Taiwan, 2021/11/24
## Utilize ##

## python get_bof_feature_vectors.py

path_data = []
save_file = ['bof_data/','bof_data/','bof_data/','bof_data/']

files = os.listdir('data')

for file in files:
    print(file)
    for path in glob.glob('data/' + file + '/*.*'):
        if 'jpg' or 'png' or 'jpeg' in path:
            path_data.append(path)

#print(path_data[0])

#img = cv2.imread(path_data[0])
#cv2.imshow("o", img)
#cv2.waitKey()
for paths in path_data:
    print(paths)
    #if 'goldfish' in paths:
    #    bag_of_features(i,paths,save_file[0])

    #elif 'European_fire_salamander' in paths:
    #   bag_of_features(i,paths,save_file[1])

    if 'American_alligator' in paths:
        bag_of_features(paths,save_file[0])

    elif 'goose' in paths:
       bag_of_features(paths,save_file[1])

    elif 'obelisk' in paths:
       bag_of_features(paths,save_file[2])

    elif 'Persian_cat' in paths:
       bag_of_features(paths,save_file[3])

    else:
    	pass


