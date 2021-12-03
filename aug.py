import glob
import os
import cv2 
import numpy as np
import random

## Data augmentation, by Ting-Jui Hsu, NTHU, Taiwan, 2021/12/03

def Data_aug_Generation():
    X_data = [];Y_data = []
    path_data = [];path_label = []

    #files = os.listdir('four_data')
    files = os.listdir('data')

    for file in files:
        print(file)

        #for path in glob.glob('four_data/' + file + '/*.*'):
        for path in glob.glob('data/' + file + '/*.*'):
            if 'jpg' or 'png' or 'jpeg' in path:
                path_data.append(path)

    random.shuffle(path_data)

    random_pick = random.choice([0,1.5,2.5,4])
    degree_pick = random.choice([30, 55, 80])
    scale_pick = random.choice([1.05, 0.98, 0.9])
    for paths in path_data:#生成標籤
        rp = random.choice([0.8,1.5,2.5,4])
        dp = random.choice([30, 55, 80])
        sp = random.choice([1.05, 0.98, 0.9])
        print(paths)
        #if 'goldfish' in paths:
        if 'goose' in paths:
            img = cv2.imread(paths)#      #
            height, width = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),int(rp*dp),sp)
            img_rotate = cv2.warpAffine(img,rotation_matrix,(width,height))#      #
            cv2.imwrite("aug_four_data/%s"%paths,img_rotate)
            #cv2.imshow("ro", img_rotate)
            #cv2.waitKey()
            #path_label.append(0)
        #elif 'European_fire_salamander' in paths:
        elif 'American_alligator' in paths:
            img = cv2.imread(paths)#      #
            height, width = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),int(rp*dp),sp)
            img_rotate = cv2.warpAffine(img,rotation_matrix,(width,height))#      #
            cv2.imwrite("aug_four_data/%s"%paths,img_rotate)
            #cv2.imshow("ro", img_rotate)
            #cv2.waitKey()
            #path_label.append(1)
        elif 'Persian_cat' in paths:
            img = cv2.imread(paths)#      #
            height, width = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),int(rp*dp),sp)
            img_rotate = cv2.warpAffine(img,rotation_matrix,(width,height))#      #
            cv2.imwrite("aug_four_data/%s"%paths,img_rotate)
            #path_label.append(2)
        elif 'obelisk' in paths:
            img = cv2.imread(paths)#      #
            height, width = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),int(rp*dp),sp)
            img_rotate = cv2.warpAffine(img,rotation_matrix,(width,height))#      #
            cv2.imwrite("aug_four_data/%s"%paths,img_rotate)
            #path_label.append(3)

    print("Aug finish!!")


start = Data_aug_Generation()

