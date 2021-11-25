import glob
import os
import cv2 
import numpy as np
import random

## function of get training data, by Ting-Jui Hsu, NTHU, Taiwan, 2021/11/24

def Data_Generation():
    X_data = [];Y_data = []
    path_data = [];path_label = []

    #files = os.listdir('bof_data/bof_2_data/')
    files = os.listdir('bof_data/des_4_data/')
    for file in files:
        print(file)
        for path in glob.glob('bof_data/des_4_data/' + file + '/*.*'):
            if 'jpg' or 'png' or 'jpeg' in path:
                path_data.append(path)

    ## shuffle the data 
    random.shuffle(path_data)

    for paths in path_data:
        #if 'goldfish' in paths:
        if 'goose' in paths:
            path_label.append(0)
        #elif 'European_fire_salamander' in paths:
        elif 'American_alligator' in paths:
            path_label.append(1)

        elif 'Persian_cat' in paths:
            path_label.append(2)
        elif 'obelisk' in paths:
            path_label.append(3)


        img = cv2.imread(paths)
        img = cv2.resize(img, (128, 30))

        X_data.append(img)


    L = len(path_data)
    print(L)

    Y_data = path_label
 
    X_data = np.array(X_data)#, dtype=float)
    Y_data = np.array(Y_data, dtype='uint8')

    X_train = X_data[0:int(L * 0.7)]
    Y_train = Y_data[0:int(L * 0.7)]

    X_test = X_data[int(L * 0.7):L]#測試資料分配
    Y_test = Y_data[int(L * 0.7):L]

    return X_train, Y_train, X_test, Y_test, L

#X_train, Y_train, X_test, Y_test, L = Data_Generation()

#np.savez(os.path.join('bof_data/des_2_data/', 'data.npz'), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

