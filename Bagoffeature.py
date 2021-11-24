import cv2
import numpy as np
import matplotlib.pyplot as plt

## function of bof, by Ting-Jui Hsu, NTHU, Taiwan, 2021/11/24

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bag_of_features(input_img, save_path):
	image = cv2.imread(input_img)
	#cv2.imshow("ori", image)
	#cv2.waitKey()
	#cv2.destroyAllWindows()

	extractor = cv2.xfeatures2d.SIFT_create()

	keypoint, descriptor = features(image, extractor)

	#print("%sbog_4_%d_%s"%(save_path,num, input_img))
	#k()
	sift_original = cv2.drawKeypoints(image, keypoint[0:], image, color=(255, 0, 255))

	save_des = descriptor

	cv2.imwrite("%sbof_2_%s"%(save_path, input_img),sift_original)
	cv2.imwrite("%sdes_2_%s"%(save_path, input_img),save_des)
	#cv2.imwrite("%sbof_4_%s"%(save_path, input_img),sift_original)
	#cv2.imwrite("%sdes_4_%s"%(save_path, input_img),save_des)

	print("done!")

def main():
	#input_img = 'n01443537_0.JPEG'
	input_img = 'Fast_and_furious_six_ver3.jpg'
	save_path = ''
	bag_of_features(1,input_img,save_path)


if __name__ == '__main__':
	main()



'''
def extract_features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def bag_of_features(num, input_img, save_path):
	image = cv2.imread(input_img)

	extractor = cv2.xfeatures2d.SIFT_create()

	keypoint, descriptor = extract_features(image, extractor)
'''