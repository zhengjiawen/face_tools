import cv2 as cv
import imgaug as ia
import os

base_path = '/data/home/zjw/dataset/face_test2w/'
test_img_path = 'test_sample/'

test_img = '15140106134136.jpg'

output_path = 'aug_test/output/'

img = cv.imread(os.path.join(base_path, test_img_path, test_img), 1)

cv.imwrite(os.path.join(base_path, output_path, 'result1.jpg'), img)





