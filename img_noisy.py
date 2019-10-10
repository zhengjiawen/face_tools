import cv2 as cv
import imgaug as ia
from imgaug import augmenters as iaa
import os

ia.seed(4)

base_path = '/data/home/zjw/dataset/face_test2w/'
test_img_path = 'test_sample/'

test_img = '15140106134136.jpg'

output_path = 'aug_test/output/'

img = cv.imread(os.path.join(base_path, test_img_path, test_img), 1)

# img aug
rotate = iaa.Affine(rotate=(-25, 25))
image_aug = rotate.augment_image(img)

img_output = os.path.join(base_path, output_path, 'result1.jpg')
print(img_output)
cv.imwrite(img_output, image_aug)





