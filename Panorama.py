# Authors:
# Inbal Altan, 201643459
# Daniel Ben Yair, 204469118

import cv2
import numpy as np
import sys
from datetime import datetime as dt


def change_height(left, right):
    height_leftImg, width_leftImg = left.shape[:2]  # save height and width for left image
    height_rightImg, width_rightImg = right.shape[:2]  # save height and width for left image
    resize_image_size_left = left
    if height_leftImg != height_rightImg:
        resize_image_size_left = cv2.resize(left, (height_rightImg, width_leftImg))  # if height of the left image is not equal to right image
    return resize_image_size_left                                                    # change to left image to the same height of the right image


def change_shape(img):  # resize the image
    if img.shape[0] > 1000:
        scale_percent = 85  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        return img
    return resized


def crop_black_edge(image):  # Crop black edges
    old = image  # save the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image to black & white
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find what is black
    cnt = contours[0]  # take the first result
    x, y, w, h = cv2.boundingRect(cnt)
    crop = old[y:y + h, x:x + w]
    return crop

now = dt.now()
current_time = now.strftime("%H:%M:%S")
print("Begin Time =", current_time)

left = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)  # load images and convert black & white
right = cv2.imread(sys.argv[2] , cv2.IMREAD_GRAYSCALE)  # load image and convert to black & white
left_original = cv2.imread(sys.argv[1])  # load left image (to display colored image)
right_original = cv2.imread(sys.argv[2])  # load right image (to display colored image)

left = change_height(left, right)  # change height to the black & white image
left_original = change_height(left_original, right_original)  # change height to the original image

left = change_shape(left)  # resize the images (black & white image)
right = change_shape(right)

left_original = change_shape(left_original)  # resize the images (black & white image)
right_original = change_shape(right_original)

sift = cv2.xfeatures2d.SIFT_create()
keypoints_left, descriptors_left = sift.detectAndCompute(left, None)  # sift points and their descriptors in vectors
keypoints_right, descriptors_right = sift.detectAndCompute(right, None)
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
ratio = 0.85
matches = list()

for m1, m2 in raw_matches:
    if m1.distance < ratio * m2.distance:
        matches.append([m1])  # m1 is a good match, save it

imMatches = cv2.drawMatchesKnn(left, keypoints_left, right, keypoints_right, matches, None)  # draw lines between matches

if len(matches) >= 4:
    src_pts = np.float32([keypoints_left[m[0].queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints_right[m[0].trainIdx].pt for m in matches])

    H, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    new_width = right.shape[1] + left.shape[1]

    res = cv2.warpPerspective(right_original, H, (new_width, right_original.shape[1]))
    res[0:left_original.shape[0], 0:left_original.shape[1]] = left_original
    final_result = crop_black_edge(res)

    finish = dt.now()
    total_time = finish - now
    finish_time = finish.strftime("%H:%M:%S")

    print("Finish Time =", finish_time)
    print("Done, total time = ", total_time)

    # cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("imMatches", cv2.WINDOW_NORMAL)
    # cv2.imshow("Final Result", final_result)
    # cv2.imshow("imMatches", imMatches)
    # cv2.waitKey()

    cv2.imwrite(sys.argv[3] + "\\output.jpg", final_result)
else:
    print("Not enough matches found")



