from os import path
import math
import numpy as np
import cv2 as cv
import os
from utils import *
import numpy as np
from skimage.morphology import skeletonize as skelt
from skimage.morphology import thin
from tqdm import tqdm

database="./Union_db3/test/fingerprint/"

def skeletonize(image_input):
    image = np.zeros_like(image_input)
    image[image_input == 0] = 1.0
    output = np.zeros_like(image_input)

    skeleton = skelt(image)

    output[skeleton] = 255
    cv.bitwise_not(output, output)

    return output


def thinning_morph(image, kernel):
    thining_image = np.zeros_like(image)
    img = image.copy()

    while 1:
        erosion = cv.erode(img, kernel, iterations = 1)
        dilatate = cv.dilate(erosion, kernel, iterations = 1)

        subs_img = np.subtract(img, dilatate)
        cv.bitwise_or(thining_image, subs_img, thining_image)
        img = erosion.copy()

        done = (np.sum(img) == 0)

        if done:
          break

    down = np.zeros_like(thining_image)
    down[1:-1, :] = thining_image[0:-2, ]
    down_mask = np.subtract(down, thining_image)
    down_mask[0:-2, :] = down_mask[1:-1, ]
    cv.imshow('down', down_mask)

    left = np.zeros_like(thining_image)
    left[:, 1:-1] = thining_image[:, 0:-2]
    left_mask = np.subtract(left, thining_image)
    left_mask[:, 0:-2] = left_mask[:, 1:-1]
    cv.imshow('left', left_mask)

    cv.bitwise_or(down_mask, down_mask, thining_image)
    output = np.zeros_like(thining_image)
    output[thining_image < 250] = 255

    return output


def getimglabel(path):
  fingerprint = cv.imread(path, cv.IMREAD_GRAYSCALE)
  gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
  gx2, gy2 = gx**2, gy**2
  gm = np.sqrt(gx2 + gy2)
  sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)
  thr = sum_gm.max() * 0.2
  mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
  c = mask
  st_x, st_y, width, height = cv.boundingRect(c)     
  bound_rect = np.array([[[st_x, st_y]], [[st_x + width, st_y]],
                          [[st_x + width, st_y + height]], [[st_x, st_y + height]]])
  x_min, x_max, y_min, y_max = st_x, st_x + width, st_y, st_y + height    

  W = (23, 23)
  gxx = cv.boxFilter(gx2, -1, W, normalize = False)
  gyy = cv.boxFilter(gy2, -1, W, normalize = False)
  gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
  gxx_gyy = gxx - gyy
  gxy2 = 2 * gxy

  orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 
  sum_gxx_gyy = gxx + gyy
  strengths = np.divide(cv.sqrt((gxx_gyy**2 + gxy2**2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy!=0)
  region = fingerprint[y_min+50:y_min+130,x_min+50:x_min+130]
  smoothed = cv.blur(region, (5,5), -1)
  xs = np.sum(smoothed, 1) # the x-signature of the region
  local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
  distances = local_maxima[1:] - local_maxima[:-1]
  ridge_period = np.average(distances)

  or_count = 8
  gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]
  nf = 255-fingerprint
  all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
  y_coords, x_coords = np.indices(fingerprint.shape)
  orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
  filtered = all_filtered[orientation_idx, y_coords, x_coords]
  enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
  return 255-enhanced

files=os.listdir(database)
for f in tqdm(files):
  p1=os.path.join(database,f)
  pics=os.listdir(p1)
  for pic in pics:
    try:
      p2=os.path.join(p1,pic)
      img=getimglabel(p2)
      p3=database+"_label"
      if not os.path.exists(p3):
       os.mkdir(p3)
      p4=os.path.join(p3,f)
      if not os.path.exists(p4):
       os.mkdir(p4)
      cv.imwrite(os.path.join(p4,pic),img)
    except:
      print(p2)

