import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path

# BGR이미지를 HSL이미지로 변환 및 노락색과 흰색 부분만 검출
def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([0, 150, 0]) # 20, 150, 20
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81]) # 0, 85, 81
    yellow_upper = np.array([190, 255, 255]) # 190, 255, 255

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    
    return masked

# 그레이 스케일, 블러링, 이진화, 엣지 검출
def convert_image(masked_img):
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # th3 = cv2.adaptiveThreshold(src = blurred, 
    #                             maxValue = 255, 
    #                             adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,         
    #                             thresholdType = cv2.THRESH_BINARY, 
    #                             blockSize = 9, 
    #                             C = 3)

    _, thresh = cv2.threshold(src = blurred, 
                              thresh = 128, 
                              maxval = 255, 
                              type = cv2.THRESH_BINARY)

    canny = cv2.Canny(image = thresh, 
                      threshold1 = 500, 
                      threshold2 = 1000, 
                      apertureSize = 5, 
                      L2gradient = True) # 1500, 3000, False

    return canny

# 허프 변환 직선 검출
def hough_transform(canny_image, org_img):
    lines = cv2.HoughLinesP(image = canny_image, 
                            rho = 0.8, 
                            theta = np.pi / 180, 
                            threshold = 50, 
                            minLineLength = 30, 
                            maxLineGap = 10) # 90, 30, 1

    degree_list = []
    new_lines = []
    lines2 = []

    for i in lines:
        if int(i[0][2]) - int(i[0][0]) != 0:
            a = (int(i[0][3]) - int(i[0][1])) / (int(i[0][2]) - int(i[0][0]))
            degree_list.append(a)
            lines2.append(i)

    cnt = Counter(degree_list)
    mode = cnt.most_common(1)

    for j in lines2:
        a = (int(j[0][3]) - int(j[0][1])) / (int(j[0][2]) - int(j[0][0]))
        if mode[0][0] < 0:
            if a > -1.5 and a < -0.5:
                    b = int(j[0][1]) - a * int(j[0][0])
                    cv2.line(org_img, (0, int(b)), (int(org_img.shape[1]), int(a*org_img.shape[1]+b)), (0, 0, 255), 1, cv2.LINE_AA)
                    new_lines.append(j)        
        else:
            if a > 1 and a < 2:
                b = int(j[0][1]) - a * int(j[0][0])
                cv2.line(org_img, (0, int(b)), (int(org_img.shape[1]), int(a*org_img.shape[1]+b)), (0, 0, 255), 1, cv2.LINE_AA)
                new_lines.append(j)

    return new_lines, lines, degree_list

# 왼쪽, 오른쪽 경계 추출
def extract_boundary(new_line_list):
    fitx_list = []

    for i in new_line_list:
        fitx_list.append(i[0][0])
        
    min_x = min(fitx_list)
    max_x = max(fitx_list)

    for j in new_line_list:
        if j[0][0] == min_x:
            left_fitx = j
        elif j[0][0] == max_x:
            right_fitx = j
            
    return left_fitx, right_fitx

# 경계선 연장
def extend_line(image, left_fitx, right_fitx):
    if int(left_fitx[0][2]) - int(left_fitx[0][0]) == 0:
        left_line = np.array([int(left_fitx[0][0]), 0, int(left_fitx[0][0]), image.shape[1]])
    else:
        a = (int(left_fitx[0][3]) - int(left_fitx[0][1])) / (int(left_fitx[0][2]) - int(left_fitx[0][0]))
        b = int(left_fitx[0][1]) - a * int(left_fitx[0][0])
        left_line = np.array([0, int(b), int(image.shape[1]), int(a * image.shape[1] + b)])

    if int(right_fitx[0][2]) - int(right_fitx[0][0]) == 0:
        right_line = np.array([int(right_fitx[0][0]), 0, int(right_fitx[0][0]), image.shape[1]])
    else:
        a = (int(right_fitx[0][3]) - int(right_fitx[0][1])) / (int(right_fitx[0][2]) - int(right_fitx[0][0]))
        b = int(right_fitx[0][1]) - a * int(right_fitx[0][0])
        right_line = np.array([0, int(b), int(image.shape[1]), int(a * image.shape[1] + b)])

    left_fitx_split = np.split(left_line, 2, axis = 0)
    right_fitx_split = np.split(right_line, 2, axis = 0)

    pts = np.concatenate((left_fitx_split, right_fitx_split), axis = 0)

    pts2 = np.array([pts[0], pts[1], pts[3], pts[2]])

    return pts2

# 경계선 내부 채우기
def fill_poly(image, pts2):
    warp_zero = np.zeros([image.shape[0], image.shape[1]])
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    cv2.fillPoly(color_warp, np.int_([pts2]), (216, 168, 74))

    result = cv2.addWeighted(image, 1, color_warp, 0.4, 0, dtype = cv2.CV_8U)
    
    return result, color_warp

# 도로 외 영역 제거
def trim_lane(image, warp):
    
    image[warp == 0] = image[warp == 0] * [0]
    
    return image

# 슬라이딩 윈도우 방식으로 이미지 분할
def sliding_window(image, result_path, file_name):
    
    windowsize = [1080 // 2, 1920 // 4]  # 7 * 3
    height_stepsize = 1080 // 4
    width_stepsize = 1920 // 8
    
    cnt = 0

    for i in range(0, image.shape[0] - height_stepsize, height_stepsize):
        for j in range(0, image.shape[1] - width_stepsize, width_stepsize):
            crop = image[i : i + windowsize[0], j : j + windowsize[1]]
            cv2.imwrite(result_path + file_name[0] + "_" + str(cnt) + ".jpg", crop)
            cnt += 1