import numpy as np
import cv2
import pandas as pd
import json

print(123)
img = cv2.imread("vis_45_M180B_3_354.jpg")
img2 = cv2.imread("vis_45_M180B_3_354.jpg")
json_path = '45_M180B_3_354.json'
with open(json_path, "r") as json_file:
    json_data = json.load(json_file)
ori = json_data['annotations']["2d_pos"]
# new = json_data['annotations']["new_2d_pos"]


for i in range(len(ori)):
    cv2.line(img, (ori[i][0],ori[i][1]), (ori[i][0],ori[i][1]), (0, 0, 255), 5)
    # cv2.line(img2, (new[i][0],new[i][1]), (new[i][0],new[i][1]), (255, 0, 255), 5)


# x_min = int(0.)
# y_min = int(310.5401)
# x_max = int(93.91642)
# y_max = int(432.18036)
#
# # 바운딩 박스 그리기
# cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow('image', img)
# cv2.imshow('image2', img2)
cv2.waitKey()
cv2.destroyAllWindows()
# cv2.imwrite('img.png',img)
# cv2.imwrite('img2.png',img2)
