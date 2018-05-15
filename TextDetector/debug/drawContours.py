import cv2
import numpy as np

img = cv2.imread('round.png', 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)

# square_size = 8
# thresh = np.zeros((square_size, square_size), dtype=np.uint8)
# starting_point = int(square_size / 4)
# for y in range(starting_point, starting_point + int(square_size / 2)):
#     for x in range(starting_point, starting_point + int(square_size / 2)):
#         thresh[y, x] = 255

_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(0, 0))
cnt = contours[0]
# cv2.drawContours(thresh, [cnt], 0, (128,128,128), 3)
rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(thresh,[box],0,(127,127,127),1)
rect_points = cv2.boxPoints(rect)
print(rect_points.shape)
for j in range(4):
    cv2.line(thresh, (rect_points[j, 0], rect_points[j, 1]), (rect_points[(j + 1) % 4, 0], rect_points[(j + 1) % 4, 1]), 127, 1, 8)
print(len(contours))
# print(len(contours[0]))
# print(contours[0])

brect_x, brect_y, brect_w, brect_h = cv2.boundingRect(cnt)

# print(thresh)

cv2.imshow('thresh', thresh[brect_y:brect_y+brect_h, brect_x:brect_x+brect_w])
cv2.waitKey(0)
cv2.destroyAllWindows()