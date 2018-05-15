from sys import argv
import numpy as np
import cv2
from sys import argv, exit
from math import sqrt
import pbcvt

POINT_CENTER_DELTA = 27.32

MIN_H_POSPROCESSING_RATIO = 30
Y_S1_REGION = 2.976562

Y_BOT_RATIO = 0.9803

X_IMG_RATIO = 0.306
Y_IMG_RATIO = 0.891

MAX_SIZE_RATIO = 800.586
MIN_SIZE_RATIO = 90500.14

MIN_H_RATIO = 79.55
MAX_H_RATIO = 8.82
X_RIGHT_RATIO = 0.9803

# NiblackVersion
NIBLACK = 0
SAUVOLA = 1
WOLFJOLION = 2

# Front or Rear
FRONT = 0
REAR = 1

def convertToGrayscale_orig(input_img, ratio=0.5):
    output_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
    for y in range(output_img.shape[0]):
        for x in range(output_img.shape[1]):
            pixel = input_img[y, x, :]
            _max = pixel[0]
            _min = pixel[0]
            if pixel[1] > pixel[0]:
                _max = pixel[1]
            else:
                _max = pixel[0]
            if _max < pixel[2]:
                _max = pixel[2]
            else:
                if _min > pixel[2]:
                    _min = pixel[2]
            output_img[y, x] = int(_max * ratio + _min * (1 - ratio))
    return output_img

def convertToGrayscale(input_img, ratio=0.5):
    output_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
    for y in range(output_img.shape[0]):
        for x in range(output_img.shape[1]):
            pixel = input_img[y, x, :]
            _max = pixel[0]
            _min = pixel[0]
            if pixel[1] > pixel[0]:
                _max = pixel[1]
            else:
                _max = pixel[0]
                _min = pixel[1]
            if _max < pixel[2]:
                _max = pixel[2]
            else:
                if _min > pixel[2]:
                    _min = pixel[2]
            output_img[y, x] = int(_max * ratio + _min * (1 - ratio))
    return output_img

def dilateImage(filtered_binary_img, h_size):
    kernel = np.ones((1, h_size), np.uint8)
    return cv2.dilate(filtered_binary_img, kernel)

def minRectFilter1(input_img, contours, min_rects):
    full_mask = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
    min_h = input_img.shape[0] / MIN_H_RATIO
    max_h = input_img.shape[0] / MAX_H_RATIO
    max_w = input_img.shape[1] / X_RIGHT_RATIO

    # p1 is the filter of image region
    p1_x = input_img.shape[1] * X_IMG_RATIO
    p1_y = input_img.shape[0] * Y_IMG_RATIO

    # p2 is the botom region
    p2_y = input_img.shape[0] * Y_BOT_RATIO
    for i in range(len(min_rects)):
        vertices = np.array(np.around(cv2.boxPoints(min_rects[i])), dtype=np.uint32)
        
        max_x = np.max(vertices[:, 0])
        max_y = np.max(vertices[:, 1])
        min_x = np.min(vertices[:, 0])
        min_y = np.min(vertices[:, 1])

        x_A, y_A = min_x, min_y
        x_B, y_B = max_x, min_y
        x_C, y_C = max_x, max_y
        x_D, y_D = min_x, max_y
        
        brect_x, brect_y, brect_w, brect_h = x_A, y_A, x_B - x_A, y_D - y_A
        if brect_h < min_h: # height
            continue
        if brect_h > max_h: # height
            continue
        if min_rects[i][0][0] < p1_x and min_rects[i][0][1] < p1_y: # centre of the min_rects[i]
            continue
        if min_rects[i][0][1] > p2_y:
            continue
        if min_rects[i][0][0] > max_w:
            continue
        if brect_x < 0:
            brect_w += brect_x
            brect_x = 0
        if brect_y < 0:
            brect_h += brect_y
            brect_y = 0
        if brect_x + brect_w >= input_img.shape[1]:
            brect_w = input_img.shape[1] - brect_x - 1
        if brect_y + brect_h >= input_img.shape[0]:
            brect_h = input_img.shape[0] - brect_y - 1

        for j in range(4):
            vertices[j][0] -= brect_x
            vertices[j][1] -= brect_y

        m_mask = full_mask[brect_y:brect_y + brect_h + 1, brect_x:brect_x + brect_w + 1]
        
        local_mask = np.zeros_like(m_mask, dtype=np.uint8)
        for j in range(4):
            cv2.line(local_mask, (vertices[j, 0], vertices[j, 1]), (vertices[(j + 1) % 4, 0], vertices[(j + 1) % 4, 1]), (255), 1, 8)

        first = -1
        last = -1
        count = 0
        points = []
        for y in range(local_mask.shape[0]):
            for x in range(local_mask.shape[1]):
                """ print("Shape:", local_mask.shape)
                print("y:", y)
                print("x:", x)
                print("Value:", local_mask[y, x])
                print("Value Type:", type(local_mask[y, x])) """
                if local_mask[y, x] == 255:
                    if first != -1:
                        last = x
                        count = 2
                    else:
                        first = x
                        count = 1
                if x == local_mask.shape[1] - 1:
                    if count == 1:
                        p_x = first
                        p_y = y
                        points.append([p_y, p_x])
                    if count == 2:
                        for I in range(first, last + 1):
                            p_x = I
                            p_y = y
                            points.append([p_y, p_x])
                    first = -1
                    last = -1
                    count = 0
        for n in range(len(points)):
            m_mask[points[n][0], points[n][1]] = 255
        del points
        cv2.imwrite("py_m_mask/m_mask_" + str(i) + '.png', m_mask)
        cv2.imwrite("py_local_mask/local_mask_" + str(i) + '.png', local_mask)
        cv2.imwrite("py_full_mask/full_mask_" + str(i) + '.png', full_mask)
    return full_mask

def getMinRectangles(input_binary_img):
    _, contours, _ = cv2.findContours(input_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(0, 0))
    min_rects = [None] * len(contours)
    for i in range(len(contours)):
        min_rects[i] = cv2.minAreaRect(contours[i])
    return contours, min_rects

def minRectFilter2(img_height, min_rects2):
    min_h_pos = float(img_height) / MIN_H_POSPROCESSING_RATIO
    min_y1 = float(img_height) / Y_S1_REGION
    for i in range(len(min_rects2)):
        vertices = np.array(np.around(cv2.boxPoints(min_rects2[i])), dtype=np.uint32)
        max_x = np.max(vertices[:, 0])
        max_y = np.max(vertices[:, 1])
        min_x = np.min(vertices[:, 0])
        min_y = np.min(vertices[:, 1])
        x_A, y_A = min_x, min_y
        x_B, y_B = max_x, min_y
        x_C, y_C = max_x, max_y
        x_D, y_D = min_x, max_y
        brect_x, brect_y, brect_w, brect_h = x_A, y_A, x_B - x_A, y_D - y_A
        if brect_h <= min_h_pos:
            min_rects2[i] = None
            continue
        if min_rects2[i][0][1] < min_y1:
            min_rects2[i] = None
            continue
    while None in min_rects2:
            min_rects2.remove(None)
    return min_rects2

def sortByX(line):
    x_values = np.zeros(len(line))
    for i in range(len(line)):
        x_values[i] = line[i][0][0]
    return np.argsort(x_values)

def groupByLine(min_rects, img_height):
    mlines = []
    lines = []
    delta_h = float(img_height) / POINT_CENTER_DELTA
    """ for i in range(len(min_rects)):
        rect = min_rects[i]
        for j in range(len(min_rects)):
            if rect != None and min_rects[j] != None:
                if abs(rect[0][1] - min_rects[j][0][1]) < delta_h:
                    print("Triggerred!")
                    min_rects[j] = None

    while None in min_rects:
            min_rects.remove(None)
    print(len(min_rects)) """

    for i in range(len(min_rects)):
        rect = min_rects[i]
        line = []
        if rect == None:
            continue
        for j in range(len(min_rects)):
            if min_rects[j] == None:
                continue
            if abs(rect[0][1] - min_rects[j][0][1]) < delta_h:
                line.append(min_rects[j])
                min_rects[j] = None
            """ if rect != None and min_rects[j] != None:
                if abs(rect[0][1] - min_rects[j][0][1]) < delta_h:
                    line.append(min_rects[j])
                    min_rects[j] = None """
        if len(line) > 0:
            mline = []
            mindex = sortByX(line)
        for j in range(len(mindex)):
            mline.append(line[mindex[j]])
        lines.append(mline)

    y_values = np.zeros(len(lines))
    for i in range(len(lines)):
        y_values[i] = lines[i][0][0][1]
    y_index = np.argsort(y_values)
    for i in range(len(lines)):
        line = []
        for j in range(len(lines[y_index[i]])):
            line.append(lines[y_index[i]][j])
        mlines.append(line)
    return mlines

def visualiseRotatedRectsByLine(input_img, lines):
    visualised_img = input_img
    colour2 = (0, 255, 0)
    for i in range(len(lines)):
        if i % 2 != 0:
            colour = (255, 0, 255)
        else:
            colour = (255, 0, 0)
        for j in range(len(lines[i])):
            rect_points = cv2.boxPoints(lines[i][j])
            # print(int(round(lines[i][j][0][0])), int(round(lines[i][j][0][1])))
            cv2.circle(visualised_img, (int(round(lines[i][j][0][0])), int(round(lines[i][j][0][1]))), 3, colour2, 2)
            for k in range(4):
                cv2.line(visualised_img, (rect_points[k, 0], rect_points[k, 1]), (rect_points[(k + 1) % 4, 0], rect_points[(k + 1) % 4, 1]), colour, 2, 8)
    # cv2.imshow('visualised_img', visualised_img)
    cv2.imwrite('py_visualised_img.png', visualised_img)
    return visualised_img


if __name__ == '__main__':
    # input_img = cv2.imread('test.jpg')
    input_img = cv2.imread(argv[1])
    grey_img = convertToGrayscale(input_img)
    binary_img = pbcvt.applyBinarisationFilter(grey_img, SAUVOLA, float(0.35), float(100))

    # Compute min_h, max_h, based on input size
    img_size = float(input_img.shape[0] * input_img.shape[1])
    min_size = int(img_size / MIN_SIZE_RATIO)
    max_size = int(img_size / MAX_SIZE_RATIO)

    inverted_binary_img = 255 - binary_img
    filtered_binary_img = pbcvt.removeUnwantedComponents(inverted_binary_img, min_size, max_size)
    for y in range(inverted_binary_img.shape[0]):
        for x in range(inverted_binary_img.shape[1]):
            if inverted_binary_img[y, x] == 0:
                filtered_binary_img[y, x] = 0
    
    h_size = 5
    dilated_binary_img = dilateImage(filtered_binary_img, h_size)
    contours, min_rects = getMinRectangles(dilated_binary_img)
    
    # Create a mask from min_rect and then find another min_rect
    id_type = FRONT
    if id_type == FRONT:
        full_mask = minRectFilter1(input_img, contours, min_rects)
        _, min_rects2 = getMinRectangles(full_mask)
        min_rects2 = minRectFilter2(input_img.shape[0], min_rects2)
        lines = groupByLine(min_rects2, input_img.shape[0])
        visualised_img = visualiseRotatedRectsByLine(input_img, lines)
    
    # cv2.imshow('visualised_img', visualised_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
