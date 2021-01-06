import os

import cv2
import numpy as np
import glob


def image_path():
    return os.path.dirname(__file__) + "/images/"


debug = False  # set True for debugging
resize_dimensions = (640, 480)
blob_threshold = 50000  # an estimated value for the kind of blob we want to evaluate
rho_threshold = 25  # used for eliminate lines that close each other, remain only 1


def main():
    for img in glob.glob(image_path() + "*.jpg"):
        image = cv2.imread(img)
        image = cv2.resize(image, resize_dimensions)

        # original image
        if debug:
            cv2.imshow("Original", image)
            cv2.waitKey(0)

        # gray image
        image_sudoku_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if debug:
            cv2.imshow("Gray", image_sudoku_gray)
            cv2.waitKey(0)

        thresh = 127
        im_bw = cv2.threshold(image_sudoku_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        if debug:
            cv2.imshow("Black & White", im_bw)
            cv2.waitKey(0)

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(image_sudoku_gray, 5, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       13, 0)
        if debug:
            cv2.imshow("Threshold", thresh)
            cv2.waitKey(0)

        # find the contours
        (_, contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # evaluate all blobs to find blob with biggest area
        # biggest rectangle in the image could probably be sudoku square if any sudoku exists
        biggest = None
        max_area = 0
        for i in range(len(contours)):
            con = contours[i]
            area = cv2.contourArea(con)
            if area > blob_threshold:
                peri = cv2.arcLength(con, True)
                approx = cv2.approxPolyDP(con, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
                    best_cont = i
        if max_area > 0:
            cv2.polylines(image, [biggest], True, (0, 0, 255), 3)  # Sudoku puzzle(biggest rectangle) will be red

            # Find average width and height to draw rectangles in sudoku later on

            if (abs((biggest[3])[0][0] - (biggest[0])[0][0])) > 100:
                average_width = (abs(((biggest[3])[0][0] - (biggest[0])[0][0]))) / 10
            if (abs((biggest[3])[0][0] - (biggest[1])[0][0])) > 100:
                average_width = (abs(((biggest[3])[0][0] - (biggest[1])[0][0]))) / 10
            if (abs((biggest[3])[0][0] - (biggest[2])[0][0])) > 100:
                average_width = (abs(((biggest[3])[0][0] - (biggest[2])[0][0]))) / 10

            if (abs((biggest[3])[0][1] - (biggest[0])[0][1])) > 100:
                average_height = (abs(((biggest[3])[0][1] - (biggest[0])[0][1]))) / 10
            if (abs((biggest[3])[0][1] - (biggest[1])[0][1])) > 100:
                average_height = (abs(((biggest[3])[0][1] - (biggest[1])[0][1]))) / 10
            if (abs((biggest[3])[0][1] - (biggest[2])[0][1])) > 100:
                average_height = (abs(((biggest[3])[0][1] - (biggest[2])[0][1]))) / 10

            # Mask image and eliminate outer space of the sudoku
            (x, y) = thresh.shape
            mask = np.zeros((x, y), np.uint8)
            mask = cv2.drawContours(mask, contours, best_cont, 255, -1)
            mask = cv2.drawContours(mask, contours, best_cont, 0, 2)
            masked = cv2.bitwise_and(mask, image_sudoku_gray)
            if debug:
                cv2.imshow("Masked", masked)
                cv2.waitKey(0)

            # Blur it
            gaussian_blur = cv2.GaussianBlur(masked, (3, 3), 0)
            if debug:
                cv2.imshow("Gaussian", gaussian_blur)
                cv2.waitKey(0)

            # Apply Canny edge detection
            edges = cv2.Canny(gaussian_blur, 50, 20, apertureSize=3)
            if debug:
                cv2.imshow("Edges", edges)
                cv2.waitKey(0)

            # Apply Hough Line Transform, minimum length of line is 120 pixels
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)

            if lines is not None:

                # Each line in sudoku will be stored here
                new_lines = []

                # Each point inside the sudoku will be stored here
                points = []

                # Used to eliminate closest lines
                horizontal_rhos = []
                vertical_rhos = []

                # Used to find bottom and right line of the sudoku rectangle
                sudoku_bottom_y = 0
                sudoku_right_x = 0

                for la in lines:

                    for rho, theta in la:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * a)
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * a)
                        available = 1
                        if b > 0.5:
                            # It is horizontal
                            for r in horizontal_rhos:
                                if r - rho_threshold < rho < r + rho_threshold:
                                    # Won't add if any line close to this line is already added before
                                    available = 0
                            if available == 1:
                                new_lines.append([rho, theta, 0])
                                horizontal_rhos.append(rho)
                                if y0 > sudoku_bottom_y:
                                    sudoku_bottom_y = y0
                                    sudoku_bottom_line = [rho, theta, 0]
                                if debug:
                                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0))
                                    cv2.imshow("Horizontal line", image)
                                    cv2.waitKey(0)

                        else:
                            # It is vertical
                            for r in vertical_rhos:
                                if abs(r - rho_threshold) < abs(rho) < abs(r + rho_threshold):
                                    # Won't add if any line close to this line is already added before
                                    available = 0
                            if available == 1:
                                new_lines.append([rho, theta, 1])
                                vertical_rhos.append(abs(rho))
                                if abs(x0) > sudoku_right_x:
                                    sudoku_right_x = abs(x0)
                                    sudoku_right_line = [rho, theta, 1]
                                if debug:
                                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0))
                                    cv2.imshow("Vertical line", image)
                                    cv2.waitKey(0)

                for i in range(len(new_lines)):
                    if new_lines[i][2] == 0:
                        for j in range(len(new_lines)):
                            if new_lines[j][2] == 1:
                                theta1 = new_lines[i][1]
                                theta2 = new_lines[j][1]
                                p1 = new_lines[i][0]
                                p2 = new_lines[j][0]
                                xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                                p = np.array([p1, p2])
                                res = np.linalg.solve(xy, p)
                                points.append(res)

                                # Don't draw rectangle if point is on the bottom line or right line of sudoku rectangle.
                                # Because it won't be inside of the sudoku
                                if new_lines[i] != sudoku_bottom_line and new_lines[j] != sudoku_right_line:
                                    cv2.rectangle(image, (int(res[0] + 5), int(res[1] + 5)), (
                                        int(res[0] + int(average_width)), int(res[1] + int(average_height))),
                                                  (255, 0, 0), 2)  # Sudoku rectangles will be blue
                                    if debug:
                                        print("right x : ", sudoku_right_x)
                                        print("bottom y : ", sudoku_bottom_y)
                                        print("res : ", res)

                                        print("right x - res0 = ", sudoku_right_x - res[0])
                                        print("bottom y - res 1 = ", sudoku_bottom_y - res[1])

                                        print("average height ", average_height)
                                        print("average width ", average_width)

                                        print("(right x - res0) / average_width = ",
                                              (sudoku_right_x - res[0]) / average_width)
                                        print("(bottom y - res1) / average_height = ",
                                              (sudoku_bottom_y - res[1]) / average_height)

                                        print("matrix x : ", 9 - int((sudoku_right_x - res[0]) / average_width))
                                        print("matrix y : ", 9 - int((sudoku_bottom_y - res[1]) / average_height))

                # If points length is 100, then I could say the biggest rectangle is most probably a sudoku
                if len(points) != 100:
                    print("Couldn't find sudoku in the image")
                else:
                    print("It is a Sudoku!")
        else:
            print("Couldn't find sudoku in the image")

        cv2.imshow(img, image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
