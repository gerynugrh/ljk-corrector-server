import cv2
import numpy as np
import base64


class Grader:
    def __init__(self, mat):
        self.mat = mat
        self.result = None

    def grade(self):
        self.result = dict()
        self.result['name'] = self.__get_name()
        self.result['number'] = self.__get_number()
        self.result['answer'] = self.__get_answer()

    def get_result(self):
        if (self.result is None):
            self.grade()
        return self.result

    def __get_answer(self):
        options = 'abcde__'
        answer = ''
        # starting position for each 10-numbers answer
        starts = [
            (39, 4), (49, 4),
            (39, 11), (49, 11),
            (39, 18), (49, 18),
            (39, 25), (49, 25),
            (39, 32), (49, 32),
            (39, 39), (49, 39),
        ]
        for start in starts:
            for i in range(10):
                answer += options[self.__get_option_horz(row=start[0] + i, col=start[1], size=5)]
        return answer

    def __get_name(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyz__'
        name = ''
        for i in range(20):
            name += alphabet[self.__get_option_vert(row=9, col=(26 + i), size=26)]
        return name

    def __get_number(self):
        numbers = '0123456789__'
        number = ''
        for i in [2, 3, 5, 6, 7, 9, 10, 11, 13]:
            number += numbers[self.__get_option_vert(row=9, col=i, size=10)]
        # reformat
        number = number[:2] + '-' + number[2:5] + '-' + number[5:8] + '-' + number[8:]
        return number

    def __get_option_horz(self, row, col, size):
        option = -1
        for i in range(col, col + size):
            if (self.mat[row][i]):
                option = (i - col) if (option == -1) else -2
        return option

    def __get_option_vert(self, row, col, size):
        option = -1
        for i in range(row, row + size):
            if (self.mat[i][col]):
                option = (i - row) if (option == -1) else -2
        return option


class Image:
    IMG_WIDTH = 1000
    LJK_RATIO = 19.2 / 26
    ROW, COL = 66, 47

    def __init__(self, img):
        self.original_image = img
        self.original_height = img.shape[0]
        self.original_width = img.shape[1]
        self.original_ratio = self.original_width * 1.0 / self.original_height
        self.result_image = self.original_image.copy()
        self.working_image = self.original_image.copy()
        self.key_points = None
        self.ljk_mat = None

    def get_base64_result_image(self):
        encoded = cv2.imencode('.png', self.result_image)[1].tostring()
        result = base64.b64encode(encoded).decode('utf-8')
        return str(result)

    def get_result(self):
        grader = Grader(self.ljk_mat)
        result = grader.get_result()
        result['encoded'] = self.get_base64_result_image()
        return result

    def process(self):
        self.__resize()
        self.__threshold()
        self.__find_key_points()
        self.__detect_and_wrap_corner()
        self.__find_key_points(blur=False, update_result=False)
        self.__create_answer_matrix()

    def __create_answer_matrix(self):
        offset = int(self.IMG_WIDTH / (self.COL) / 2)
        self.ljk_mat = [[0 for i in range(self.COL)] for j in range(self.ROW)]
        height = self.working_image.shape[0]
        width = self.working_image.shape[1]

        for key_point in self.key_points:
            point = key_point.pt
            self.ljk_mat[int(point[1] * self.ROW / height)][int(point[0] * self.COL / width)] = 1
        for i in range(0, self.ROW):
            for j in range(0, self.COL):
                if (self.ljk_mat[i][j]):
                    r, c = self.__get_coordinate_from_indices(i, j)
                    cv2.rectangle(self.result_image, (c, r), (c + offset * 2, r + offset * 2), 128, 2)

    def __detect_and_wrap_corner(self):
        points = self.__find_four_key_point()

        # sort clockwise from top-left
        def cmp(point):
            center = self.working_image.shape[1] / 2, self.working_image.shape[0] / 2
            if point[0] < center[0]:
                if (point[1] < center[1]):
                    return 0
                else:
                    return 3
            else:
                if (point[1] < center[1]):
                    return 1
                else:
                    return 2

        points = np.array(sorted(points, key=cmp), dtype=np.float32)
        out_size = (self.IMG_WIDTH, int(self.IMG_WIDTH / self.LJK_RATIO))
        offset = (out_size[0] / self.COL / 2, out_size[1] / self.ROW / 2)
        dst = np.array([
            [offset[0], offset[1]],
            [out_size[0] - offset[0], offset[1]],
            [out_size[0] - offset[0], out_size[1] - offset[1]],
            [offset[0], out_size[1] - offset[1]]],
            dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(points, dst)

        self.working_image = cv2.warpPerspective(self.working_image, matrix, out_size)
        self.result_image = cv2.warpPerspective(self.result_image, matrix, out_size)

    def __find_key_points(self, blur=True, min_circularity=0.75, min_convexity=0.85, update_result=False):
        img = self.working_image.copy()
        if (blur):
            blur_radius = int(((self.IMG_WIDTH * 0.75 / 2) // self.COL) * 2 + 1)
            img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)
        # r, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        img_width = img.shape[1]

        params = cv2.SimpleBlobDetector_Params()

        params.filterByCircularity = True
        params.minCircularity = min_circularity
        params.maxCircularity = 1

        params.filterByConvexity = True
        params.minConvexity = min_convexity

        params.filterByArea = True
        params.minArea = (img_width * 0.65 // self.COL) ** 2

        det = cv2.SimpleBlobDetector_create(params)
        self.key_points = det.detect(img)
        if (update_result):
            self.result_image = img

    def __find_four_key_point(self):
        if (self.key_points is None):
            self.__find_key_points(min_circularity=0.5, min_convexity=0.3)

        if (len(self.key_points) == 0):
            return []

        hulls = cv2.convexHull(cv2.KeyPoint_convert(self.key_points))
        points = [
            [self.working_image.shape[0], self.working_image.shape[1]],
            [0, self.working_image.shape[1]],
            [self.working_image.shape[0], 0],
            [0, 0]
        ]
        for hull in hulls:
            if (hull[0][0] + hull[0][1] < points[0][0] + points[0][1]):
                points[0] = hull[0]
            if (hull[0][0] - hull[0][1] > points[1][0] - points[1][1]):
                points[1] = hull[0]
            if (hull[0][0] - hull[0][1] < points[2][0] - points[2][1]):
                points[2] = hull[0]
            if (hull[0][0] + hull[0][1] > points[3][0] + points[3][1]):
                points[3] = hull[0]

        # edited because of LJK bad format, which doesn't provide
        # right-bottom dots.
        points[3] = [points[1][0], points[2][1]]
        return points

    def __get_coordinate_from_indices(self, row, col):
        t_row = self.working_image.shape[0]
        t_col = self.working_image.shape[1]

        r = int(row * t_row / self.ROW)
        c = int(col * t_col / self.COL)

        return r, c

    def __threshold(self):
        self.working_image = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY)
        blur_radius = int(self.IMG_WIDTH * 0.8 / 2 / self.COL) * 2 + 1
        self.working_image = cv2.GaussianBlur(self.working_image, (blur_radius, blur_radius), 0)
        # _, self.working_image = cv2.threshold(self.working_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # self.working_image = cv2.adaptiveThreshold(
        #     src=self.working_image,
        #     maxValue=255,
        #     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     thresholdType=cv2.THRESH_BINARY,
        #     blockSize=11,
        #     C=2)
        # self.result_image = self.working_image

    def __resize(self):
        self.working_image = cv2.resize(
            self.working_image,
            (self.IMG_WIDTH, int(self.IMG_WIDTH / self.original_ratio)))
        self.result_image = self.working_image.copy()


def eval_img(raw_img):
    img = cv2.imdecode(np.fromstring(raw_img, np.uint8), cv2.IMREAD_UNCHANGED)
    image = Image(img)
    image.process()
    return image.get_result()
