# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


# 使用仿射变化进行图像变换(一般用于深度学习图像增强)
# 使用HSV色彩变换(一般用于深度学习图像增强)
import numpy as np
import cv2 as cv


class ImageTransformer:
    """请注意几何变换的顺序"""

    def __init__(self, x):
        self.image = x
        self.matrix = np.eye(3)

    def rotate(self, angle):
        """以(0, 0)为旋转中心, 逆时针

        :param angle: 角度(不是弧度)
        :return: shape[3, 3]
        """
        alpha = np.cos(angle / 180 * np.pi)
        beta = np.sin(angle / 180 * np.pi)
        rotate_matrix = np.array([  # 查公式即可
            [alpha, beta, 0],
            [-beta, alpha, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = rotate_matrix @ self.matrix
        return rotate_matrix

    def scale(self, scale):
        """以(0, 0)点缩放

        :param scale: 缩放比例
        :return: shape[3, 3]
        """
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = scale_matrix @ self.matrix
        return scale_matrix

    def translation(self, x=0, y=0):
        """平移

        :param x: int
        :param y: int
        :return: shape[3, 3]
        """
        translation_matrix = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = translation_matrix @ self.matrix
        return translation_matrix

    def flip_lr(self):
        """以x=0轴左右翻转

        :return: shape[3, 3]
        """
        flip_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = flip_matrix @ self.matrix
        return flip_matrix

    def flip_ud(self):
        """以y=0轴上下翻转

        :return: shape[3, 3]
        """
        flip_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = flip_matrix @ self.matrix
        return flip_matrix

    def shear(self, x_shear, y_shear):
        """剪切变换(不是crop裁剪，不要搞混了)

        :param x_shear: y轴向x正轴倾斜的角度
        :param y_shear: x轴向y正轴倾斜的角度
        :return: shape[3, 3]
        """
        x_shear = np.tan(x_shear * np.pi / 180)
        y_shear = np.tan(y_shear * np.pi / 180)

        shear_matrix = np.array([
            [1, x_shear, 0],
            [y_shear, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = shear_matrix @ self.matrix
        return shear_matrix

    def get_image(self, dsize=None):
        dsize = dsize or (self.image.shape[1], self.image.shape[0])
        return cv.warpAffine(self.image, self.matrix[:2], dsize, borderValue=(114, 114, 114))


def resize_max(image, max_height=None, max_width=None):
    """将图像resize成最大不超过max_height, max_width的图像. (双线性插值)
    用于辅助展示(example()中使用)

    :param image: ndarray[H, W, C]. BGR. uint8
    :param max_width: int
    :param max_height: int
    :return: ndarray[H, W, C]. BGR. uint8"""

    # 1. 输入
    height0, width0 = image.shape[:2]
    max_width = max_width or width0
    max_height = max_height or height0
    # 2. 算法
    ratio = min(max_height / height0, max_width / width0)
    if ratio < 1:
        new_shape = int(round(width0 * ratio)), int(round(height0 * ratio))
        image = cv.resize(image, new_shape, interpolation=cv.INTER_LINEAR)
    return image


def example1():
    """测试功能的完备性"""
    x = cv.imread("images/dog.jpg")
    x = resize_max(x, 800, 800)
    height, width = x.shape[:2]
    cv.imshow("1", x)
    # ------------------------------------- test rotate scale translation
    image_transformer = ImageTransformer(x)
    image_transformer.rotate(90)
    image_transformer.scale(0.5)
    image_transformer.translation(0, width / 2)
    x = image_transformer.get_image()
    x2 = x
    cv.imshow("2", x)
    # ------------------------------------- test flip_lr flip_ud
    image_transformer.flip_lr()
    image_transformer.flip_ud()
    image_transformer.rotate(180)
    x = image_transformer.get_image()
    print(np.all(x == x2))  # True
    cv.imshow("3", x)
    # ------------------------------------- test shear
    image_transformer.shear(45, -45)
    image_transformer.translation(0, height / 2)
    x = image_transformer.get_image()
    cv.imshow("4", x)
    image_transformer.translation(0, -height / 2)
    image_transformer.rotate(-45)
    x = image_transformer.get_image()
    cv.imshow("5", x)
    cv.waitKey(0)


def example2():
    """图像增强示例"""
    pass


if __name__ == "__main__":
    example1()
