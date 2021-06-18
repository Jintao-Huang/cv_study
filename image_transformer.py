# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


# 使用仿射变化进行图像变换(一般用于深度学习图像增强)
# 使用HSV色彩变换(一般用于深度学习图像增强)
import numpy as np
import cv2 as cv


class ImageTransformer:
    """请注意几何变换的顺序.
    注意: 变换顺序: hsv变换, 几何变换"""

    def __init__(self, x):
        self.image = x
        self.matrix = np.eye(3)

    def rotate(self, angle):
        """以(0, 0)为旋转中心, 逆时针

        :param angle: 角度(不是弧度). 0代表不变
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

        :param scale: 缩放比例. 1代表不变.
        :return: shape[3, 3]
        """
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        self.matrix = scale_matrix @ self.matrix
        return scale_matrix

    def translation(self, x, y):
        """平移

        :param x: int. 0代表不变
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

        :param x_shear: y轴向x正轴倾斜的角度. 0代表不变
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

    @staticmethod
    def _hsv_transform(src, hsv_c):
        """hsv色彩变换

        :param src: ndarray[H, W, C]. uint8. const
        :param hsv_c: Tuple[float, float, float]. 1代表不变
        :return: ndarray[H, W, C]. uint8
        """
        hue, sat, val = np.transpose(cv.cvtColor(src, cv.COLOR_BGR2HSV), (2, 0, 1))
        hue = ((hue * hsv_c[0]) % 180).astype(np.uint8)
        sat = np.clip(sat * hsv_c[1], 0, 255).astype(np.uint8)
        val = np.clip(val * hsv_c[2], 0, 255).astype(np.uint8)
        dst = cv.cvtColor(np.stack([hue, sat, val], axis=-1), cv.COLOR_HSV2BGR)
        return dst

    def get_image(self, dsize=None, hsv_c=None, flags=None):
        dsize = dsize or (self.image.shape[1], self.image.shape[0])
        dst = self.image
        if hsv_c and hsv_c != (1, 1, 1):
            dst = self._hsv_transform(dst, hsv_c)
        if not np.all(np.abs(self.matrix[:2] - np.eye(3)[:2]) < 1e-12):  # 矩阵不是eye
            dst = cv.warpAffine(dst, self.matrix[:2], dsize, flags=flags, borderValue=(114, 114, 114))
        return dst


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


def img_transform(src, hsv_c, rotate, scale, shear, flip_lr, flip_ud, translation):
    """以图像中心进行数据增强. 变换顺序与参数顺序一致

    :param src: ndarray[H, W, C]. uint8. BGR
    :param hsv_c: Tuple[H: float, S, V]. (1, 1, 1)为不变
    :param rotate: float. 0为不变. 角度
    :param scale: float. 1为不变
    :param shear: Tuple[shear_x: float, shear_y]. (0, 0)为不变. 角度
    :param flip_lr: bool. False为不变
    :param flip_ud: bool. False为不变
    :param translation: Tuple[X: float, Y]. 相比于原图的比例. (0, 0)为不变
    :return: ndarray[H, W, C]. uint8. BGR
    """
    image_transformer = ImageTransformer(src)
    h, w = src.shape[:2]
    image_transformer.translation(-w / 2, -h / 2)  # 将中心移动到(0, 0)
    image_transformer.rotate(rotate)
    image_transformer.scale(scale)
    image_transformer.shear(*shear)
    image_transformer.flip_lr() if flip_lr else None
    image_transformer.flip_ud() if flip_ud else None
    image_transformer.translation((translation[0] + 0.5) * w, (translation[1] + 0.5) * h)
    return image_transformer.get_image(None, hsv_c)


def img_augment(src, hsv_c, rotate, scale, shear, flip_lr, flip_ud, translation):
    """随机数据增强.

    :param src: ndarray[H, W, C]. uint8. BGR
    :param hsv_c: Tuple[H, S, V]. H,S,V: Union[float, Tuple[float, float]]
    :param rotate: Union[float, Tuple[float, float]]
    :param scale: Union[float, Tuple[float, float]]
    :param shear: Union[float, Tuple[float, float]]
    :param flip_lr: float. 概率
    :param flip_ud: float. 概率
    :param translation: Union[float, Tuple[float, float]]
    :return: ndarray[H, W, C]. uint8. BGR, List[各个增强参数]
    """
    # 参数预处理(都转成Tuple)
    hsv_c = [(-item + 1, item + 1) if isinstance(item, (int, float)) else item for item in hsv_c]
    rotate = (-rotate, rotate) if isinstance(rotate, (int, float)) else rotate
    scale = (-scale + 1, scale + 1) if isinstance(scale, (int, float)) else scale
    shear = (-shear, shear) if isinstance(shear, (int, float)) else shear
    translation = (-translation, translation) if isinstance(translation, (int, float)) else translation
    # 随机
    hsv_c = tuple(np.random.uniform(item[0], item[1]) for item in hsv_c)
    rotate = np.random.uniform(rotate[0], rotate[1])
    scale = np.random.uniform(scale[0], scale[1])
    shear = np.random.uniform(shear[0], shear[1]), \
            np.random.uniform(shear[0], shear[1])
    flip_lr = bool(int(np.random.random() + flip_lr))
    flip_ud = bool(int(np.random.random() + flip_ud))
    translation = np.random.uniform(translation[0], translation[1]), \
                  np.random.uniform(translation[0], translation[1])
    aug_param = hsv_c, rotate, scale, shear, flip_lr, flip_ud, translation
    # 增强
    dst = img_transform(src, *aug_param)
    return dst, aug_param


# ----------------------------------- examples


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
    cv.imshow("2", x)
    x0 = x
    # ------------------------------------- test flip_lr flip_ud
    image_transformer.flip_lr()
    image_transformer.flip_ud()
    image_transformer.rotate(180)
    x = image_transformer.get_image()
    print(np.all(x == x0))  # True
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
    x0 = x
    # ------------------------------------- test hsv_transform
    x = image_transformer.get_image(hsv_c=(0.8, 0.5, 0.5))
    cv.imshow("6", x)
    # ------------------------------------- 变回来
    image_transformer.image = x0
    # image_transformer.matrix = np.linalg.inv(image_transformer.matrix)
    # x = image_transformer.get_image()
    # or
    x = image_transformer.get_image(flags=cv.WARP_INVERSE_MAP)

    cv.imshow("7", x)
    cv.waitKey(0)


def test_hsv():
    x = cv.imread("images/dog.jpg")
    x = resize_max(x, 800, 800)
    image_transformer = ImageTransformer(x)
    for i in range(1, 11):
        i *= 0.2  # 0.2 - 2
        print(i)
        y = image_transformer.get_image(hsv_c=(i, 1, 1))
        cv.imshow("1", y)
        cv.waitKey(0)
    for i in range(1, 11):
        i *= 0.2  # 0.2 - 2
        print(i)
        y = image_transformer.get_image(hsv_c=(1, i, 1))
        cv.imshow("1", y)
        cv.waitKey(0)
    for i in range(1, 11):
        i *= 0.2  # 0.2 - 2
        print(i)
        y = image_transformer.get_image(hsv_c=(1, 1, i))
        cv.imshow("1", y)
        cv.waitKey(0)


def aug_example():
    """图像增强示例"""
    # --------------------- 超参数
    hsv_c = (0.05, 0.7, 0.4)  # (0.015, (0.3, 1.7), 0.4)
    rotate = 3  # (-3, 3)
    scale = 0.6  # (0.4, 1.6)
    shear = 3  # (-3, 3)
    flip_lr = 0.5
    flip_ud = 0.01
    translation = 0.2  # 比例. (-0.2, 0.2)

    # 开始图像增强
    x = cv.imread("images/dog.jpg")
    x = resize_max(x, 800, 800)
    print("hsv_c, rotate, scale, shear, flip_lr, flip_ud, translation")
    np.random.seed(618)
    for i in range(100):
        dst, aug_param = img_augment(x, hsv_c, rotate, scale, shear, flip_lr, flip_ud, translation)
        cv.imshow("1", dst)
        print(*aug_param)
        cv.waitKey(0)


def aug_example2():
    x = cv.imread("images/dog.jpg")
    x = resize_max(x, 800, 800)
    dst = img_transform(x, (1.05, 1.7, 1.4), 3, 0.4, (3, -3), False, False, (0.2, 0.2))
    cv.imshow("1", dst)
    cv.waitKey(0)


if __name__ == "__main__":
    # example1()
    # test_hsv()
    aug_example()
    # aug_example2()
