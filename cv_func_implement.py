# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
import numpy as np
import cv2 as cv
from numpy.linalg import inv


def _invertAffineTransform(matrix):
    """cv.invertAffineTransform()

    :param matrix: shape[2, 3]. float32
    :return: shape[2, 3]
    """
    output = np.eye(3)
    output[:matrix.shape[0]] = matrix
    output = inv(output)
    return output[:matrix.shape[0]]


# matrix = np.array([[0.5, 1, 2], [0.4, 2, 1]])
# print(_invertAffineTransform(matrix))
# print(cv.invertAffineTransform(matrix))

def _warpAffine(x, matrix, dsize=None, flags=None):
    """cv.warpAffine(borderMode=None, borderValue=(114, 114, 114))

    :param x: shape[H, W, C]. uint8
    :param matrix: 仿射矩阵. shape[2, 3]. float32
    :param dsize: Tuple[W, H]. 输出的shape
    :param flags: cv.WARP_INVERSE_MAP. 唯一可选参数
    :return: shape[dsize[1], dsize[0], C]. uint8
    """
    dsize = dsize or (x.shape[1], x.shape[0])
    borderValue = np.array((114, 114, 114), dtype=x.dtype)
    if flags is None or flags & cv.WARP_INVERSE_MAP == 0:
        matrix = _invertAffineTransform(matrix)  # flags没有cv.WARP_INVERSE_MAP时
    i_x, i_y = np.meshgrid(np.arange(dsize[0]), np.arange(dsize[1]))  # np.int32
    src_x = (matrix[0, 0] * i_x + matrix[0, 1] * i_y + matrix[0, 2]).round().astype(np.int32)  # X
    src_y = (matrix[1, 0] * i_x + matrix[1, 1] * i_y + matrix[1, 2]).round().astype(np.int32)  # Y
    src_x_clip = np.clip(src_x, 0, x.shape[1] - 1)
    src_y_clip = np.clip(src_y, 0, x.shape[0] - 1)
    output = np.where(((0 <= src_x) & (src_x < x.shape[1]) & (0 <= src_y) & (src_y < x.shape[0]))[:, :, None],
                      x[src_y_clip, src_x_clip], borderValue[None, None])
    return output


# x0 = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
# matrix = np.array([[1, 1, 100], [1, 2, 80.]], dtype=np.float32)
# y = _warpAffine(x0, matrix, (500, 1000))
# y_ = cv.warpAffine(x0, matrix, (500, 1000), borderValue=(114, 114, 114))
# print(np.all(y == y_))


def _getRotationMatrix2D(center, angle, scale):
    """

    :param center: Tuple[X, Y]. 旋转中心
    :param angle: float. 旋转角度. 正: 逆时针
    :param scale: float. 比例
    :return:
    """
    alpha = scale * np.cos(angle / 180 * np.pi)
    beta = scale * np.sin(angle / 180 * np.pi)
    return np.array([
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
    ], dtype=np.float32)


# print(_getRotationMatrix2D((10, 20), 1, 0.5))
# print(cv.getRotationMatrix2D((10, 20), 1, 0.5))

def _getAffineTransform(src, dst):
    """

    :param src: Tuple[Point, Point, Point]. Point: Tuple[X, Y]
    :param dst: Tuple[Point, Point, Point]
    :return: 仿射矩阵. 第一行生成X, 第二行生成Y.
    """
    src = np.concatenate((src, np.ones(3)[:, None]), -1)  # 加上1，变为方阵
    return dst.T @ inv(src.T)

# src = np.array([[0, 0], [100, 100], [0, 100]], dtype=np.float32)
# dst = np.array([[30, 30], [70, 70], [30, 70]], dtype=np.float32)
# M = cv.getAffineTransform(src, dst)
# M_ = _getAffineTransform(src, dst)
# print(np.all(np.abs(M - M_) < 1e-6))
