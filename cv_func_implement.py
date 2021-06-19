# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-16

import numpy as np
import cv2 as cv
from numpy.linalg import inv


def _threshold(src, thresh, max_val, thresh_type):
    """cv.threshold()
    参考: https://docs.opencv.org/4.5.2/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59

    :param src: ndarray[H, W, C]. uint8
    :param thresh: int
    :param max_val: int. cv.THRESH_BINARY, cv.THRESH_BINARY_INV才会用到该参数. 但确实必填参数
    :param thresh_type:
    :return: ndarray[H, W, C]. uint8
    """
    dtype = src.dtype
    if thresh_type == cv.THRESH_BINARY:
        return thresh, np.where(src > thresh, np.asarray(max_val, dtype=dtype), np.asarray(0, dtype=dtype))
    elif thresh_type == cv.THRESH_BINARY_INV:
        return thresh, np.where(src > thresh, np.asarray(0, dtype=dtype), np.asarray(max_val, dtype=dtype))
        # return np.asarray(255, dtype=dtype) - cv.threshold(x, thresh, max_val, cv.THRESH_BINARY)
    elif thresh_type == cv.THRESH_TRUNC:
        return thresh, np.where(src > thresh, np.asarray(thresh, dtype=dtype), src)
    elif thresh_type == cv.THRESH_TOZERO:
        return thresh, np.where(src > thresh, src, 0)
    elif thresh_type == cv.THRESH_TOZERO_INV:
        return thresh, np.where(src > thresh, 0, src)
    else:
        # cv.THRESH_MASK
        # cv.THRESH_OTSU
        # cv.THRESH_TRIANGLE
        raise NotImplementedError


# thresh_type = cv.THRESH_TRUNC
# x0 = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
# y1 = cv.threshold(x0, 100, 0, thresh_type)[1]
# y2 = _threshold(x0, 100, 255, thresh_type)[1]
# print(np.all(y1 == y2))  # True


def _invertAffineTransform(matrix):
    """cv.invertAffineTransform(). 本质是求逆
    参考: https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#ga57d3505a878a7e1a636645727ca08f51

    :param matrix: shape[2, 3]. float32
    :return: shape[2, 3]
    """
    matrix = np.concatenate([matrix, np.array([0, 0, 1], dtype=matrix.dtype)[None]])  # for求逆
    return inv(matrix)[:2]


# matrix = np.array([[0.5, 1, 2], [0.4, 2, 1]])
# print(_invertAffineTransform(matrix))
# print(cv.invertAffineTransform(matrix))


def _remap(src, map1, map2, interpolation):
    """cv.remap(borderMode=None, borderValue=(114, 114, 114))

    :param src: ndarray[H, W, C]. uint8
    :param map1: ndarray[H, W]. X. np.float32
    :param map2: ndarray[H, W]. Y. np.float32
    :return: ndarray[H, W, C]. uint8
    """
    borderValue = np.array((114, 114, 114), dtype=src.dtype)  # 背景填充
    # 最近邻
    if interpolation == cv.INTER_NEAREST:
        map_x = map1.round().astype(np.int32)  # X
        map_y = map2.round().astype(np.int32)  # Y
        map_x_clip = np.clip(map_x, 0, src.shape[1] - 1)  # for索引合法. X
        map_y_clip = np.clip(map_y, 0, src.shape[0] - 1)  # Y
        dst = np.where(((0 <= map_x) & (map_x < src.shape[1]) & (0 <= map_y) & (map_y < src.shape[0]))[:, :, None],
                       src[map_y_clip, map_x_clip], borderValue[None, None])  # 广播机制
    else:
        raise NotImplementedError

    return dst


# matrix = np.array([[0.5, 1, 2], [0.4, 2, 1]])
# src = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
# dsize = src.shape[1], src.shape[0]  # 输出的size
# grid_x, grid_y = np.meshgrid(np.arange(dsize[0], dtype=np.float32),
#                              np.arange(dsize[1], dtype=np.float32))
# matrix = matrix.astype(np.float32)
# src_x = matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]  # X
# src_y = matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]  # Y
#
# y = cv.remap(src, src_x, src_y, cv.INTER_NEAREST, borderValue=(114, 114, 114))
# y2 = _remap(src, src_x, src_y, cv.INTER_NEAREST)
# print(np.all(y == y2))

def _resize(src, dsize, interpolation):
    """cv.resize(). 可以通过放射变化实现

    :param src: shape[H, W ,C]
    :param dsize: Tuple[W, H]. out. H: int
    :param interpolation: cv.INTER_NEAREST, cv.INTER_LINEAR
    :return: shape[H, W ,C]
    """
    h_in, w_in = src.shape[:2]
    scale = w_in / dsize[0], h_in / dsize[1]  # 相当于仿射变换中matrix[0, 0], matrix[1, 1]. X, Y
    grid_x, grid_y = np.meshgrid(np.arange(dsize[0], dtype=np.float32),
                                 np.arange(dsize[1], dtype=np.float32))
    # map_x, map_y 代表着将 dst中的点(int32)映射到src中的点(float32)
    # src中的点(float32)再通过插值方式转变成(int32). 即remap中的interpolation参数
    map_x = grid_x * scale[0] + (scale[0] - 1) / 2
    map_y = grid_y * scale[1] + (scale[1] - 1) / 2
    dst = cv.remap(src, map_x, map_y, interpolation, borderMode=cv.BORDER_REPLICATE)
    return dst


# np.random.seed(0)
# x = np.random.randint(0, 256, (2, 2), dtype=np.uint8)
# y = cv.resize(x, (3, 3), interpolation=cv.INTER_LINEAR)
# y_ = _resize(x, (3, 3), interpolation=cv.INTER_LINEAR)
# print(x)
# print(y)
# print(y_)
# print(np.all(y == y_))


def _warpAffine(src, matrix, dsize=None, flags=None):
    """cv.warpAffine(borderMode=None, borderValue=(114, 114, 114))
    参考: https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983

    :param src: shape[H, W, C]. uint8
    :param matrix: 仿射矩阵. shape[2, 3]. float32
    :param dsize: Tuple[W, H]. 输出的size
    :param flags: cv.WARP_INVERSE_MAP. 唯一可选参数. 其他参数忽略(不支持)同CV
    :return: shape[dsize[1], dsize[0], C]. uint8
    """
    dsize = dsize or (src.shape[1], src.shape[0])  # 输出的size
    borderValue = (114, 114, 114)
    if flags is None or flags & cv.WARP_INVERSE_MAP == 0:  # flags无cv.WARP_INVERSE_MAP参数
        matrix = cv.invertAffineTransform(matrix)
    grid_x, grid_y = np.meshgrid(np.arange(dsize[0], dtype=np.float32),
                                 np.arange(dsize[1], dtype=np.float32))
    matrix = matrix.astype(np.float32)
    # map_x, map_y 代表着将 dst中的点(int32)映射到src中的点(float32)
    # src中的点(float32)再通过插值方式转变成(int32). 即remap中的interpolation参数
    map_x = matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]  # X. dst -> src
    map_y = matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]  # Y. dst -> src
    # 也可以这样实现，是等价的(将点乘加转化为矩阵乘)
    # map_x, map_y = np.transpose(  # transpose把2的维度提上来
    #     (matrix @ np.stack([grid_x, grid_y, np.ones_like(grid_x)], -1)[..., None])[..., 0],(2, 0, 1))
    dst = cv.remap(src, map_x, map_y, cv.INTER_LINEAR, borderValue=borderValue)
    return dst


# x0 = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
# matrix = np.array([[1, 0.5, 100], [1, 2, 80.]], dtype=np.float32)
# y = _warpAffine(x0, matrix, (500, 1000), cv.INTER_LINEAR)
# y_ = cv.warpAffine(x0, matrix, (500, 1000), cv.INTER_NEAREST, borderValue=(114, 114, 114))
# print(np.all(y == y_))


def _warpPerspective(src, matrix, dsize=None, flags=None):
    """cv.warpPerspective(borderMode=None, borderValue=(114, 114, 114))
    参考: https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87

    :param src: shape[H, W, C]. uint8
    :param matrix: 仿射矩阵. shape[2, 3]. float32
    :param dsize: Tuple[W, H]. 输出的size
    :param flags: cv.WARP_INVERSE_MAP. 唯一可选参数
    :return: shape[dsize[1], dsize[0], C]. uint8
    """
    dsize = dsize or (src.shape[1], src.shape[0])  # 输出的size
    borderValue = (114, 114, 114)
    if flags is None or flags & cv.WARP_INVERSE_MAP == 0:  # flags无cv.WARP_INVERSE_MAP参数
        matrix = cv.invert(matrix)[1]
    grid_x, grid_y = np.meshgrid(np.arange(dsize[0], dtype=np.float32),
                                 np.arange(dsize[1], dtype=np.float32))
    src_x = ((matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]) /
             (matrix[2, 0] * grid_x + matrix[2, 1] * grid_y + matrix[2, 2]))  # X
    src_y = (matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2] /
             (matrix[2, 0] * grid_x + matrix[2, 1] * grid_y + matrix[2, 2]))  # Y
    dst = cv.remap(src, src_x, src_y,
                   flags if flags is None else flags & ~cv.WARP_INVERSE_MAP, borderValue=borderValue)
    return dst


# x0 = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
# matrix = np.array([[1, 1, 100], [1, 2, 80.], [0, 0, 1]], dtype=np.float32)
# y = _warpPerspective(x0, matrix, (500, 1000), cv.WARP_INVERSE_MAP)
# y_ = cv.warpPerspective(x0, matrix, (500, 1000), flags=cv.WARP_INVERSE_MAP, borderValue=(114, 114, 114))
# y__ = cv.warpAffine(x0, matrix[:2], (500, 1000), flags=cv.WARP_INVERSE_MAP, borderValue=(114, 114, 114))
# print(np.all((y == y_) & (y_ == y__)))


def _getRotationMatrix2D(center, angle, scale):
    """cv.getRotationMatrix2D()
    参考: https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326

    :param center: Tuple[X, Y]. 旋转中心
    :param angle: float. 旋转角度. 正: 逆时针
    :param scale: float. 比例
    :return:
    """
    # 中心点为(0, 0)则无需平移: 即 M[:, 2] 都为0
    alpha = scale * np.cos(angle / 180 * np.pi)
    beta = scale * np.sin(angle / 180 * np.pi)
    return np.array([  # 查公式即可
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
    ], dtype=np.float32)


# print(_getRotationMatrix2D((10, 20), 1, 0.5))
# print(cv.getRotationMatrix2D((10, 20), 1, 0.5))


def _getAffineTransform(src, dst):
    """cv.getAffineTransform()
    参考: https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999

    :param src: Tuple[Point, Point, Point]. Point: Tuple[X, Y]
    :param dst: Tuple[Point, Point, Point]
    :return: 仿射矩阵. 第一行生成X, 第二行生成Y.
    """
    src = np.concatenate((src, np.ones(3)[:, None]), -1)  # 填充1，变为方阵
    return dst.T @ inv(src.T)


# src = np.array([[0, 0], [100, 100], [0, 100]], dtype=np.float32)
# dst = np.array([[30, 30], [70, 70], [30, 70]], dtype=np.float32)
# M = cv.getAffineTransform(src, dst)
# M_ = _getAffineTransform(src, dst)
# print(np.all(np.abs(M - M_) < 1e-6))


def _cvtColor(src, code):
    """cv.cvtColor()
    参考: https://docs.opencv.org/4.5.2/de/d25/imgproc_color_conversions.html

    :param src: ndarray[H, W, C]. uint8
    :param code:
    :return: ndarray[H, W, C]. uint8
    """
    if code in (cv.COLOR_BGR2GRAY, cv.COLOR_RGB2GRAY):
        weight = np.array([0.299, 0.587, 0.114]) if code == cv.COLOR_RGB2GRAY else np.array([0.114, 0.587, 0.299])
        # [1, 3] @ [H, W, 3, 1] -> [H, W, 1, 1]
        return (weight[None] @ src[..., None])[:, :, 0, 0].round().astype(np.uint8)
    elif code in (cv.COLOR_GRAY2BGR, cv.COLOR_GRAY2RGB):
        src = src[..., None] if src.ndim == 2 else src
        return np.repeat(src, 3, -1)
    elif code in (cv.COLOR_RGB2BGR, cv.COLOR_BGR2RGB):
        return src[:, :, ::-1]
    elif code in (cv.COLOR_BGR2HSV, cv.COLOR_RGB2HSV):
        # 参考: https://docs.opencv.org/4.5.2/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
        src = src[:, :, ::-1] if code == cv.COLOR_RGB2HSV else src
        # (hue saturation value). 色调、饱和度、明度
        src = src.astype(np.float32) / 255  # to float
        val = np.max(src, -1)
        max_sub_min = val - np.min(src, -1)
        sat = np.where(val == 0, np.zeros_like(val), max_sub_min / val)
        blue, green, red = np.transpose(src, (2, 0, 1))
        hue = np.where(
            max_sub_min == 0, np.zeros_like(val),
            np.where(
                val == red, 60 * (green - blue) / max_sub_min, np.where(
                    val == green, 120 + 60 * (blue - red) / max_sub_min,
                    240 + 60 * (red - green) / max_sub_min
                )))
        # H: [-60, 300). S, V: [0, 1]
        hue = ((hue + 360) % 360 / 2).round().astype(np.uint8)  # `/ 2` for uint8
        sat = (sat * 255).round().astype(np.uint8)
        val = (val * 255).round().astype(np.uint8)
        # H: [0, 180]. S, V: [0, 255]
        return np.stack([hue, sat, val], -1)
    elif code in (cv.COLOR_HSV2BGR, cv.COLOR_HSV2RGB):
        # 参考: https://www.cnblogs.com/klchang/p/6784856.html
        hue, sat, val = np.transpose(src, (2, 0, 1)).astype(np.float32)
        # [0, 360], [0, 1], [0, 1]
        hue, sat, val = hue * 2, sat / 255, val / 255
        max_sub_min = sat * val
        min_ = val - max_sub_min  # max: val
        min2_sub = np.abs(((hue + 60) % 120 - 60) / 60 * max_sub_min)  # e.g. green - blue
        dst = np.where(
            ((300 <= hue) & (hue < 360))[..., None],
            np.stack([min2_sub, np.zeros_like(max_sub_min), max_sub_min], -1),
            np.where(
                ((0 <= hue) & (hue < 60))[..., None],
                np.stack([np.zeros_like(max_sub_min), min2_sub, max_sub_min], -1),
                np.where(
                    ((60 <= hue) & (hue < 120))[..., None],
                    np.stack([np.zeros_like(max_sub_min), max_sub_min, min2_sub], -1),
                    np.where(
                        ((120 <= hue) & (hue < 180))[..., None],
                        np.stack([min2_sub, max_sub_min, np.zeros_like(max_sub_min)], -1),
                        np.where(
                            ((180 <= hue) & (hue < 240))[..., None],
                            np.stack([max_sub_min, min2_sub, np.zeros_like(max_sub_min)], -1),
                            np.stack([max_sub_min, np.zeros_like(max_sub_min), min2_sub], -1))))))
        dst = ((dst + min_[..., None]) * 255).round().astype(np.uint8)
        dst = dst[:, :, ::-1] if code == cv.COLOR_HSV2RGB else dst
        return dst
    else:
        raise NotImplementedError


# ----------------------------------- to GRAY, 互换
# x = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
# y = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
# y_ = _cvtColor(x, cv.COLOR_BGR2GRAY)
# print(np.all(y == y_))  # True. 也可能出现False的情况
# y = cv.cvtColor(x, cv.COLOR_BGR2RGB)
# y_ = _cvtColor(x, cv.COLOR_BGR2RGB)
# print(np.all(y == y_))  # True

# ----------------------------------- GRAY to
# x = np.random.randint(0, 256, (10, 10, 1), dtype=np.uint8)
# y = cv.cvtColor(x, cv.COLOR_GRAY2BGR)
# y_ = _cvtColor(x, cv.COLOR_GRAY2BGR)
# print(np.all(y == y_))  # True

# ----------------------------------- HSV
# x = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
# y = cv.cvtColor(x, cv.COLOR_RGB2HSV)
# y_ = _cvtColor(x, cv.COLOR_RGB2HSV)
# print(np.all(y == y_))  # True. 也可能出现False的情况
# x1 = cv.cvtColor(y, cv.COLOR_HSV2BGR)
# x2 = _cvtColor(y_, cv.COLOR_HSV2BGR)
# print(np.all(x1 == x2))  # True. 也可能出现False的情况


def _LUT(src, lut):
    """cv.LUT(). 颜色0 - 255通过lut映射. src -> lut[src]
    参考: https://docs.opencv.org/4.5.2/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f

    :param src: shape[H, W]. uint8
    :param lut: shape[256]. uint8等int
    :return: shape[H, W]
    """
    return lut[src].astype(np.uint8)

# x = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
# lut = np.random.randint(0, 256, (256,), dtype=np.uint8)
# print(np.all(cv.LUT(x, lut) == _LUT(x, lut)))  # True
