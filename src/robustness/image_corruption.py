"""
CIR 图像损坏模块
================
实现像素级、区域级、全局级图像损坏

损坏分类体系:
├── Pixel-Level (像素级) - 破坏高频纹理信息
│   ├── gaussian_noise, shot_noise, impulse_noise, speckle_noise, poisson_noise
├── Region-Level (区域级) - 破坏空间相关性/局部结构
│   ├── Blur: gaussian_blur, defocus_blur, motion_blur, zoom_blur, glass_blur
│   └── Digital: pixelate, jpeg_compression, elastic_transform
└── Global-Level (全局级) - 改变整体统计分布
    ├── Weather: snow, frost, fog, rain, spatter
    ├── Color: brightness, contrast, saturation, hue_shift, channel_shuffle
    └── Geometric: scale, rotate, shear, perspective

Usage:
    corruptor = ImageCorruptor(seed=42)
    corrupted = corruptor.apply(image, 'gaussian_noise', severity=3)
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from enum import Enum
from typing import Union, Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
import random
import io
import warnings

# 尝试导入可选依赖
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV (cv2) not found. Some corruptions will use fallback implementations.")

try:
    from scipy.ndimage import zoom as scipy_zoom
    from scipy.ndimage import map_coordinates
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not found. Some corruptions will use fallback implementations.")

try:
    from skimage.filters import gaussian
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class CorruptionLevel(Enum):
    """损坏层级枚举"""
    PIXEL = "pixel"       # 像素级：噪声
    REGION = "region"     # 区域级：模糊、数字伪影
    GLOBAL = "global"     # 全局级：天气、光学、几何


class CorruptionCategory(Enum):
    """损坏类别枚举"""
    NOISE = "noise"           # 噪声
    BLUR = "blur"             # 模糊
    DIGITAL = "digital"       # 数字处理
    WEATHER = "weather"       # 天气
    COLOR = "color"           # 色彩/光学
    GEOMETRIC = "geometric"   # 几何变换


@dataclass
class CorruptionInfo:
    """损坏信息"""
    name: str
    level: CorruptionLevel
    category: CorruptionCategory
    description: str


class ImageCorruptor:
    """
    CIR图像损坏器
    
    支持24种损坏类型，每种支持5级严重度(1-5)
    
    Attributes:
        seed: 随机种子，用于可复现性
        
    Example:
        >>> corruptor = ImageCorruptor(seed=42)
        >>> # 应用单一损坏
        >>> corrupted = corruptor.apply(image, 'gaussian_noise', severity=3)
        >>> # 随机应用损坏
        >>> corrupted, name, severity = corruptor.apply_random(image, level='pixel')
        >>> # 应用复合损坏
        >>> corrupted = corruptor.apply_composite(image, [('gaussian_noise', 2), ('motion_blur', 3)])
    """
    
    # ============== 损坏类型注册表 ==============
    CORRUPTION_REGISTRY: Dict[str, CorruptionInfo] = {
        # 像素级 - 噪声
        'gaussian_noise': CorruptionInfo('gaussian_noise', CorruptionLevel.PIXEL, CorruptionCategory.NOISE, '高斯噪声 - 电子传感器热噪声'),
        'shot_noise': CorruptionInfo('shot_noise', CorruptionLevel.PIXEL, CorruptionCategory.NOISE, '散粒噪声 - 光子泊松分布'),
        'impulse_noise': CorruptionInfo('impulse_noise', CorruptionLevel.PIXEL, CorruptionCategory.NOISE, '脉冲噪声 - 椒盐/传输错误'),
        'speckle_noise': CorruptionInfo('speckle_noise', CorruptionLevel.PIXEL, CorruptionCategory.NOISE, '散斑噪声 - 乘性噪声'),
        'poisson_noise': CorruptionInfo('poisson_noise', CorruptionLevel.PIXEL, CorruptionCategory.NOISE, '泊松噪声 - 传感器量子噪声'),
        
        # 区域级 - 模糊
        'gaussian_blur': CorruptionInfo('gaussian_blur', CorruptionLevel.REGION, CorruptionCategory.BLUR, '高斯模糊 - 通用模糊'),
        'defocus_blur': CorruptionInfo('defocus_blur', CorruptionLevel.REGION, CorruptionCategory.BLUR, '散焦模糊 - 对焦失败'),
        'motion_blur': CorruptionInfo('motion_blur', CorruptionLevel.REGION, CorruptionCategory.BLUR, '运动模糊 - 相机/物体移动'),
        'zoom_blur': CorruptionInfo('zoom_blur', CorruptionLevel.REGION, CorruptionCategory.BLUR, '变焦模糊 - 径向模糊'),
        'glass_blur': CorruptionInfo('glass_blur', CorruptionLevel.REGION, CorruptionCategory.BLUR, '玻璃模糊 - 毛玻璃效果'),
        
        # 区域级 - 数字处理
        'pixelate': CorruptionInfo('pixelate', CorruptionLevel.REGION, CorruptionCategory.DIGITAL, '像素化 - 分辨率降低'),
        'jpeg_compression': CorruptionInfo('jpeg_compression', CorruptionLevel.REGION, CorruptionCategory.DIGITAL, 'JPEG压缩 - 块效应'),
        'elastic_transform': CorruptionInfo('elastic_transform', CorruptionLevel.REGION, CorruptionCategory.DIGITAL, '弹性变换 - 局部扭曲'),
        
        # 全局级 - 天气
        'snow': CorruptionInfo('snow', CorruptionLevel.GLOBAL, CorruptionCategory.WEATHER, '雪 - 白色遮挡'),
        'frost': CorruptionInfo('frost', CorruptionLevel.GLOBAL, CorruptionCategory.WEATHER, '霜 - 边缘模糊+白色'),
        'fog': CorruptionInfo('fog', CorruptionLevel.GLOBAL, CorruptionCategory.WEATHER, '雾 - 对比度降低'),
        'rain': CorruptionInfo('rain', CorruptionLevel.GLOBAL, CorruptionCategory.WEATHER, '雨 - 条状遮挡'),
        'spatter': CorruptionInfo('spatter', CorruptionLevel.GLOBAL, CorruptionCategory.WEATHER, '溅射 - 点状遮挡'),
        
        # 全局级 - 色彩/光学
        'brightness': CorruptionInfo('brightness', CorruptionLevel.GLOBAL, CorruptionCategory.COLOR, '亮度 - 过曝/欠曝'),
        'contrast': CorruptionInfo('contrast', CorruptionLevel.GLOBAL, CorruptionCategory.COLOR, '对比度 - 灰度压缩'),
        'saturation': CorruptionInfo('saturation', CorruptionLevel.GLOBAL, CorruptionCategory.COLOR, '饱和度 - 颜色鲜艳度'),
        'hue_shift': CorruptionInfo('hue_shift', CorruptionLevel.GLOBAL, CorruptionCategory.COLOR, '色调偏移 - 色环旋转'),
        'channel_shuffle': CorruptionInfo('channel_shuffle', CorruptionLevel.GLOBAL, CorruptionCategory.COLOR, '通道重排 - RGB互换'),
        
        # 全局级 - 几何
        'scale': CorruptionInfo('scale', CorruptionLevel.GLOBAL, CorruptionCategory.GEOMETRIC, '缩放'),
        'rotate': CorruptionInfo('rotate', CorruptionLevel.GLOBAL, CorruptionCategory.GEOMETRIC, '旋转'),
        'shear': CorruptionInfo('shear', CorruptionLevel.GLOBAL, CorruptionCategory.GEOMETRIC, '剪切'),
        'perspective': CorruptionInfo('perspective', CorruptionLevel.GLOBAL, CorruptionCategory.GEOMETRIC, '透视变换'),
    }
    
    def __init__(self, seed: int = 42):
        """
        初始化图像损坏器
        
        Args:
            seed: 随机种子，用于可复现性
        """
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        random.seed(seed)
        
    def set_seed(self, seed: int):
        """设置随机种子"""
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def _to_numpy(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """将图像转换为 numpy 数组 (uint8, 0-255)"""
        if isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _to_pil(self, image: np.ndarray) -> Image.Image:
        """将 numpy 数组转换为 PIL.Image"""
        return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    
    def _clip(self, image: np.ndarray) -> np.ndarray:
        """裁剪到有效范围 [0, 255]"""
        return np.clip(image, 0, 255).astype(np.uint8)
    
    # ============== 像素级损坏 (Pixel-Level) ==============
    
    def gaussian_noise(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        高斯噪声 - 模拟低光照下的电子传感器热噪声
        
        Args:
            image: 输入图像 (numpy array, uint8)
            severity: 严重程度 1-5
        """
        # 严重度对应的标准差 (相对于255)
        sigma_map = {1: 10, 2: 20, 3: 35, 4: 55, 5: 80}
        sigma = sigma_map[severity]
        
        noise = self._rng.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + noise
        return self._clip(noisy)
    
    def shot_noise(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        散粒噪声 - 模拟光子计数的泊松分布噪声
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的缩放因子 (越小噪声越大)
        scale_map = {1: 60, 2: 25, 3: 12, 4: 5, 5: 3}
        scale = scale_map[severity]
        
        # 泊松噪声
        noisy = self._rng.poisson(image.astype(np.float32) / 255.0 * scale) / scale * 255
        return self._clip(noisy)
    
    def impulse_noise(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        脉冲噪声(椒盐噪声) - 模拟传输错误
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的噪声比例
        prob_map = {1: 0.03, 2: 0.06, 3: 0.10, 4: 0.15, 5: 0.22}
        prob = prob_map[severity]
        
        noisy = image.copy()
        # 生成随机掩码
        salt_mask = self._rng.random(image.shape[:2]) < prob / 2
        pepper_mask = self._rng.random(image.shape[:2]) < prob / 2
        
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
        return noisy
    
    def speckle_noise(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        散斑噪声 - 乘性噪声
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的噪声强度
        sigma_map = {1: 0.15, 2: 0.25, 3: 0.40, 4: 0.55, 5: 0.70}
        sigma = sigma_map[severity]
        
        noise = self._rng.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) * (1 + noise)
        return self._clip(noisy)
    
    def poisson_noise(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        泊松噪声 - 传感器固有的量子噪声
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的峰值 (越低噪声越大)
        peak_map = {1: 200, 2: 100, 3: 50, 4: 25, 5: 10}
        peak = peak_map[severity]
        
        # 归一化后应用泊松噪声
        noisy = self._rng.poisson(image.astype(np.float32) / 255.0 * peak) / peak * 255
        return self._clip(noisy)
    
    # ============== 区域级损坏 - 模糊 (Region-Level Blur) ==============
    
    def gaussian_blur(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        高斯模糊 - 通用模糊
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的模糊半径
        radius_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}
        radius = radius_map[severity]
        
        pil_img = self._to_pil(image)
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(blurred)
    
    def defocus_blur(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        散焦模糊 - 模拟相机对焦失败
        使用圆盘形状的模糊核
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的模糊半径
        radius_map = {1: 3, 2: 5, 3: 7, 4: 9, 5: 12}
        radius = radius_map[severity]
        
        if HAS_CV2:
            # 创建圆盘模糊核
            kernel_size = 2 * radius + 1
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            cv2.circle(kernel, (radius, radius), radius, 1, -1)
            kernel = kernel / kernel.sum()
            
            blurred = cv2.filter2D(image, -1, kernel)
            return blurred
        else:
            # 回退到高斯模糊
            return self.gaussian_blur(image, severity)
    
    def motion_blur(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        运动模糊 - 模拟拍摄对象或相机的快速移动
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的核大小
        kernel_size_map = {1: 7, 2: 11, 3: 15, 4: 21, 5: 27}
        kernel_size = kernel_size_map[severity]
        
        # 随机角度
        angle = self._rng.uniform(-45, 45)
        
        if HAS_CV2:
            # 创建运动模糊核
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1
            kernel = kernel / kernel_size
            
            # 旋转核
            center = (kernel_size // 2, kernel_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / (kernel.sum() + 1e-8)
            
            blurred = cv2.filter2D(image, -1, kernel)
            return blurred
        else:
            # 回退实现
            pil_img = self._to_pil(image)
            blurred = pil_img.filter(ImageFilter.BoxBlur(radius=kernel_size // 3))
            return np.array(blurred)
    
    def zoom_blur(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        变焦模糊 - 模拟变焦过程中的径向模糊
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的缩放因子列表
        zoom_factors_map = {
            1: np.arange(1, 1.06, 0.01),
            2: np.arange(1, 1.10, 0.01),
            3: np.arange(1, 1.15, 0.02),
            4: np.arange(1, 1.20, 0.02),
            5: np.arange(1, 1.30, 0.02),
        }
        zoom_factors = zoom_factors_map[severity]
        
        h, w = image.shape[:2]
        out = np.zeros_like(image, dtype=np.float32)
        
        for zoom_factor in zoom_factors:
            # 缩放
            zh, zw = int(h * zoom_factor), int(w * zoom_factor)
            
            if HAS_CV2:
                zoomed = cv2.resize(image, (zw, zh), interpolation=cv2.INTER_LINEAR)
            else:
                pil_img = self._to_pil(image)
                zoomed = np.array(pil_img.resize((zw, zh), Image.BILINEAR))
            
            # 中心裁剪
            top = (zh - h) // 2
            left = (zw - w) // 2
            cropped = zoomed[top:top+h, left:left+w]
            
            out += cropped.astype(np.float32)
        
        out = out / len(zoom_factors)
        return self._clip(out)
    
    def glass_blur(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        玻璃模糊 - 模拟透过毛玻璃观看的效果
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的参数
        params_map = {
            1: (0.7, 1, 1),   # (sigma, max_delta, iterations)
            2: (0.9, 1, 2),
            3: (1.0, 2, 2),
            4: (1.1, 2, 3),
            5: (1.5, 3, 3),
        }
        sigma, max_delta, iterations = params_map[severity]
        
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        for _ in range(iterations):
            # 先应用高斯模糊
            pil_img = self._to_pil(self._clip(result))
            blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=sigma)))
            
            # 局部像素交换
            dx = self._rng.randint(-max_delta, max_delta + 1, (h, w))
            dy = self._rng.randint(-max_delta, max_delta + 1, (h, w))
            
            # 创建采样坐标
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = np.clip(x + dx, 0, w - 1)
            y_new = np.clip(y + dy, 0, h - 1)
            
            # 采样
            result = blurred[y_new, x_new]
        
        return self._clip(result)
    
    # ============== 区域级损坏 - 数字处理 (Region-Level Digital) ==============
    
    def pixelate(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        像素化 - 降低分辨率，丢失细节
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的缩放因子 (越小像素化越严重)
        factor_map = {1: 0.7, 2: 0.5, 3: 0.35, 4: 0.2, 5: 0.1}
        factor = factor_map[severity]
        
        h, w = image.shape[:2]
        small_h, small_w = int(h * factor), int(w * factor)
        
        pil_img = self._to_pil(image)
        # 缩小再放大
        small = pil_img.resize((small_w, small_h), Image.NEAREST)
        pixelated = small.resize((w, h), Image.NEAREST)
        
        return np.array(pixelated)
    
    def jpeg_compression(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        JPEG压缩 - 产生块效应，破坏边缘
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的JPEG质量 (越低压缩越严重)
        quality_map = {1: 80, 2: 60, 3: 40, 4: 25, 5: 10}
        quality = quality_map[severity]
        
        pil_img = self._to_pil(image)
        
        # 保存到内存中的JPEG，然后重新读取
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        
        return np.array(compressed.convert('RGB'))
    
    def elastic_transform(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        弹性变换 - 局部扭曲，模拟布料褶皱或介质变形
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的参数
        params_map = {
            1: (100, 6),   # (alpha, sigma)
            2: (200, 8),
            3: (400, 10),
            4: (600, 12),
            5: (800, 14),
        }
        alpha, sigma = params_map[severity]
        
        h, w = image.shape[:2]
        
        if HAS_SCIPY:
            # 生成随机位移场
            dx = self._rng.uniform(-1, 1, (h, w)) * alpha
            dy = self._rng.uniform(-1, 1, (h, w)) * alpha
            
            # 高斯平滑
            from scipy.ndimage import gaussian_filter
            dx = gaussian_filter(dx, sigma)
            dy = gaussian_filter(dy, sigma)
            
            # 创建采样网格
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = (x + dx).astype(np.float32)
            y_new = (y + dy).astype(np.float32)
            
            # 双线性插值
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = map_coordinates(
                    image[:, :, c], [y_new, x_new], order=1, mode='reflect'
                )
            return self._clip(result)
        else:
            # 简化版：使用轻微的高斯模糊作为回退
            return self.gaussian_blur(image, max(1, severity - 1))
    
    # ============== 全局级损坏 - 天气 (Global-Level Weather) ==============
    
    def snow(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        雪 - 白色遮挡
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的参数
        params_map = {
            1: (0.1, 0.2, 0.5),   # (snow_prob, snow_alpha, brightness_boost)
            2: (0.15, 0.3, 0.6),
            3: (0.2, 0.4, 0.7),
            4: (0.3, 0.5, 0.8),
            5: (0.4, 0.6, 0.9),
        }
        snow_prob, snow_alpha, brightness_boost = params_map[severity]
        
        h, w = image.shape[:2]
        result = image.astype(np.float32)
        
        # 雪花层
        snow_layer = self._rng.random((h, w)) < snow_prob
        snow_layer = snow_layer.astype(np.float32) * 255
        
        # 轻微模糊雪花
        pil_snow = self._to_pil(snow_layer[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8))
        snow_layer = np.array(pil_snow.filter(ImageFilter.GaussianBlur(radius=1)))[:, :, 0]
        
        # 混合
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - snow_alpha * snow_layer / 255) + snow_layer * snow_alpha
        
        # 增加整体亮度
        result = result * (1 + brightness_boost * 0.3)
        
        return self._clip(result)
    
    def frost(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        霜 - 边缘模糊+白色覆盖
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的参数
        params_map = {
            1: (0.3, 1.0),   # (frost_alpha, blur_radius)
            2: (0.4, 1.5),
            3: (0.5, 2.0),
            4: (0.6, 2.5),
            5: (0.7, 3.0),
        }
        frost_alpha, blur_radius = params_map[severity]
        
        h, w = image.shape[:2]
        
        # 创建霜层 (随机纹理)
        frost_layer = self._rng.random((h, w)) * 255
        pil_frost = self._to_pil(frost_layer[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8))
        frost_layer = np.array(pil_frost.filter(ImageFilter.GaussianBlur(radius=10)))[:, :, 0]
        
        # 归一化
        frost_layer = (frost_layer - frost_layer.min()) / (frost_layer.max() - frost_layer.min() + 1e-8)
        
        # 模糊原图
        pil_img = self._to_pil(image)
        blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        
        # 混合
        result = blurred.astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - frost_alpha * frost_layer) + 255 * frost_alpha * frost_layer
        
        return self._clip(result)
    
    def fog(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        雾 - 降低对比度
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的参数
        params_map = {
            1: (0.2, 200),   # (fog_density, fog_brightness)
            2: (0.35, 210),
            3: (0.5, 220),
            4: (0.65, 230),
            5: (0.8, 240),
        }
        fog_density, fog_brightness = params_map[severity]
        
        h, w = image.shape[:2]
        
        # 创建雾层 (深度相关)
        # 模拟远处雾更浓
        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        fog_layer = y_coords * fog_density + (1 - fog_density) * 0.5
        fog_layer = fog_layer + self._rng.random((h, w)) * 0.1  # 添加一些随机性
        
        # 混合
        result = image.astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - fog_layer) + fog_brightness * fog_layer
        
        return self._clip(result)
    
    def rain(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        雨 - 条状遮挡
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的参数
        params_map = {
            1: (100, 0.3),    # (num_drops, alpha)
            2: (200, 0.4),
            3: (350, 0.5),
            4: (500, 0.6),
            5: (700, 0.7),
        }
        num_drops, alpha = params_map[severity]
        
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # 生成雨滴
        rain_layer = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(num_drops):
            # 随机起点
            x = self._rng.randint(0, w)
            y = self._rng.randint(0, h)
            
            # 雨滴长度和角度
            length = self._rng.randint(10, 30)
            angle = self._rng.uniform(-20, 20)  # 近似垂直
            
            # 绘制雨滴
            dx = int(length * np.sin(np.radians(angle)))
            dy = int(length * np.cos(np.radians(angle)))
            
            x2, y2 = x + dx, y + dy
            x1, y1 = x, y
            
            # 简单的线绘制 (Bresenham算法简化版)
            steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
            xs = np.linspace(x1, x2, steps).astype(int)
            ys = np.linspace(y1, y2, steps).astype(int)
            
            # 裁剪到图像范围
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys = xs[valid], ys[valid]
            
            rain_layer[ys, xs] = 255
        
        # 模糊雨层
        pil_rain = self._to_pil(rain_layer[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8))
        rain_layer = np.array(pil_rain.filter(ImageFilter.GaussianBlur(radius=0.5)))[:, :, 0]
        
        # 混合
        rain_layer = rain_layer / 255.0
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - alpha * rain_layer) + 200 * alpha * rain_layer
        
        return self._clip(result)
    
    def spatter(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        溅射 - 点状遮挡
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的参数
        params_map = {
            1: (50, (3, 8), 0.4),    # (num_splatters, size_range, alpha)
            2: (100, (5, 12), 0.5),
            3: (180, (8, 18), 0.6),
            4: (280, (10, 25), 0.7),
            5: (400, (15, 35), 0.8),
        }
        num_splatters, size_range, alpha = params_map[severity]
        
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        # 生成溅射
        spatter_layer = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(num_splatters):
            # 随机位置和大小
            x = self._rng.randint(0, w)
            y = self._rng.randint(0, h)
            radius = self._rng.randint(*size_range)
            
            # 绘制圆形
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            spatter_layer[mask] = 255
        
        # 模糊溅射层
        pil_spatter = self._to_pil(spatter_layer[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8))
        spatter_layer = np.array(pil_spatter.filter(ImageFilter.GaussianBlur(radius=2)))[:, :, 0]
        
        # 随机颜色 (泥土色/水色/灰色)
        colors = [
            [139, 90, 43],    # 泥土色
            [100, 149, 237],  # 水色
            [128, 128, 128],  # 灰色
        ]
        color_idx = self._rng.randint(0, len(colors))
        color = colors[color_idx]
        
        # 混合
        spatter_layer = spatter_layer / 255.0
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - alpha * spatter_layer) + color[c] * alpha * spatter_layer
        
        return self._clip(result)
    
    # ============== 全局级损坏 - 色彩/光学 (Global-Level Color) ==============
    
    def brightness(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        亮度 - 过曝或欠曝
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的亮度因子范围
        factor_map = {
            1: (0.8, 1.2),
            2: (0.6, 1.4),
            3: (0.4, 1.6),
            4: (0.25, 1.8),
            5: (0.1, 2.0),
        }
        low, high = factor_map[severity]
        
        # 随机选择增亮或减暗
        factor = self._rng.choice([low, high])
        
        pil_img = self._to_pil(image)
        enhancer = ImageEnhance.Brightness(pil_img)
        result = enhancer.enhance(factor)
        
        return np.array(result)
    
    def contrast(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        对比度 - 灰度范围压缩/扩展
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的对比度因子范围
        factor_map = {
            1: (0.8, 1.3),
            2: (0.6, 1.5),
            3: (0.4, 1.8),
            4: (0.25, 2.0),
            5: (0.1, 2.5),
        }
        low, high = factor_map[severity]
        
        # 随机选择低或高对比度
        factor = self._rng.choice([low, high])
        
        pil_img = self._to_pil(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        result = enhancer.enhance(factor)
        
        return np.array(result)
    
    def saturation(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        饱和度 - 颜色鲜艳度变化
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的饱和度因子范围
        factor_map = {
            1: (0.7, 1.4),
            2: (0.5, 1.7),
            3: (0.3, 2.0),
            4: (0.15, 2.5),
            5: (0.0, 3.0),
        }
        low, high = factor_map[severity]
        
        # 随机选择降低或提高饱和度
        factor = self._rng.choice([low, high])
        
        pil_img = self._to_pil(image)
        enhancer = ImageEnhance.Color(pil_img)
        result = enhancer.enhance(factor)
        
        return np.array(result)
    
    def hue_shift(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        色调偏移 - 色环旋转
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的色调偏移角度
        shift_map = {1: 20, 2: 40, 3: 70, 4: 110, 5: 150}
        max_shift = shift_map[severity]
        shift = self._rng.randint(-max_shift, max_shift)
        
        pil_img = self._to_pil(image)
        
        # 转换到 HSV
        hsv = pil_img.convert('HSV')
        h, s, v = hsv.split()
        
        # 偏移色调
        h_array = np.array(h, dtype=np.int16)
        h_array = (h_array + shift) % 256
        h_shifted = Image.fromarray(h_array.astype(np.uint8), mode='L')
        
        # 合并回来
        shifted = Image.merge('HSV', (h_shifted, s, v))
        result = shifted.convert('RGB')
        
        return np.array(result)
    
    def channel_shuffle(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        通道重排 - RGB通道互换
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5 (这里severity主要影响随机性)
        """
        # 可能的通道排列
        permutations = [
            (0, 2, 1),  # RBG
            (1, 0, 2),  # GRB
            (1, 2, 0),  # GBR
            (2, 0, 1),  # BRG
            (2, 1, 0),  # BGR
        ]
        
        # 根据严重度选择排列
        idx = min(severity - 1, len(permutations) - 1)
        perm = permutations[idx]
        
        result = image[:, :, perm]
        return result
    
    # ============== 全局级损坏 - 几何 (Global-Level Geometric) ==============
    
    def scale(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        缩放 - 随机缩放
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的缩放范围
        scale_map = {
            1: (0.9, 1.1),
            2: (0.8, 1.2),
            3: (0.7, 1.4),
            4: (0.6, 1.5),
            5: (0.5, 1.7),
        }
        low, high = scale_map[severity]
        factor = self._rng.uniform(low, high)
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        
        pil_img = self._to_pil(image)
        scaled = pil_img.resize((new_w, new_h), Image.BILINEAR)
        
        # 中心裁剪或填充到原始大小
        result = Image.new('RGB', (w, h), (128, 128, 128))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        
        if factor < 1:
            # 缩小：居中粘贴
            result.paste(scaled, (paste_x, paste_y))
        else:
            # 放大：中心裁剪
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            result = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        
        return np.array(result)
    
    def rotate(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        旋转
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的旋转角度范围
        angle_map = {1: 5, 2: 10, 3: 20, 4: 35, 5: 50}
        max_angle = angle_map[severity]
        angle = self._rng.uniform(-max_angle, max_angle)
        
        pil_img = self._to_pil(image)
        rotated = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        
        return np.array(rotated)
    
    def shear(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        剪切变换
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的剪切因子
        shear_map = {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.25, 5: 0.35}
        max_shear = shear_map[severity]
        shear_x = self._rng.uniform(-max_shear, max_shear)
        shear_y = self._rng.uniform(-max_shear, max_shear)
        
        h, w = image.shape[:2]
        pil_img = self._to_pil(image)
        
        # 仿射变换矩阵
        # [1, shear_x, 0]
        # [shear_y, 1, 0]
        coeffs = (1, shear_x, -shear_x * h / 2, shear_y, 1, -shear_y * w / 2)
        sheared = pil_img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        
        return np.array(sheared)
    
    def perspective(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        透视变换
        
        Args:
            image: 输入图像
            severity: 严重程度 1-5
        """
        # 严重度对应的透视强度
        strength_map = {1: 0.02, 2: 0.05, 3: 0.08, 4: 0.12, 5: 0.18}
        strength = strength_map[severity]
        
        h, w = image.shape[:2]
        
        # 原始四个角点
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # 随机扰动角点
        dst_pts = src_pts + self._rng.uniform(-strength * min(h, w), strength * min(h, w), src_pts.shape).astype(np.float32)
        
        if HAS_CV2:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (w, h), borderValue=(128, 128, 128))
            return warped
        else:
            # 回退：使用轻微旋转
            return self.rotate(image, max(1, severity - 1))
    
    # ============== 统一接口 ==============
    
    def apply(
        self, 
        image: Union[np.ndarray, Image.Image], 
        corruption_type: str, 
        severity: int = 3
    ) -> np.ndarray:
        """
        应用指定类型的损坏
        
        Args:
            image: 输入图像 (numpy array 或 PIL.Image)
            corruption_type: 损坏类型名称
            severity: 严重程度 1-5
            
        Returns:
            损坏后的图像 (numpy array, uint8)
        """
        if corruption_type not in self.CORRUPTION_REGISTRY:
            raise ValueError(f"Unknown corruption type: {corruption_type}. "
                           f"Available: {list(self.CORRUPTION_REGISTRY.keys())}")
        
        if not 1 <= severity <= 5:
            raise ValueError(f"Severity must be in [1, 5], got {severity}")
        
        # 转换为 numpy
        img_array = self._to_numpy(image)
        
        # 获取并调用对应的损坏函数
        func = getattr(self, corruption_type)
        return func(img_array, severity)
    
    def apply_pil(
        self,
        image: Union[np.ndarray, Image.Image],
        corruption_type: str,
        severity: int = 3
    ) -> Image.Image:
        """
        应用损坏并返回 PIL.Image
        
        Args:
            image: 输入图像
            corruption_type: 损坏类型
            severity: 严重程度
            
        Returns:
            损坏后的 PIL.Image
        """
        corrupted = self.apply(image, corruption_type, severity)
        return self._to_pil(corrupted)
    
    def apply_random(
        self,
        image: Union[np.ndarray, Image.Image],
        level: Optional[str] = None,
        category: Optional[str] = None,
        severity: Optional[int] = None
    ) -> Tuple[np.ndarray, str, int]:
        """
        随机应用一种损坏
        
        Args:
            image: 输入图像
            level: 限制损坏层级 ('pixel', 'region', 'global')
            category: 限制损坏类别 ('noise', 'blur', 'digital', 'weather', 'color', 'geometric')
            severity: 指定严重度，None则随机
            
        Returns:
            (corrupted_image, corruption_type, severity)
        """
        # 筛选可用的损坏类型
        available = list(self.CORRUPTION_REGISTRY.keys())
        
        if level:
            level_enum = CorruptionLevel(level)
            available = [k for k, v in self.CORRUPTION_REGISTRY.items() if v.level == level_enum]
        
        if category:
            cat_enum = CorruptionCategory(category)
            available = [k for k in available if self.CORRUPTION_REGISTRY[k].category == cat_enum]
        
        if not available:
            raise ValueError(f"No corruption types match the given constraints: level={level}, category={category}")
        
        # 随机选择
        corruption_type = self._rng.choice(available)
        if severity is None:
            severity = self._rng.randint(1, 6)
        
        corrupted = self.apply(image, corruption_type, severity)
        return corrupted, corruption_type, severity
    
    def apply_composite(
        self,
        image: Union[np.ndarray, Image.Image],
        corruptions: List[Tuple[str, int]]
    ) -> np.ndarray:
        """
        应用复合损坏 (多种损坏叠加)
        
        Args:
            image: 输入图像
            corruptions: 损坏列表 [(type, severity), ...]
            
        Returns:
            损坏后的图像
        """
        result = self._to_numpy(image)
        for corruption_type, severity in corruptions:
            result = self.apply(result, corruption_type, severity)
        return result
    
    @classmethod
    def get_all_types(cls) -> Dict[str, List[str]]:
        """
        获取所有损坏类型，按层级分组
        
        Returns:
            {level: [corruption_types]}
        """
        result = {}
        for name, info in cls.CORRUPTION_REGISTRY.items():
            level = info.level.value
            if level not in result:
                result[level] = []
            result[level].append(name)
        return result
    
    @classmethod
    def get_types_by_category(cls) -> Dict[str, List[str]]:
        """
        获取所有损坏类型，按类别分组
        
        Returns:
            {category: [corruption_types]}
        """
        result = {}
        for name, info in cls.CORRUPTION_REGISTRY.items():
            cat = info.category.value
            if cat not in result:
                result[cat] = []
            result[cat].append(name)
        return result
    
    @classmethod
    def get_corruption_info(cls, corruption_type: str) -> CorruptionInfo:
        """获取损坏类型的详细信息"""
        if corruption_type not in cls.CORRUPTION_REGISTRY:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        return cls.CORRUPTION_REGISTRY[corruption_type]


# ============== 便捷函数 ==============

def corrupt_image(
    image: Union[np.ndarray, Image.Image],
    corruption_type: str,
    severity: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    便捷函数：应用单一损坏
    
    Args:
        image: 输入图像
        corruption_type: 损坏类型
        severity: 严重程度 1-5
        seed: 随机种子
        
    Returns:
        损坏后的图像
    """
    corruptor = ImageCorruptor(seed=seed)
    return corruptor.apply(image, corruption_type, severity)


def list_corruptions() -> None:
    """打印所有可用的损坏类型"""
    print("=" * 60)
    print("Available Image Corruptions")
    print("=" * 60)
    
    types_by_level = ImageCorruptor.get_all_types()
    for level in ['pixel', 'region', 'global']:
        print(f"\n{level.upper()} Level:")
        for name in types_by_level.get(level, []):
            info = ImageCorruptor.CORRUPTION_REGISTRY[name]
            print(f"  - {name}: {info.description}")


if __name__ == '__main__':
    list_corruptions()

