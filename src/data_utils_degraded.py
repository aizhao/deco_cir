"""
支持视觉降质的CIRR数据集类

用于在降质数据上训练和评估SPRC模型
"""

import json
from pathlib import Path
from typing import List, Optional
import numpy as np

import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import ImageFilter

base_path = Path(__file__).absolute().parents[1].absolute()


# ==================== 图像降质工具 ====================
class ImageDegradation:
    """图像降质变换工具类"""
    
    @staticmethod
    def apply_cutout(image, boxes: List[dict]):
        """应用Cutout遮挡"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for box in boxes:
            x1 = int(box['x'] * w)
            y1 = int(box['y'] * h)
            x2 = int((box['x'] + box['w']) * w)
            y2 = int((box['y'] + box['h']) * h)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            img_array[y1:y2, x1:x2] = 0
        
        return PIL.Image.fromarray(img_array)
    
    @staticmethod
    def apply_gaussian_blur(image, kernel_size: int, sigma: float):
        """应用高斯模糊"""
        radius = max(1, kernel_size // 2)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def apply_motion_blur(image, kernel_size: int, angle: float):
        """应用运动模糊（简化版本使用BoxBlur）"""
        try:
            import cv2
            img_array = np.array(image)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1
            kernel = kernel / kernel_size
            center = (kernel_size // 2, kernel_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / (kernel.sum() + 1e-8)
            blurred = cv2.filter2D(img_array, -1, kernel)
            return PIL.Image.fromarray(blurred)
        except ImportError:
            radius = max(1, kernel_size // 2)
            return image.filter(ImageFilter.BoxBlur(radius=radius))
    
    @classmethod
    def apply_degradation(cls, image, params: dict):
        """根据参数应用降质"""
        if params is None:
            return image
        
        deg_type = params.get('type', 'none')
        
        if deg_type == 'cutout':
            return cls.apply_cutout(image, params.get('boxes', []))
        elif deg_type in ('gaussian_blur', 'gaussian'):
            return cls.apply_gaussian_blur(
                image,
                params.get('kernel_size', 5),
                params.get('sigma', 1.0)
            )
        elif deg_type in ('motion_blur', 'motion'):
            return cls.apply_motion_blur(
                image,
                params.get('kernel_size', 9),
                params.get('angle', 0)
            )
        return image
    
    @classmethod
    def apply_multi_degradation(cls, image, params_list: List[dict]):
        """应用多个降质"""
        if not params_list:
            return image
        result = image
        for params in params_list:
            result = cls.apply_degradation(result, params)
        return result


# ==================== 预处理工具 ====================
def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TargetPad:
    def __init__(self, target_ratio: float, size: int):
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(w, h) / self.target_ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def targetpad_transform(target_ratio: float, dim: int):
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


# ==================== 降质数据集类 ====================
class CIRRDegradedDataset(Dataset):
    """
    支持视觉降质的CIRR数据集
    
    Args:
        split: 'train', 'val', 'test1'
        mode: 'relative' 或 'classic'
        preprocess: 预处理函数
        config_dir: 降质配置目录路径
        apply_degradation: 是否应用降质（训练时设为True，gallery提取时设为False）
    """
    
    def __init__(
        self,
        split: str,
        mode: str,
        preprocess: callable,
        config_dir: Optional[str] = None,
        apply_degradation: bool = True,
    ):
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.config_dir = Path(config_dir) if config_dir else None
        self.apply_degradation = apply_degradation
        
        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        
        # 加载配置
        if self.config_dir and (self.config_dir / f'cap.rc2.{split}.json').exists():
            caption_path = self.config_dir / f'cap.rc2.{split}.json'
            split_path = self.config_dir / f'split.rc2.{split}.json'
            print(f"Loading degraded config from: {self.config_dir}")
        else:
            caption_path = base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json'
            split_path = base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
            print(f"Loading original CIRR data")
        
        with open(caption_path) as f:
            self.triplets = json.load(f)
        with open(split_path) as f:
            self.name_to_relpath = json.load(f)
        
        self._print_stats()
        print(f"CIRRDegraded {split} dataset in {mode} mode initialized")
    
    def _print_stats(self):
        """打印降质统计信息"""
        if not self.config_dir:
            return
        
        stats = {'clean': 0, 'cutout': 0, 'gaussian_blur': 0, 'motion_blur': 0, 'noisy_label': 0, 'mixed': 0}
        for t in self.triplets:
            deg = t.get('reference_degradation', 'none')
            if t.get('is_label_noisy', False):
                stats['noisy_label'] += 1
            
            if deg == 'none' or deg == 'clean':
                stats['clean'] += 1
            elif 'cutout' in deg and '+' not in deg:
                stats['cutout'] += 1
            elif 'gaussian' in deg and '+' not in deg:
                stats['gaussian_blur'] += 1
            elif 'motion' in deg and '+' not in deg:
                stats['motion_blur'] += 1
            elif '+' in deg:
                stats['mixed'] += 1
        
        print(f"  Degradation stats: {stats}")
    
    def _load_image(self, image_name: str, degradation_params=None):
        """加载图像并可选地应用降质"""
        image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
        image = PIL.Image.open(image_path).convert('RGB')
        
        # 应用降质（在预处理之前）
        if self.apply_degradation and degradation_params:
            if isinstance(degradation_params, list):
                image = ImageDegradation.apply_multi_degradation(image, degradation_params)
            else:
                image = ImageDegradation.apply_degradation(image, degradation_params)
        
        return self.preprocess(image)
    
    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
    
    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                triplet = self.triplets[index]
                group_members = triplet['img_set']['members']
                reference_name = triplet['reference']
                rel_caption = triplet['caption']
                
                # 获取降质参数（仅对参考图像）
                degradation_params = triplet.get('degradation_params', None)
                
                if self.split == 'train':
                    # 训练时返回图像张量
                    reference_image = self._load_image(reference_name, degradation_params)
                    target_hard_name = triplet['target_hard']
                    target_image = self._load_image(target_hard_name, None)  # 目标图不降质
                    return reference_image, target_image, rel_caption
                
                elif self.split == 'val':
                    target_hard_name = triplet['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members
                
                elif self.split == 'test1':
                    pair_id = triplet['pairid']
                    return pair_id, reference_name, rel_caption, group_members
            
            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                # classic模式用于提取gallery特征，不应用降质
                image = self._load_image(image_name, None)
                return image_name, image
        
        except Exception as e:
            print(f"Exception in __getitem__: {e}")
            return None


class FashionIQDegradedDataset(Dataset):
    """支持降质的FashionIQ数据集（如果需要的话）"""
    # 可以按照类似方式实现
    pass









