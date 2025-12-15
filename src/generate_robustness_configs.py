"""
生成CIRR数据集的各种鲁棒性测试配置

Config A (Clean): 原始数据集
Config B (Noisy Label): 随机打乱20%和50%的三元组配对
Config C (Visual Degradation - Occlusion): Cutout遮挡
Config D (Visual Degradation - Blur): Gaussian/Motion Blur
Config E (Mixed): 同时包含B, C, D的混合噪声

Usage:
    python src/generate_robustness_configs.py --output-dir ../robustness_configs
"""

import json
import random
import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np

# 尝试导入可选的库
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not found. Motion blur will use PIL fallback.")

try:
    from PIL import Image, ImageFilter, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class DegradationType(Enum):
    """降质类型枚举"""
    NONE = "none"
    CUTOUT = "cutout"
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"


class RobustnessConfigGenerator:
    """鲁棒性配置生成器"""
    
    def __init__(
        self,
        cirr_root: str,
        output_dir: str,
        seed: int = 42
    ):
        """
        Args:
            cirr_root: CIRR数据集根目录
            output_dir: 输出目录
            seed: 随机种子
        """
        self.cirr_root = Path(cirr_root)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载原始数据
        self.splits = ['train', 'val', 'test1']
        self.captions = {}
        self.image_splits = {}
        
        self._load_original_data()
    
    def _load_original_data(self):
        """加载原始CIRR数据"""
        for split in self.splits:
            caption_path = self.cirr_root / 'cirr' / 'captions' / f'cap.rc2.{split}.json'
            split_path = self.cirr_root / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
            
            if caption_path.exists():
                with open(caption_path, 'r') as f:
                    self.captions[split] = json.load(f)
                print(f"Loaded {len(self.captions[split])} triplets for {split}")
            
            if split_path.exists():
                with open(split_path, 'r') as f:
                    self.image_splits[split] = json.load(f)
                print(f"Loaded {len(self.image_splits[split])} images for {split}")
    
    # ==================== Config A: Clean ====================
    def generate_config_a(self) -> Dict[str, List[dict]]:
        """
        Config A (Clean): 原始数据集，不做任何修改
        直接复制原始标注
        """
        print("\n" + "=" * 60)
        print("Generating Config A (Clean)")
        print("=" * 60)
        
        config = {}
        for split in self.splits:
            if split in self.captions:
                # 直接复制原始数据
                config[split] = [
                    {
                        **sample,
                        "config": "clean",
                        "is_label_noisy": False,
                        "reference_degradation": "none"
                    }
                    for sample in self.captions[split]
                ]
                print(f"  {split}: {len(config[split])} samples")
        
        return config
    
    # ==================== Config B: Noisy Label ====================
    def generate_config_b(self, noise_ratio: float = 0.2) -> Dict[str, List[dict]]:
        """
        Config B (Noisy Label): 随机打乱指定比例的三元组配对
        参考TME论文的做法，打乱reference-target配对
        
        Args:
            noise_ratio: 噪声比例 (0.2 或 0.5)
        """
        print("\n" + "=" * 60)
        print(f"Generating Config B (Noisy Label - {int(noise_ratio*100)}%)")
        print("=" * 60)
        
        config = {}
        
        for split in ['train']:  # 只对训练集添加噪声
            if split not in self.captions:
                continue
            
            samples = self.captions[split].copy()
            n_samples = len(samples)
            n_noisy = int(n_samples * noise_ratio)
            
            # 随机选择要打乱的样本索引
            noisy_indices = set(random.sample(range(n_samples), n_noisy))
            
            # 收集所有目标图像名
            all_targets = [s['target_hard'] for s in samples]
            
            config[split] = []
            shuffled_count = 0
            
            for i, sample in enumerate(samples):
                new_sample = sample.copy()
                new_sample["config"] = f"noisy_label_{int(noise_ratio*100)}"
                
                if i in noisy_indices:
                    # 随机选择一个不同的目标图像
                    while True:
                        random_target = random.choice(all_targets)
                        if random_target != sample['target_hard']:
                            break
                    
                    new_sample['target_hard'] = random_target
                    new_sample['is_label_noisy'] = True
                    new_sample['original_target'] = sample['target_hard']
                    shuffled_count += 1
                else:
                    new_sample['is_label_noisy'] = False
                
                new_sample['reference_degradation'] = "none"
                config[split].append(new_sample)
            
            print(f"  {split}: {len(config[split])} samples, {shuffled_count} noisy ({shuffled_count/n_samples*100:.1f}%)")
        
        # 验证集和测试集保持不变
        for split in ['val', 'test1']:
            if split in self.captions:
                config[split] = [
                    {
                        **sample,
                        "config": f"noisy_label_{int(noise_ratio*100)}",
                        "is_label_noisy": False,
                        "reference_degradation": "none"
                    }
                    for sample in self.captions[split]
                ]
                print(f"  {split}: {len(config[split])} samples (unchanged)")
        
        return config
    
    # ==================== Config C: Visual Degradation - Cutout ====================
    def generate_config_c_cutout(
        self,
        num_boxes_range: Tuple[int, int] = (3, 5),
        box_size_range: Tuple[float, float] = (0.25, 0.45)
    ) -> Dict[str, List[dict]]:
        """
        Config C (Cutout): 在参考图像上随机生成黑色矩形块遮挡
        遮挡率约为 20%-50%
        
        生成的是变换参数索引，实际变换在数据加载时执行
        """
        print("\n" + "=" * 60)
        print("Generating Config C (Cutout)")
        print("=" * 60)
        
        config = {}
        
        for split in ['train']:
            if split not in self.captions:
                continue
            
            config[split] = []
            
            for sample in self.captions[split]:
                # 生成随机cutout参数
                num_boxes = random.randint(*num_boxes_range)
                boxes = []
                
                for _ in range(num_boxes):
                    # 随机生成矩形位置和大小（归一化坐标）
                    box_w = random.uniform(*box_size_range)
                    box_h = random.uniform(*box_size_range)
                    box_x = random.uniform(0, 1 - box_w)
                    box_y = random.uniform(0, 1 - box_h)
                    boxes.append({
                        'x': round(box_x, 4),
                        'y': round(box_y, 4),
                        'w': round(box_w, 4),
                        'h': round(box_h, 4)
                    })
                
                new_sample = sample.copy()
                new_sample.update({
                    "config": "cutout",
                    "is_label_noisy": False,
                    "reference_degradation": "cutout",
                    "degradation_params": {
                        "type": "cutout",
                        "boxes": boxes
                    }
                })
                config[split].append(new_sample)
            
            print(f"  {split}: {len(config[split])} samples with cutout")
        
        # 验证集和测试集也应用相同的降质
        for split in ['val', 'test1']:
            if split in self.captions:
                config[split] = []
                for sample in self.captions[split]:
                    # 生成随机cutout参数
                    num_boxes = random.randint(*num_boxes_range)
                    boxes = []
                    
                    for _ in range(num_boxes):
                        box_w = random.uniform(*box_size_range)
                        box_h = random.uniform(*box_size_range)
                        box_x = random.uniform(0, 1 - box_w)
                        box_y = random.uniform(0, 1 - box_h)
                        boxes.append({
                            'x': round(box_x, 4),
                            'y': round(box_y, 4),
                            'w': round(box_w, 4),
                            'h': round(box_h, 4)
                        })
                    
                    new_sample = sample.copy()
                    new_sample.update({
                        "config": "cutout",
                        "is_label_noisy": False,
                        "reference_degradation": "cutout",
                        "degradation_params": {
                            "type": "cutout",
                            "boxes": boxes
                        }
                    })
                    config[split].append(new_sample)
                print(f"  {split}: {len(config[split])} samples with cutout")
        
        return config
    
    # ==================== Config D: Visual Degradation - Blur ====================
    def generate_config_d(
        self,
        blur_type: str = "gaussian",
        kernel_range: Tuple[int, int] = (5, 15),
        sigma_range: Tuple[float, float] = (1.0, 5.0)
    ) -> Dict[str, List[dict]]:
        """
        Config D (Blur): 对参考图像应用模糊
        
        Args:
            blur_type: "gaussian" 或 "motion"
        """
        print("\n" + "=" * 60)
        print(f"Generating Config D ({blur_type.capitalize()} Blur)")
        print("=" * 60)
        
        config = {}
        
        for split in ['train']:
            if split not in self.captions:
                continue
            
            config[split] = []
            
            for sample in self.captions[split]:
                # 生成随机模糊参数
                kernel_size = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)  # 确保奇数
                
                blur_params = {
                    "type": f"{blur_type}_blur",
                    "kernel_size": kernel_size
                }
                
                if blur_type == "gaussian":
                    blur_params["sigma"] = round(random.uniform(*sigma_range), 2)
                elif blur_type == "motion":
                    # 运动模糊方向角度
                    blur_params["angle"] = round(random.uniform(0, 360), 1)
                
                new_sample = sample.copy()
                new_sample.update({
                    "config": f"{blur_type}_blur",
                    "is_label_noisy": False,
                    "reference_degradation": f"{blur_type}_blur",
                    "degradation_params": blur_params
                })
                config[split].append(new_sample)
            
            print(f"  {split}: {len(config[split])} samples with {blur_type} blur")
        
        # 验证集和测试集也应用相同的降质
        for split in ['val', 'test1']:
            if split in self.captions:
                config[split] = []
                for sample in self.captions[split]:
                    # 生成随机模糊参数
                    kernel_size = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)
                    
                    blur_params = {
                        "type": f"{blur_type}_blur",
                        "kernel_size": kernel_size
                    }
                    
                    if blur_type == "gaussian":
                        blur_params["sigma"] = round(random.uniform(*sigma_range), 2)
                    elif blur_type == "motion":
                        blur_params["angle"] = round(random.uniform(0, 360), 1)
                    
                    new_sample = sample.copy()
                    new_sample.update({
                        "config": f"{blur_type}_blur",
                        "is_label_noisy": False,
                        "reference_degradation": f"{blur_type}_blur",
                        "degradation_params": blur_params
                    })
                    config[split].append(new_sample)
                print(f"  {split}: {len(config[split])} samples with {blur_type} blur")
        
        return config
    
    # ==================== Config E: Mixed ====================
    def generate_config_e(
        self,
        noisy_ratio: float = 0.15,
        cutout_ratio: float = 0.20,
        blur_ratio: float = 0.20
    ) -> Dict[str, List[dict]]:
        """
        Config E (Mixed - Real World): 混合B, C, D的各种噪声
        
        每种噪声类型独立应用到指定比例的样本上
        一个样本可能同时具有多种噪声
        """
        print("\n" + "=" * 60)
        print("Generating Config E (Mixed - Real World)")
        print(f"  Noisy labels: {noisy_ratio*100:.0f}%")
        print(f"  Cutout: {cutout_ratio*100:.0f}%")
        print(f"  Blur: {blur_ratio*100:.0f}%")
        print("=" * 60)
        
        config = {}
        
        for split in ['train']:
            if split not in self.captions:
                continue
            
            samples = self.captions[split]
            n_samples = len(samples)
            
            # 随机选择各类噪声的样本索引
            all_indices = list(range(n_samples))
            noisy_indices = set(random.sample(all_indices, int(n_samples * noisy_ratio)))
            cutout_indices = set(random.sample(all_indices, int(n_samples * cutout_ratio)))
            blur_indices = set(random.sample(all_indices, int(n_samples * blur_ratio)))
            
            all_targets = [s['target_hard'] for s in samples]
            
            config[split] = []
            stats = {'noisy': 0, 'cutout': 0, 'blur': 0, 'clean': 0, 'multi': 0}
            
            for i, sample in enumerate(samples):
                new_sample = sample.copy()
                degradations = []
                params_list = []
                
                # 标签噪声
                if i in noisy_indices:
                    while True:
                        random_target = random.choice(all_targets)
                        if random_target != sample['target_hard']:
                            break
                    new_sample['target_hard'] = random_target
                    new_sample['is_label_noisy'] = True
                    new_sample['original_target'] = sample['target_hard']
                    degradations.append('noisy_label')
                    stats['noisy'] += 1
                else:
                    new_sample['is_label_noisy'] = False
                
                # Cutout
                if i in cutout_indices:
                    num_boxes = random.randint(1, 3)
                    boxes = []
                    for _ in range(num_boxes):
                        box_w = random.uniform(0.1, 0.3)
                        box_h = random.uniform(0.1, 0.3)
                        boxes.append({
                            'x': round(random.uniform(0, 1 - box_w), 4),
                            'y': round(random.uniform(0, 1 - box_h), 4),
                            'w': round(box_w, 4),
                            'h': round(box_h, 4)
                        })
                    params_list.append({'type': 'cutout', 'boxes': boxes})
                    degradations.append('cutout')
                    stats['cutout'] += 1
                
                # Blur (随机选择gaussian或motion)
                if i in blur_indices:
                    blur_type = random.choice(['gaussian', 'motion'])
                    kernel_size = random.randrange(5, 15, 2)
                    blur_params = {'type': f'{blur_type}_blur', 'kernel_size': kernel_size}
                    if blur_type == 'gaussian':
                        blur_params['sigma'] = round(random.uniform(1.0, 3.0), 2)
                    else:
                        blur_params['angle'] = round(random.uniform(0, 360), 1)
                    params_list.append(blur_params)
                    degradations.append(f'{blur_type}_blur')
                    stats['blur'] += 1
                
                if not degradations:
                    degradations.append('clean')
                    stats['clean'] += 1
                elif len(degradations) > 1:
                    stats['multi'] += 1
                
                new_sample.update({
                    "config": "mixed",
                    "reference_degradation": "+".join(degradations),
                    "degradation_params": params_list if params_list else None
                })
                config[split].append(new_sample)
            
            print(f"  {split}: {len(config[split])} samples")
            print(f"    - Noisy labels: {stats['noisy']}")
            print(f"    - Cutout: {stats['cutout']}")
            print(f"    - Blur: {stats['blur']}")
            print(f"    - Clean: {stats['clean']}")
            print(f"    - Multi-degradation: {stats['multi']}")
        
        # 验证集和测试集也应用相同的视觉降质（但不添加标签噪声）
        for split in ['val', 'test1']:
            if split in self.captions:
                samples = self.captions[split]
                n_samples = len(samples)
                
                # 对验证/测试集也随机应用视觉降质（但不添加标签噪声）
                all_indices = list(range(n_samples))
                cutout_indices = set(random.sample(all_indices, int(n_samples * cutout_ratio)))
                blur_indices = set(random.sample(all_indices, int(n_samples * blur_ratio)))
                
                config[split] = []
                stats_split = {'cutout': 0, 'blur': 0, 'clean': 0}
                
                for i, sample in enumerate(samples):
                    new_sample = sample.copy()
                    degradations = []
                    params_list = []
                    
                    # 不添加标签噪声
                    new_sample['is_label_noisy'] = False
                    
                    # Cutout
                    if i in cutout_indices:
                        num_boxes = random.randint(1, 3)
                        boxes = []
                        for _ in range(num_boxes):
                            box_w = random.uniform(0.1, 0.3)
                            box_h = random.uniform(0.1, 0.3)
                            boxes.append({
                                'x': round(random.uniform(0, 1 - box_w), 4),
                                'y': round(random.uniform(0, 1 - box_h), 4),
                                'w': round(box_w, 4),
                                'h': round(box_h, 4)
                            })
                        params_list.append({'type': 'cutout', 'boxes': boxes})
                        degradations.append('cutout')
                        stats_split['cutout'] += 1
                    
                    # Blur
                    if i in blur_indices:
                        blur_type = random.choice(['gaussian', 'motion'])
                        kernel_size = random.randrange(5, 15, 2)
                        blur_params = {'type': f'{blur_type}_blur', 'kernel_size': kernel_size}
                        if blur_type == 'gaussian':
                            blur_params['sigma'] = round(random.uniform(1.0, 3.0), 2)
                        else:
                            blur_params['angle'] = round(random.uniform(0, 360), 1)
                        params_list.append(blur_params)
                        degradations.append(f'{blur_type}_blur')
                        stats_split['blur'] += 1
                    
                    if not degradations:
                        degradations.append('clean')
                        stats_split['clean'] += 1
                    
                    new_sample.update({
                        "config": "mixed",
                        "reference_degradation": "+".join(degradations),
                        "degradation_params": params_list if params_list else None
                    })
                    config[split].append(new_sample)
                
                print(f"  {split}: {len(config[split])} samples")
                print(f"    - Cutout: {stats_split['cutout']}")
                print(f"    - Blur: {stats_split['blur']}")
                print(f"    - Clean: {stats_split['clean']}")
        
        return config
    
    def save_config(self, config: Dict[str, List[dict]], config_name: str):
        """保存配置到JSON文件"""
        config_dir = self.output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        for split, samples in config.items():
            output_path = config_dir / f'cap.rc2.{split}.json'
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {output_path}")
        
        # 复制image_splits（不变）
        for split in self.splits:
            if split in self.image_splits:
                src_path = self.cirr_root / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
                dst_path = config_dir / f'split.rc2.{split}.json'
                if src_path.exists():
                    shutil.copy(src_path, dst_path)
        
        # 保存配置元信息
        meta = {
            "config_name": config_name,
            "seed": self.seed,
            "splits": list(config.keys()),
            "sample_counts": {split: len(samples) for split, samples in config.items()}
        }
        with open(config_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
    
    def generate_all_configs(self):
        """生成所有配置"""
        print("\n" + "=" * 80)
        print("CIRR Robustness Benchmark Config Generator")
        print("=" * 80)
        
        # Config A: Clean
        config_a = self.generate_config_a()
        self.save_config(config_a, "config_a_clean")
        
        # Config B: Noisy Label (20% and 50%)
        for ratio in [0.2, 0.5]:
            config_b = self.generate_config_b(noise_ratio=ratio)
            self.save_config(config_b, f"config_b_noisy_{int(ratio*100)}")
        
        # Config C: Visual Degradation - Cutout
        config_c = self.generate_config_c_cutout()
        self.save_config(config_c, "config_c_cutout")
        
        # Config D: Visual Degradation - Blur
        config_d_gaussian = self.generate_config_d(blur_type="gaussian")
        self.save_config(config_d_gaussian, "config_d_gaussian_blur")
        
        config_d_motion = self.generate_config_d(blur_type="motion")
        self.save_config(config_d_motion, "config_d_motion_blur")
        
        # Config E: Mixed
        config_e = self.generate_config_e()
        self.save_config(config_e, "config_e_mixed")
        
        print("\n" + "=" * 80)
        print("All configs generated successfully!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)


# ==================== 图像变换工具类 ====================
class ImageDegradation:
    """
    图像降质变换工具类（用于数据加载时应用变换）
    
    Usage:
        from generate_robustness_configs import ImageDegradation
        
        # 单个变换
        degraded_img = ImageDegradation.apply_degradation(img, params)
        
        # 多个变换（Config E）
        degraded_img = ImageDegradation.apply_multi_degradation(img, params_list)
    """
    
    @staticmethod
    def apply_cutout(image, boxes: List[dict]):
        """
        应用Cutout遮挡
        
        Args:
            image: PIL Image
            boxes: 遮挡框列表，每个框包含 {x, y, w, h} (归一化坐标)
        
        Returns:
            PIL Image
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for box in boxes:
            x1 = int(box['x'] * w)
            y1 = int(box['y'] * h)
            x2 = int((box['x'] + box['w']) * w)
            y2 = int((box['y'] + box['h']) * h)
            
            # 确保坐标在有效范围内
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            img_array[y1:y2, x1:x2] = 0  # 黑色填充
        
        if HAS_PIL:
            return Image.fromarray(img_array)
        return img_array
    
    @staticmethod
    def apply_gaussian_blur(image, kernel_size: int, sigma: float):
        """
        应用高斯模糊
        
        Args:
            image: PIL Image
            kernel_size: 模糊核大小
            sigma: 高斯sigma
        
        Returns:
            PIL Image
        """
        if HAS_PIL:
            # PIL的GaussianBlur只接受radius参数
            radius = max(1, kernel_size // 2)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif HAS_CV2:
            img_array = np.array(image)
            blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
            return Image.fromarray(blurred)
        return image
    
    @staticmethod
    def apply_motion_blur(image, kernel_size: int, angle: float):
        """
        应用运动模糊
        
        Args:
            image: PIL Image
            kernel_size: 模糊核大小
            angle: 运动方向角度 (0-360)
        
        Returns:
            PIL Image
        """
        if HAS_CV2:
            img_array = np.array(image)
            
            # 创建运动模糊核
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1
            kernel = kernel / kernel_size
            
            # 旋转核
            center = (kernel_size // 2, kernel_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / (kernel.sum() + 1e-8)
            
            # 应用模糊
            blurred = cv2.filter2D(img_array, -1, kernel)
            return Image.fromarray(blurred)
        elif HAS_PIL:
            # PIL fallback - 使用简单的BoxBlur
            radius = max(1, kernel_size // 2)
            return image.filter(ImageFilter.BoxBlur(radius=radius))
        return image
    
    @classmethod
    def apply_degradation(cls, image, params: dict):
        """
        根据参数应用单个降质
        
        Args:
            image: PIL Image
            params: 降质参数字典
        
        Returns:
            PIL Image
        """
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
        else:
            return image
    
    @classmethod
    def apply_multi_degradation(cls, image, params_list: List[dict]):
        """
        应用多个降质（用于Config E）
        
        Args:
            image: PIL Image
            params_list: 降质参数列表
        
        Returns:
            PIL Image
        """
        if not params_list:
            return image
        
        result = image
        for params in params_list:
            result = cls.apply_degradation(result, params)
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate CIRR robustness benchmark configs"
    )
    parser.add_argument(
        '--cirr-root',
        type=str,
        default='./cirr_dataset',
        help='Path to CIRR dataset root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../robustness_configs',
        help='Output directory for generated configs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=['all', 'a', 'b', 'c', 'd', 'e'],
        default='all',
        help='Which config to generate'
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录，以便计算相对路径
    script_dir = Path(__file__).parent.absolute()
    
    # 处理相对路径
    if not os.path.isabs(args.cirr_root):
        cirr_root = (script_dir / args.cirr_root).resolve()
    else:
        cirr_root = Path(args.cirr_root)
    
    if not os.path.isabs(args.output_dir):
        output_dir = (script_dir / args.output_dir).resolve()
    else:
        output_dir = Path(args.output_dir)
    
    print(f"CIRR root: {cirr_root}")
    print(f"Output dir: {output_dir}")
    
    generator = RobustnessConfigGenerator(
        cirr_root=str(cirr_root),
        output_dir=str(output_dir),
        seed=args.seed
    )
    
    if args.config == 'all':
        generator.generate_all_configs()
    else:
        # 单独生成指定配置
        if args.config == 'a':
            config = generator.generate_config_a()
            generator.save_config(config, "config_a_clean")
        elif args.config == 'b':
            for ratio in [0.2, 0.5]:
                config = generator.generate_config_b(noise_ratio=ratio)
                generator.save_config(config, f"config_b_noisy_{int(ratio*100)}")
        elif args.config == 'c':
            config = generator.generate_config_c_cutout()
            generator.save_config(config, "config_c_cutout")
        elif args.config == 'd':
            config_g = generator.generate_config_d(blur_type="gaussian")
            generator.save_config(config_g, "config_d_gaussian_blur")
            config_m = generator.generate_config_d(blur_type="motion")
            generator.save_config(config_m, "config_d_motion_blur")
        elif args.config == 'e':
            config = generator.generate_config_e()
            generator.save_config(config, "config_e_mixed")


if __name__ == "__main__":
    main()

