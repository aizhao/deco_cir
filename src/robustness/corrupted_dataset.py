"""
CIR 带损坏的数据集模块
======================
封装 CIRR 和 FashionIQ 数据集，在 __getitem__ 中动态应用图像损坏

核心设计:
1. 损坏在 preprocess 之前应用 (PIL.Image 阶段)
2. 仅对参考图像 (reference_image) 应用损坏
3. 目标图像 (target_image) 保持干净
4. 支持在线 (online) 和确定性 (deterministic) 两种模式

Usage:
    from robustness import CorruptedCIRRDataset, CorruptionConfig
    
    config = CorruptionConfig.from_preset('moderate')
    dataset = CorruptedCIRRDataset(
        split='val',
        mode='relative',
        preprocess=preprocess,
        corruption_config=config
    )
    
    # 或使用消融配置
    config = CorruptionConfig.ablation('gaussian_noise', severity=3)
    dataset = CorruptedCIRRDataset(split='val', mode='relative', preprocess=preprocess, corruption_config=config)
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
import PIL.Image
from torch.utils.data import Dataset

from .image_corruption import ImageCorruptor
from .text_perturbation import TextPerturber
from .corruption_config import CorruptionConfig


# 获取项目根路径
base_path = Path(__file__).absolute().parents[2].absolute()


class CorruptedCIRRDataset(Dataset):
    """
    带损坏的 CIRR 数据集
    
    在 __getitem__ 中对参考图像动态应用损坏
    
    Attributes:
        split: 数据集划分 ('train', 'val', 'test1')
        mode: 数据集模式 ('relative', 'classic')
        preprocess: 图像预处理函数
        corruption_config: 损坏配置
    """
    
    def __init__(
        self,
        split: str,
        mode: str,
        preprocess: callable,
        corruption_config: Optional[CorruptionConfig] = None,
    ):
        """
        初始化带损坏的 CIRR 数据集
        
        Args:
            split: 数据集划分 ('train', 'val', 'test1')
            mode: 数据集模式
                - 'relative': 返回 (reference_image, target_image, caption) 或相关变体
                - 'classic': 返回 (image_name, image)
            preprocess: 图像预处理函数 (如 targetpad_transform)
            corruption_config: 损坏配置，None 表示不应用损坏
        """
        self.split = split
        self.mode = mode
        self.preprocess = preprocess
        self.corruption_config = corruption_config or CorruptionConfig(image_corruption_prob=0.0)
        
        # 验证参数
        if split not in ['train', 'val', 'test1']:
            raise ValueError(f"split should be in ['train', 'val', 'test1'], got {split}")
        if mode not in ['relative', 'classic']:
            raise ValueError(f"mode should be in ['relative', 'classic'], got {mode}")
        
        # 加载数据
        with open(base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)
        
        with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)
        
        # 初始化损坏器
        self._corruptor = ImageCorruptor(seed=self.corruption_config.seed)
        self._perturber = TextPerturber(seed=self.corruption_config.seed)
        self._rng = np.random.RandomState(self.corruption_config.seed)
        
        # 如果是确定性模式，预生成每个样本的损坏配置
        self._sample_corruptions = None
        self._sample_perturbations = None
        if self.corruption_config.deterministic:
            self._precompute_corruptions()
            self._precompute_text_perturbations()
        
        print(f"CorruptedCIRR {split} dataset in {mode} mode initialized")
        print(f"  Corruption config: {self.corruption_config}")
        print(f"  Total samples: {len(self)}")
    
    def _precompute_corruptions(self):
        """预计算每个样本的损坏配置 (确定性模式)"""
        self._sample_corruptions = []
        
        for idx in range(len(self)):
            # 使用索引作为种子的一部分，确保每个样本有固定的损坏
            sample_seed = self.corruption_config.seed + idx
            rng = np.random.RandomState(sample_seed)
            
            if rng.random() < self.corruption_config.image_corruption_prob:
                # 选择损坏类型
                if self.corruption_config.image_corruption_types:
                    corruption_type = rng.choice(self.corruption_config.image_corruption_types)
                else:
                    # 根据层级/类别筛选
                    available = self._get_available_corruptions()
                    corruption_type = rng.choice(available)
                
                # 选择严重度
                min_sev, max_sev = self.corruption_config.image_severity_range
                severity = rng.randint(min_sev, max_sev + 1)
                
                self._sample_corruptions.append({
                    'apply': True,
                    'type': corruption_type,
                    'severity': severity,
                    'seed': sample_seed
                })
            else:
                self._sample_corruptions.append({'apply': False})
    
    def _get_available_corruptions(self) -> List[str]:
        """获取当前配置下可用的图像损坏类型"""
        from .image_corruption import ImageCorruptor, CorruptionLevel, CorruptionCategory
        
        available = list(ImageCorruptor.CORRUPTION_REGISTRY.keys())
        
        if self.corruption_config.image_corruption_level:
            level = CorruptionLevel(self.corruption_config.image_corruption_level)
            available = [
                k for k, v in ImageCorruptor.CORRUPTION_REGISTRY.items()
                if v.level == level
            ]
        
        if self.corruption_config.image_corruption_category:
            cat = CorruptionCategory(self.corruption_config.image_corruption_category)
            available = [
                k for k in available
                if ImageCorruptor.CORRUPTION_REGISTRY[k].category == cat
            ]
        
        return available
    
    def _get_available_perturbations(self) -> List[str]:
        """获取当前配置下可用的文本扰动类型"""
        from .text_perturbation import TextPerturber, PerturbationLevel, PerturbationCategory
        
        available = list(TextPerturber.PERTURBATION_REGISTRY.keys())
        
        if self.corruption_config.text_perturbation_level:
            level = PerturbationLevel(self.corruption_config.text_perturbation_level)
            available = [
                k for k, v in TextPerturber.PERTURBATION_REGISTRY.items()
                if v.level == level
            ]
        
        if self.corruption_config.text_perturbation_category:
            cat = PerturbationCategory(self.corruption_config.text_perturbation_category)
            available = [
                k for k in available
                if TextPerturber.PERTURBATION_REGISTRY[k].category == cat
            ]
        
        return available
    
    def _precompute_text_perturbations(self):
        """预计算每个样本的文本扰动配置 (确定性模式)"""
        self._sample_perturbations = []
        
        for idx in range(len(self)):
            # 使用索引作为种子的一部分
            sample_seed = self.corruption_config.seed + idx + 10000  # 偏移避免与图像重复
            rng = np.random.RandomState(sample_seed)
            
            if rng.random() < self.corruption_config.text_perturbation_prob:
                # 选择扰动类型
                if self.corruption_config.text_perturbation_types:
                    perturbation_type = rng.choice(self.corruption_config.text_perturbation_types)
                else:
                    available = self._get_available_perturbations()
                    perturbation_type = rng.choice(available)
                
                # 选择严重度
                min_sev, max_sev = self.corruption_config.text_severity_range
                severity = rng.randint(min_sev, max_sev + 1)
                
                self._sample_perturbations.append({
                    'apply': True,
                    'type': perturbation_type,
                    'severity': severity,
                    'seed': sample_seed
                })
            else:
                self._sample_perturbations.append({'apply': False})
    
    def _apply_corruption(
        self, 
        image: PIL.Image.Image, 
        index: int
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        """
        对图像应用损坏
        
        Args:
            image: PIL.Image 格式的输入图像
            index: 样本索引
            
        Returns:
            (corrupted_image, corruption_info)
        """
        corruption_info = {
            'applied': False,
            'type': None,
            'severity': None,
        }
        
        # 检查是否应用损坏
        if self.corruption_config.mode == 'clean':
            return image, corruption_info
        
        if self.corruption_config.deterministic and self._sample_corruptions:
            # 确定性模式：使用预计算的配置
            sample_config = self._sample_corruptions[index]
            if not sample_config['apply']:
                return image, corruption_info
            
            corruption_type = sample_config['type']
            severity = sample_config['severity']
            self._corruptor.set_seed(sample_config['seed'])
        else:
            # 在线随机模式
            if self._rng.random() >= self.corruption_config.image_corruption_prob:
                return image, corruption_info
            
            # 选择损坏类型
            if self.corruption_config.image_corruption_types:
                corruption_type = self._rng.choice(self.corruption_config.image_corruption_types)
            else:
                available = self._get_available_corruptions()
                corruption_type = self._rng.choice(available)
            
            # 选择严重度
            min_sev, max_sev = self.corruption_config.image_severity_range
            severity = self._rng.randint(min_sev, max_sev + 1)
        
        # 应用损坏
        corrupted_array = self._corruptor.apply(image, corruption_type, severity)
        corrupted_image = PIL.Image.fromarray(corrupted_array)
        
        corruption_info = {
            'applied': True,
            'type': corruption_type,
            'severity': severity,
        }
        
        return corrupted_image, corruption_info
    
    def _apply_text_perturbation(
        self,
        text: str,
        index: int
    ) -> Tuple[str, Dict[str, Any]]:
        """
        对文本应用扰动
        
        Args:
            text: 输入文本 (caption)
            index: 样本索引
            
        Returns:
            (perturbed_text, perturbation_info)
        """
        perturbation_info = {
            'applied': False,
            'type': None,
            'severity': None,
        }
        
        # 检查是否应用扰动
        if self.corruption_config.mode == 'clean':
            return text, perturbation_info
        
        if self.corruption_config.text_perturbation_prob <= 0:
            return text, perturbation_info
        
        if self.corruption_config.deterministic and self._sample_perturbations:
            # 确定性模式：使用预计算的配置
            sample_config = self._sample_perturbations[index]
            if not sample_config['apply']:
                return text, perturbation_info
            
            perturbation_type = sample_config['type']
            severity = sample_config['severity']
            self._perturber.set_seed(sample_config['seed'])
        else:
            # 在线随机模式
            if self._rng.random() >= self.corruption_config.text_perturbation_prob:
                return text, perturbation_info
            
            # 选择扰动类型
            if self.corruption_config.text_perturbation_types:
                perturbation_type = self._rng.choice(self.corruption_config.text_perturbation_types)
            else:
                available = self._get_available_perturbations()
                perturbation_type = self._rng.choice(available)
            
            # 选择严重度
            min_sev, max_sev = self.corruption_config.text_severity_range
            severity = self._rng.randint(min_sev, max_sev + 1)
        
        # 应用扰动
        perturbed_text = self._perturber.apply(text, perturbation_type, severity)
        
        perturbation_info = {
            'applied': True,
            'type': perturbation_type,
            'severity': severity,
        }
        
        return perturbed_text, perturbation_info
    
    def __getitem__(self, index: int):
        """
        获取数据样本
        
        Returns:
            根据 mode 和 split 返回不同格式:
            
            mode='relative', split='train':
                - return_corruption_info=False: (ref_img, target_img, caption)
                - return_corruption_info=True: (ref_img, target_img, caption, corruption_info)
            
            mode='relative', split='val':
                (ref_name, target_name, caption, group_members)
            
            mode='relative', split='test1':
                (pair_id, ref_name, caption, group_members)
            
            mode='classic':
                (image_name, image)
        """
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']
                
                if self.split == 'train':
                    # 加载参考图像
                    reference_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[reference_name]
                    reference_image = PIL.Image.open(reference_image_path).convert('RGB')
                    
                    # ★ 应用图像损坏 (在 preprocess 之前)
                    reference_image, corruption_info = self._apply_corruption(reference_image, index)
                    
                    # ★ 应用文本扰动
                    rel_caption, text_perturbation_info = self._apply_text_perturbation(rel_caption, index)
                    
                    # 预处理
                    reference_image = self.preprocess(reference_image)
                    
                    # 加载目标图像 (不损坏)
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert('RGB'))
                    
                    if self.corruption_config.return_corruption_info:
                        # 返回完整的双模态损坏信息
                        full_corruption_info = {
                            'image': corruption_info,
                            'text': text_perturbation_info,
                        }
                        return reference_image, target_image, rel_caption, full_corruption_info
                    return reference_image, target_image, rel_caption
                
                elif self.split == 'val':
                    # 验证集：也应用文本扰动
                    rel_caption, _ = self._apply_text_perturbation(rel_caption, index)
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members
                
                elif self.split == 'test1':
                    # 测试集：也应用文本扰动
                    rel_caption, _ = self._apply_text_perturbation(rel_caption, index)
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members
            
            elif self.mode == 'classic':
                # Gallery 模式，用于提取目标图像特征
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                image = PIL.Image.open(image_path).convert('RGB')
                
                # Gallery 图像也可以选择性地应用损坏 (通常不应用)
                # 这里保持干净
                image = self.preprocess(image)
                return image_name, image
        
        except Exception as e:
            print(f"Exception at index {index}: {e}")
            return None
    
    def __len__(self) -> int:
        if self.mode == 'relative':
            return len(self.triplets)
        return len(self.name_to_relpath)
    
    def get_corruption_stats(self) -> Dict[str, Any]:
        """获取损坏统计信息 (仅确定性模式)"""
        if not self._sample_corruptions:
            return {"message": "Stats only available in deterministic mode"}
        
        applied_count = sum(1 for c in self._sample_corruptions if c['apply'])
        type_counts = {}
        severity_counts = {}
        
        for c in self._sample_corruptions:
            if c['apply']:
                t = c['type']
                s = c['severity']
                type_counts[t] = type_counts.get(t, 0) + 1
                severity_counts[s] = severity_counts.get(s, 0) + 1
        
        return {
            'total_samples': len(self._sample_corruptions),
            'corrupted_samples': applied_count,
            'corruption_rate': applied_count / len(self._sample_corruptions),
            'type_distribution': type_counts,
            'severity_distribution': severity_counts,
        }


class CorruptedFashionIQDataset(Dataset):
    """
    带损坏的 FashionIQ 数据集
    
    在 __getitem__ 中对参考图像动态应用损坏
    """
    
    def __init__(
        self,
        split: str,
        dress_types: List[str],
        mode: str,
        preprocess: callable,
        corruption_config: Optional[CorruptionConfig] = None,
    ):
        """
        初始化带损坏的 FashionIQ 数据集
        
        Args:
            split: 数据集划分 ('train', 'val', 'test')
            dress_types: 服装类型列表 ['dress', 'shirt', 'toptee']
            mode: 数据集模式 ('relative', 'classic')
            preprocess: 图像预处理函数
            corruption_config: 损坏配置
        """
        self.split = split
        self.dress_types = dress_types
        self.mode = mode
        self.preprocess = preprocess
        self.corruption_config = corruption_config or CorruptionConfig(image_corruption_prob=0.0)
        
        # 验证参数
        if mode not in ['relative', 'classic']:
            raise ValueError(f"mode should be in ['relative', 'classic'], got {mode}")
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split should be in ['train', 'val', 'test'], got {split}")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError(f"dress_type should be in ['dress', 'shirt', 'toptee'], got {dress_type}")
        
        # 加载数据
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))
        
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))
        
        # 初始化损坏器
        self._corruptor = ImageCorruptor(seed=self.corruption_config.seed)
        self._perturber = TextPerturber(seed=self.corruption_config.seed)
        self._rng = np.random.RandomState(self.corruption_config.seed)
        
        # 确定性模式预计算
        self._sample_corruptions = None
        self._sample_perturbations = None
        if self.corruption_config.deterministic:
            self._precompute_corruptions()
            self._precompute_text_perturbations()
        
        print(f"CorruptedFashionIQ {split} - {dress_types} dataset in {mode} mode initialized")
        print(f"  Corruption config: {self.corruption_config}")
        print(f"  Total samples: {len(self)}")
    
    def _precompute_corruptions(self):
        """预计算每个样本的损坏配置"""
        self._sample_corruptions = []
        
        for idx in range(len(self)):
            sample_seed = self.corruption_config.seed + idx
            rng = np.random.RandomState(sample_seed)
            
            if rng.random() < self.corruption_config.image_corruption_prob:
                available = self._get_available_corruptions()
                if self.corruption_config.image_corruption_types:
                    corruption_type = rng.choice(self.corruption_config.image_corruption_types)
                else:
                    corruption_type = rng.choice(available)
                
                min_sev, max_sev = self.corruption_config.image_severity_range
                severity = rng.randint(min_sev, max_sev + 1)
                
                self._sample_corruptions.append({
                    'apply': True,
                    'type': corruption_type,
                    'severity': severity,
                    'seed': sample_seed
                })
            else:
                self._sample_corruptions.append({'apply': False})
    
    def _get_available_corruptions(self) -> List[str]:
        """获取当前配置下可用的图像损坏类型"""
        from .image_corruption import ImageCorruptor, CorruptionLevel, CorruptionCategory
        
        available = list(ImageCorruptor.CORRUPTION_REGISTRY.keys())
        
        if self.corruption_config.image_corruption_level:
            level = CorruptionLevel(self.corruption_config.image_corruption_level)
            available = [
                k for k, v in ImageCorruptor.CORRUPTION_REGISTRY.items()
                if v.level == level
            ]
        
        if self.corruption_config.image_corruption_category:
            cat = CorruptionCategory(self.corruption_config.image_corruption_category)
            available = [
                k for k in available
                if ImageCorruptor.CORRUPTION_REGISTRY[k].category == cat
            ]
        
        return available
    
    def _get_available_perturbations(self) -> List[str]:
        """获取当前配置下可用的文本扰动类型"""
        from .text_perturbation import TextPerturber, PerturbationLevel, PerturbationCategory
        
        available = list(TextPerturber.PERTURBATION_REGISTRY.keys())
        
        if self.corruption_config.text_perturbation_level:
            level = PerturbationLevel(self.corruption_config.text_perturbation_level)
            available = [
                k for k, v in TextPerturber.PERTURBATION_REGISTRY.items()
                if v.level == level
            ]
        
        if self.corruption_config.text_perturbation_category:
            cat = PerturbationCategory(self.corruption_config.text_perturbation_category)
            available = [
                k for k in available
                if TextPerturber.PERTURBATION_REGISTRY[k].category == cat
            ]
        
        return available
    
    def _precompute_text_perturbations(self):
        """预计算每个样本的文本扰动配置 (确定性模式)"""
        self._sample_perturbations = []
        
        for idx in range(len(self)):
            sample_seed = self.corruption_config.seed + idx + 10000
            rng = np.random.RandomState(sample_seed)
            
            if rng.random() < self.corruption_config.text_perturbation_prob:
                if self.corruption_config.text_perturbation_types:
                    perturbation_type = rng.choice(self.corruption_config.text_perturbation_types)
                else:
                    available = self._get_available_perturbations()
                    perturbation_type = rng.choice(available)
                
                min_sev, max_sev = self.corruption_config.text_severity_range
                severity = rng.randint(min_sev, max_sev + 1)
                
                self._sample_perturbations.append({
                    'apply': True,
                    'type': perturbation_type,
                    'severity': severity,
                    'seed': sample_seed
                })
            else:
                self._sample_perturbations.append({'apply': False})
    
    def _apply_corruption(
        self, 
        image: PIL.Image.Image, 
        index: int
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        """对图像应用损坏"""
        corruption_info = {'applied': False, 'type': None, 'severity': None}
        
        if self.corruption_config.mode == 'clean':
            return image, corruption_info
        
        if self.corruption_config.deterministic and self._sample_corruptions:
            sample_config = self._sample_corruptions[index]
            if not sample_config['apply']:
                return image, corruption_info
            
            corruption_type = sample_config['type']
            severity = sample_config['severity']
            self._corruptor.set_seed(sample_config['seed'])
        else:
            if self._rng.random() >= self.corruption_config.image_corruption_prob:
                return image, corruption_info
            
            if self.corruption_config.image_corruption_types:
                corruption_type = self._rng.choice(self.corruption_config.image_corruption_types)
            else:
                available = self._get_available_corruptions()
                corruption_type = self._rng.choice(available)
            
            min_sev, max_sev = self.corruption_config.image_severity_range
            severity = self._rng.randint(min_sev, max_sev + 1)
        
        corrupted_array = self._corruptor.apply(image, corruption_type, severity)
        corrupted_image = PIL.Image.fromarray(corrupted_array)
        
        corruption_info = {
            'applied': True,
            'type': corruption_type,
            'severity': severity,
        }
        
        return corrupted_image, corruption_info
    
    def _apply_text_perturbation(
        self,
        text: Union[str, List[str]],
        index: int
    ) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
        """
        对文本应用扰动 (支持单个字符串或字符串列表)
        
        Args:
            text: 输入文本 (FashionIQ的caption是列表)
            index: 样本索引
        """
        perturbation_info = {'applied': False, 'type': None, 'severity': None}
        
        if self.corruption_config.mode == 'clean':
            return text, perturbation_info
        
        if self.corruption_config.text_perturbation_prob <= 0:
            return text, perturbation_info
        
        if self.corruption_config.deterministic and self._sample_perturbations:
            sample_config = self._sample_perturbations[index]
            if not sample_config['apply']:
                return text, perturbation_info
            
            perturbation_type = sample_config['type']
            severity = sample_config['severity']
            self._perturber.set_seed(sample_config['seed'])
        else:
            if self._rng.random() >= self.corruption_config.text_perturbation_prob:
                return text, perturbation_info
            
            if self.corruption_config.text_perturbation_types:
                perturbation_type = self._rng.choice(self.corruption_config.text_perturbation_types)
            else:
                available = self._get_available_perturbations()
                perturbation_type = self._rng.choice(available)
            
            min_sev, max_sev = self.corruption_config.text_severity_range
            severity = self._rng.randint(min_sev, max_sev + 1)
        
        # 处理列表形式的caption (FashionIQ)
        if isinstance(text, list):
            perturbed_text = [self._perturber.apply(t, perturbation_type, severity) for t in text]
        else:
            perturbed_text = self._perturber.apply(text, perturbation_type, severity)
        
        perturbation_info = {
            'applied': True,
            'type': perturbation_type,
            'severity': severity,
        }
        
        return perturbed_text, perturbation_info
    
    def __getitem__(self, index: int):
        """
        获取数据样本
        
        Returns:
            mode='relative', split='train':
                (ref_img, target_img, captions) 或 (ref_img, target_img, captions, corruption_info)
            
            mode='relative', split='val':
                (ref_name, target_name, captions)
            
            mode='classic':
                (image_name, image)
        """
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                
                if self.split == 'train':
                    # 加载参考图像
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = PIL.Image.open(reference_image_path).convert('RGB')
                    
                    # ★ 应用图像损坏
                    reference_image, corruption_info = self._apply_corruption(reference_image, index)
                    
                    # ★ 应用文本扰动
                    image_captions, text_perturbation_info = self._apply_text_perturbation(image_captions, index)
                    
                    # 预处理
                    reference_image = self.preprocess(reference_image)
                    
                    # 加载目标图像 (不损坏)
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert('RGB'))
                    
                    if self.corruption_config.return_corruption_info:
                        full_corruption_info = {
                            'image': corruption_info,
                            'text': text_perturbation_info,
                        }
                        return reference_image, target_image, image_captions, full_corruption_info
                    return reference_image, target_image, image_captions
                
                elif self.split == 'val':
                    # 验证集：也应用文本扰动
                    image_captions, _ = self._apply_text_perturbation(image_captions, index)
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions
                
                elif self.split == 'test':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = PIL.Image.open(reference_image_path).convert('RGB')
                    
                    # 测试时也可以应用损坏
                    reference_image, corruption_info = self._apply_corruption(reference_image, index)
                    reference_image = self.preprocess(reference_image)
                    
                    # ★ 应用文本扰动
                    image_captions, text_perturbation_info = self._apply_text_perturbation(image_captions, index)
                    
                    if self.corruption_config.return_corruption_info:
                        full_corruption_info = {
                            'image': corruption_info,
                            'text': text_perturbation_info,
                        }
                        return reference_name, reference_image, image_captions, full_corruption_info
                    return reference_name, reference_image, image_captions
            
            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path).convert('RGB'))
                return image_name, image
        
        except Exception as e:
            print(f"Exception at index {index}: {e}")
            return None
    
    def __len__(self) -> int:
        if self.mode == 'relative':
            return len(self.triplets)
        return len(self.image_names)


class CorruptedGalleryDataset(Dataset):
    """
    带损坏的 Gallery 数据集 (用于评估时对目标图像库应用损坏)
    
    通常评估时目标图像保持干净，但这个类允许测试当gallery也被损坏时的性能
    """
    
    def __init__(
        self,
        dataset_name: str,  # 'CIRR' or 'FashionIQ'
        split: str,
        preprocess: callable,
        corruption_config: Optional[CorruptionConfig] = None,
        dress_types: Optional[List[str]] = None,  # FashionIQ only
    ):
        """
        初始化带损坏的 Gallery 数据集
        
        Args:
            dataset_name: 'CIRR' 或 'FashionIQ'
            split: 数据集划分
            preprocess: 预处理函数
            corruption_config: 损坏配置
            dress_types: FashionIQ 的服装类型
        """
        self.dataset_name = dataset_name
        self.split = split
        self.preprocess = preprocess
        self.corruption_config = corruption_config or CorruptionConfig(image_corruption_prob=0.0)
        
        if dataset_name == 'CIRR':
            with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
                self.name_to_relpath = json.load(f)
            self.image_names = list(self.name_to_relpath.keys())
            self.image_dir = base_path / 'cirr_dataset'
        
        elif dataset_name == 'FashionIQ':
            dress_types = dress_types or ['dress', 'shirt', 'toptee']
            self.image_names = []
            for dress_type in dress_types:
                with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                    self.image_names.extend(json.load(f))
            self.image_dir = base_path / 'fashionIQ_dataset' / 'images'
            self.name_to_relpath = None
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # 初始化损坏器
        self._corruptor = ImageCorruptor(seed=self.corruption_config.seed)
        self._rng = np.random.RandomState(self.corruption_config.seed)
        
        print(f"CorruptedGallery for {dataset_name} {split} initialized")
        print(f"  Total images: {len(self.image_names)}")
    
    def _apply_corruption(self, image: PIL.Image.Image, index: int) -> PIL.Image.Image:
        """应用损坏"""
        if self.corruption_config.mode == 'clean':
            return image
        
        if self._rng.random() >= self.corruption_config.image_corruption_prob:
            return image
        
        if self.corruption_config.image_corruption_types:
            corruption_type = self._rng.choice(self.corruption_config.image_corruption_types)
        else:
            available = list(ImageCorruptor.CORRUPTION_REGISTRY.keys())
            corruption_type = self._rng.choice(available)
        
        min_sev, max_sev = self.corruption_config.image_severity_range
        severity = self._rng.randint(min_sev, max_sev + 1)
        
        corrupted_array = self._corruptor.apply(image, corruption_type, severity)
        return PIL.Image.fromarray(corrupted_array)
    
    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        
        if self.dataset_name == 'CIRR':
            image_path = self.image_dir / self.name_to_relpath[image_name]
        else:
            image_path = self.image_dir / f"{image_name}.png"
        
        image = PIL.Image.open(image_path).convert('RGB')
        image = self._apply_corruption(image, index)
        image = self.preprocess(image)
        
        return image_name, image
    
    def __len__(self) -> int:
        return len(self.image_names)


