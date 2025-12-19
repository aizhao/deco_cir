"""
CIR 损坏配置模块
================
定义损坏配置类，支持从YAML/JSON加载和预设配置

Usage:
    # 基本使用
    config = CorruptionConfig(image_corruption_prob=0.5, image_severity_range=(2, 4))
    
    # 从预设加载
    config = CorruptionConfig.from_preset('moderate')
    
    # 从文件加载
    config = CorruptionConfig.from_yaml('config.yaml')
    
    # 消融实验专用
    config = CorruptionConfig.ablation('gaussian_noise', severity=3)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any, Literal
import json
import yaml
from pathlib import Path


@dataclass
class CorruptionConfig:
    """
    损坏配置类
    
    Attributes:
        # 图像损坏配置
        image_corruption_prob: 应用图像损坏的概率
        image_corruption_types: 限制使用的损坏类型，None表示所有
        image_severity_range: 严重度范围 (min, max)
        image_corruption_level: 限制损坏层级 ('pixel', 'region', 'global', None)
        image_corruption_category: 限制损坏类别 ('noise', 'blur', 'digital', 'weather', 'color', 'geometric', None)
        
        # 文本扰动配置
        text_perturbation_prob: 应用文本扰动的概率
        text_perturbation_types: 限制使用的扰动类型，None表示所有
        text_severity_range: 严重度范围 (min, max)
        text_perturbation_level: 限制扰动层级 ('character', 'word', 'semantic', 'format', None)
        text_perturbation_category: 限制扰动类别 ('typo', 'lexical', 'syntactic', 'stylistic', None)
        
        # 模式配置
        mode: 损坏模式 ('online', 'offline', 'clean')
        offline_config_path: 离线配置文件路径 (mode='offline'时使用)
        
        # 随机性配置
        seed: 随机种子
        deterministic: 是否确定性模式 (每个样本的损坏固定)
        
        # 输出配置
        return_corruption_info: 是否返回损坏信息
    """
    
    # 图像损坏配置
    image_corruption_prob: float = 0.5
    image_corruption_types: Optional[List[str]] = None
    image_severity_range: Tuple[int, int] = (1, 5)
    image_corruption_level: Optional[str] = None  # 'pixel', 'region', 'global'
    image_corruption_category: Optional[str] = None  # 'noise', 'blur', 'digital', 'weather', 'color', 'geometric'
    
    # 文本扰动配置
    text_perturbation_prob: float = 0.0  # 默认不应用文本扰动
    text_perturbation_types: Optional[List[str]] = None
    text_severity_range: Tuple[int, int] = (1, 5)
    text_perturbation_level: Optional[str] = None  # 'character', 'word', 'semantic', 'format'
    text_perturbation_category: Optional[str] = None  # 'typo', 'lexical', 'syntactic', 'stylistic'
    
    # 模式配置
    mode: Literal['online', 'offline', 'clean'] = 'online'
    offline_config_path: Optional[str] = None
    
    # 随机性配置
    seed: int = 42
    deterministic: bool = False  # 如果True，同一个index总是产生相同的损坏
    
    # 输出配置
    return_corruption_info: bool = False
    
    def __post_init__(self):
        """验证配置"""
        # 验证图像损坏概率范围
        if not 0.0 <= self.image_corruption_prob <= 1.0:
            raise ValueError(f"image_corruption_prob must be in [0, 1], got {self.image_corruption_prob}")
        
        # 验证图像严重度范围
        min_sev, max_sev = self.image_severity_range
        if not (1 <= min_sev <= max_sev <= 5):
            raise ValueError(f"image_severity_range must be in [1, 5] with min <= max, got {self.image_severity_range}")
        
        # 验证文本扰动概率范围
        if not 0.0 <= self.text_perturbation_prob <= 1.0:
            raise ValueError(f"text_perturbation_prob must be in [0, 1], got {self.text_perturbation_prob}")
        
        # 验证文本严重度范围
        text_min_sev, text_max_sev = self.text_severity_range
        if not (1 <= text_min_sev <= text_max_sev <= 5):
            raise ValueError(f"text_severity_range must be in [1, 5] with min <= max, got {self.text_severity_range}")
        
        # 验证模式
        if self.mode == 'offline' and self.offline_config_path is None:
            raise ValueError("offline_config_path must be provided when mode='offline'")
        
        # 验证图像损坏层级
        valid_image_levels = ['pixel', 'region', 'global', None]
        if self.image_corruption_level not in valid_image_levels:
            raise ValueError(f"image_corruption_level must be in {valid_image_levels}, got {self.image_corruption_level}")
        
        # 验证图像损坏类别
        valid_image_categories = ['noise', 'blur', 'digital', 'weather', 'color', 'geometric', None]
        if self.image_corruption_category not in valid_image_categories:
            raise ValueError(f"image_corruption_category must be in {valid_image_categories}, got {self.image_corruption_category}")
        
        # 验证文本扰动层级
        valid_text_levels = ['character', 'word', 'semantic', 'format', None]
        if self.text_perturbation_level not in valid_text_levels:
            raise ValueError(f"text_perturbation_level must be in {valid_text_levels}, got {self.text_perturbation_level}")
        
        # 验证文本扰动类别
        valid_text_categories = ['typo', 'lexical', 'syntactic', 'stylistic', None]
        if self.text_perturbation_category not in valid_text_categories:
            raise ValueError(f"text_perturbation_category must be in {valid_text_categories}, got {self.text_perturbation_category}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """保存为JSON文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, path: str) -> None:
        """保存为YAML文件"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CorruptionConfig':
        """从字典创建"""
        # 处理 severity_range 可能是 list 的情况
        if 'image_severity_range' in d and isinstance(d['image_severity_range'], list):
            d['image_severity_range'] = tuple(d['image_severity_range'])
        if 'text_severity_range' in d and isinstance(d['text_severity_range'], list):
            d['text_severity_range'] = tuple(d['text_severity_range'])
        return cls(**d)
    
    @classmethod
    def from_json(cls, path: str) -> 'CorruptionConfig':
        """从JSON文件加载"""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'CorruptionConfig':
        """从YAML文件加载"""
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    @classmethod
    def from_preset(cls, preset: str) -> 'CorruptionConfig':
        """
        从预设配置创建
        
        Available presets:
            基础预设:
            - clean: 无损坏
            - light: 轻度损坏 (验证基础鲁棒性)
            - moderate: 中度损坏 (典型测试场景)
            - severe: 重度损坏 (极端测试)
            - extreme: 极端损坏
            
            图像按层级:
            - pixel_only: 仅像素级损坏
            - region_only: 仅区域级损坏
            - global_only: 仅全局级损坏
            
            图像按类别:
            - noise_only: 仅噪声
            - blur_only: 仅模糊
            - weather_only: 仅天气
            - color_only: 仅色彩
            - geometric_only: 仅几何
            
            文本扰动预设:
            - text_light: 轻度文本扰动
            - text_moderate: 中度文本扰动
            - text_severe: 重度文本扰动
            - text_typo_only: 仅打字错误
            - text_lexical_only: 仅词汇变换
            
            双模态预设:
            - bimodal_light: 轻度双模态扰动
            - bimodal_moderate: 中度双模态扰动
            - bimodal_severe: 重度双模态扰动
        """
        presets = {
            # 基础预设 (仅图像)
            'clean': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=0.0,
                mode='clean',
            ),
            'light': cls(
                image_corruption_prob=0.3,
                image_severity_range=(1, 2),
            ),
            'moderate': cls(
                image_corruption_prob=0.5,
                image_severity_range=(2, 4),
            ),
            'severe': cls(
                image_corruption_prob=0.7,
                image_severity_range=(3, 5),
            ),
            'extreme': cls(
                image_corruption_prob=0.9,
                image_severity_range=(4, 5),
            ),
            
            # 图像按层级
            'pixel_only': cls(
                image_corruption_prob=1.0,
                image_corruption_level='pixel',
                image_severity_range=(1, 5),
            ),
            'region_only': cls(
                image_corruption_prob=1.0,
                image_corruption_level='region',
                image_severity_range=(1, 5),
            ),
            'global_only': cls(
                image_corruption_prob=1.0,
                image_corruption_level='global',
                image_severity_range=(1, 5),
            ),
            
            # 图像按类别
            'noise_only': cls(
                image_corruption_prob=1.0,
                image_corruption_category='noise',
                image_severity_range=(1, 5),
            ),
            'blur_only': cls(
                image_corruption_prob=1.0,
                image_corruption_category='blur',
                image_severity_range=(1, 5),
            ),
            'weather_only': cls(
                image_corruption_prob=1.0,
                image_corruption_category='weather',
                image_severity_range=(1, 5),
            ),
            'color_only': cls(
                image_corruption_prob=1.0,
                image_corruption_category='color',
                image_severity_range=(1, 5),
            ),
            'geometric_only': cls(
                image_corruption_prob=1.0,
                image_corruption_category='geometric',
                image_severity_range=(1, 5),
            ),
            'digital_only': cls(
                image_corruption_prob=1.0,
                image_corruption_category='digital',
                image_severity_range=(1, 5),
            ),
            
            # 文本扰动预设
            'text_light': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=0.3,
                text_severity_range=(1, 2),
            ),
            'text_moderate': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=0.5,
                text_severity_range=(2, 4),
            ),
            'text_severe': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=0.7,
                text_severity_range=(3, 5),
            ),
            'text_typo_only': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=1.0,
                text_perturbation_category='typo',
                text_severity_range=(1, 5),
            ),
            'text_lexical_only': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=1.0,
                text_perturbation_category='lexical',
                text_severity_range=(1, 5),
            ),
            'text_character_only': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=1.0,
                text_perturbation_level='character',
                text_severity_range=(1, 5),
            ),
            'text_word_only': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=1.0,
                text_perturbation_level='word',
                text_severity_range=(1, 5),
            ),
            'text_semantic_only': cls(
                image_corruption_prob=0.0,
                text_perturbation_prob=1.0,
                text_perturbation_level='semantic',
                text_severity_range=(1, 5),
            ),
            
            # 双模态预设
            'bimodal_light': cls(
                image_corruption_prob=0.3,
                image_severity_range=(1, 2),
                text_perturbation_prob=0.3,
                text_severity_range=(1, 2),
            ),
            'bimodal_moderate': cls(
                image_corruption_prob=0.5,
                image_severity_range=(2, 4),
                text_perturbation_prob=0.5,
                text_severity_range=(2, 4),
            ),
            'bimodal_severe': cls(
                image_corruption_prob=0.7,
                image_severity_range=(3, 5),
                text_perturbation_prob=0.7,
                text_severity_range=(3, 5),
            ),
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]
    
    @classmethod
    def ablation(
        cls, 
        corruption_type: str, 
        severity: int,
        deterministic: bool = True
    ) -> 'CorruptionConfig':
        """
        创建图像消融实验专用配置
        
        Args:
            corruption_type: 图像损坏类型
            severity: 严重程度 1-5
            deterministic: 是否确定性
            
        Returns:
            针对单一损坏类型和严重度的配置
        """
        return cls(
            image_corruption_prob=1.0,
            image_corruption_types=[corruption_type],
            image_severity_range=(severity, severity),
            deterministic=deterministic,
            return_corruption_info=True,
        )
    
    @classmethod
    def ablation_text(
        cls,
        perturbation_type: str,
        severity: int,
        deterministic: bool = True
    ) -> 'CorruptionConfig':
        """
        创建文本消融实验专用配置
        
        Args:
            perturbation_type: 文本扰动类型
            severity: 严重程度 1-5
            deterministic: 是否确定性
            
        Returns:
            针对单一扰动类型和严重度的配置
        """
        return cls(
            image_corruption_prob=0.0,
            text_perturbation_prob=1.0,
            text_perturbation_types=[perturbation_type],
            text_severity_range=(severity, severity),
            deterministic=deterministic,
            return_corruption_info=True,
        )
    
    @classmethod
    def ablation_bimodal(
        cls,
        image_corruption_type: str,
        text_perturbation_type: str,
        image_severity: int,
        text_severity: int,
        deterministic: bool = True
    ) -> 'CorruptionConfig':
        """
        创建双模态消融实验专用配置
        
        Args:
            image_corruption_type: 图像损坏类型
            text_perturbation_type: 文本扰动类型
            image_severity: 图像严重程度 1-5
            text_severity: 文本严重程度 1-5
            deterministic: 是否确定性
            
        Returns:
            双模态损坏配置
        """
        return cls(
            image_corruption_prob=1.0,
            image_corruption_types=[image_corruption_type],
            image_severity_range=(image_severity, image_severity),
            text_perturbation_prob=1.0,
            text_perturbation_types=[text_perturbation_type],
            text_severity_range=(text_severity, text_severity),
            deterministic=deterministic,
            return_corruption_info=True,
        )
    
    @classmethod
    def ablation_level(
        cls,
        level: str,
        severity: int,
        deterministic: bool = True
    ) -> 'CorruptionConfig':
        """
        创建按层级的消融实验配置
        
        Args:
            level: 'pixel', 'region', 'global'
            severity: 严重程度
            deterministic: 是否确定性
        """
        return cls(
            image_corruption_prob=1.0,
            image_corruption_level=level,
            image_severity_range=(severity, severity),
            deterministic=deterministic,
            return_corruption_info=True,
        )
    
    @classmethod
    def ablation_category(
        cls,
        category: str,
        severity: int,
        deterministic: bool = True
    ) -> 'CorruptionConfig':
        """
        创建按类别的消融实验配置
        
        Args:
            category: 'noise', 'blur', 'digital', 'weather', 'color', 'geometric'
            severity: 严重程度
            deterministic: 是否确定性
        """
        return cls(
            image_corruption_prob=1.0,
            image_corruption_category=category,
            image_severity_range=(severity, severity),
            deterministic=deterministic,
            return_corruption_info=True,
        )
    
    def get_description(self) -> str:
        """获取配置的文字描述"""
        if self.mode == 'clean':
            return "Clean (no corruption)"
        
        parts = []
        
        # 图像损坏描述
        if self.image_corruption_prob > 0:
            img_parts = [f"img_prob={self.image_corruption_prob}"]
            img_parts.append(f"img_sev={self.image_severity_range}")
            if self.image_corruption_types:
                img_parts.append(f"img_types={self.image_corruption_types}")
            if self.image_corruption_level:
                img_parts.append(f"img_level={self.image_corruption_level}")
            if self.image_corruption_category:
                img_parts.append(f"img_cat={self.image_corruption_category}")
            parts.append("Image[" + ", ".join(img_parts) + "]")
        
        # 文本扰动描述
        if self.text_perturbation_prob > 0:
            txt_parts = [f"txt_prob={self.text_perturbation_prob}"]
            txt_parts.append(f"txt_sev={self.text_severity_range}")
            if self.text_perturbation_types:
                txt_parts.append(f"txt_types={self.text_perturbation_types}")
            if self.text_perturbation_level:
                txt_parts.append(f"txt_level={self.text_perturbation_level}")
            if self.text_perturbation_category:
                txt_parts.append(f"txt_cat={self.text_perturbation_category}")
            parts.append("Text[" + ", ".join(txt_parts) + "]")
        
        return " | ".join(parts) if parts else "No corruption configured"
    
    def __str__(self) -> str:
        return f"CorruptionConfig({self.get_description()})"
    
    def __repr__(self) -> str:
        return self.__str__()


def generate_ablation_configs(
    corruption_types: Optional[List[str]] = None,
    severities: List[int] = [1, 2, 3, 4, 5],
    output_dir: Optional[str] = None,
) -> List[CorruptionConfig]:
    """
    生成图像消融实验的所有配置
    
    Args:
        corruption_types: 损坏类型列表，None表示所有
        severities: 严重度列表
        output_dir: 保存配置文件的目录，None表示不保存
        
    Returns:
        配置列表
    """
    from .image_corruption import ImageCorruptor
    
    if corruption_types is None:
        corruption_types = list(ImageCorruptor.CORRUPTION_REGISTRY.keys())
    
    configs = []
    
    for corruption_type in corruption_types:
        for severity in severities:
            config = CorruptionConfig.ablation(corruption_type, severity)
            configs.append(config)
            
            if output_dir:
                output_path = Path(output_dir) / f"image_{corruption_type}_s{severity}.yaml"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                config.to_yaml(str(output_path))
    
    return configs


def generate_text_ablation_configs(
    perturbation_types: Optional[List[str]] = None,
    severities: List[int] = [1, 2, 3, 4, 5],
    output_dir: Optional[str] = None,
) -> List[CorruptionConfig]:
    """
    生成文本消融实验的所有配置
    
    Args:
        perturbation_types: 扰动类型列表，None表示所有
        severities: 严重度列表
        output_dir: 保存配置文件的目录，None表示不保存
        
    Returns:
        配置列表
    """
    from .text_perturbation import TextPerturber
    
    if perturbation_types is None:
        perturbation_types = list(TextPerturber.PERTURBATION_REGISTRY.keys())
    
    configs = []
    
    for perturbation_type in perturbation_types:
        for severity in severities:
            config = CorruptionConfig.ablation_text(perturbation_type, severity)
            configs.append(config)
            
            if output_dir:
                output_path = Path(output_dir) / f"text_{perturbation_type}_s{severity}.yaml"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                config.to_yaml(str(output_path))
    
    return configs


if __name__ == '__main__':
    # 示例
    print("=== Preset Configs ===")
    for preset in ['clean', 'light', 'moderate', 'severe']:
        config = CorruptionConfig.from_preset(preset)
        print(f"{preset}: {config}")
    
    print("\n=== Ablation Config ===")
    config = CorruptionConfig.ablation('gaussian_noise', 3)
    print(config)
    print(f"Dict: {config.to_dict()}")


