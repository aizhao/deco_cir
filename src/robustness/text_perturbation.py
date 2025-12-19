"""
CIR 文本扰动模块 (基于 nlpaug)
================================
使用 nlpaug 库实现字符级、词级、语义级文本扰动

扰动分类体系:
├── Character-Level (字符级) - 模拟打字错误
│   ├── keyboard_typo, ocr_error, char_random (insert/swap/delete)
├── Word-Level (词级) - 词汇变换
│   ├── synonym_replace, antonym_replace, word_random (insert/swap/delete), spelling_error
├── Semantic-Level (语义级) - 语义相关变换
│   ├── contextual_word (BERT-based), back_translation
└── Format-Level (格式级) - 格式变化
    ├── case_change, split_merge

依赖:
    pip install nlpaug

Usage:
    perturber = TextPerturber(seed=42)
    perturbed = perturber.apply(text, 'keyboard_typo', severity=3)
"""

import numpy as np
import random
from enum import Enum
from typing import Union, Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings

# 尝试导入 nlpaug
try:
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    HAS_NLPAUG = True
except ImportError:
    HAS_NLPAUG = False
    warnings.warn(
        "nlpaug not installed. Please install it with: pip install nlpaug\n"
        "Some features may not work without nlpaug."
    )


class PerturbationLevel(Enum):
    """扰动层级枚举"""
    CHARACTER = "character"   # 字符级
    WORD = "word"             # 词级
    SEMANTIC = "semantic"     # 语义级
    FORMAT = "format"         # 格式级


class PerturbationCategory(Enum):
    """扰动类别枚举"""
    TYPO = "typo"             # 打字错误
    LEXICAL = "lexical"       # 词汇变换
    SYNTACTIC = "syntactic"   # 句法变换
    STYLISTIC = "stylistic"   # 风格变换


@dataclass
class PerturbationInfo:
    """扰动信息"""
    name: str
    level: PerturbationLevel
    category: PerturbationCategory
    description: str


class TextPerturber:
    """
    CIR文本扰动器 (基于 nlpaug)
    
    支持多种扰动类型，每种支持5级严重度(1-5)
    
    Attributes:
        seed: 随机种子，用于可复现性
        
    Example:
        >>> perturber = TextPerturber(seed=42)
        >>> # 应用单一扰动
        >>> perturbed = perturber.apply(text, 'keyboard_typo', severity=3)
        >>> # 随机应用扰动
        >>> perturbed, name, severity = perturber.apply_random(text, level='character')
    """
    
    # ============== 扰动类型注册表 ==============
    PERTURBATION_REGISTRY: Dict[str, PerturbationInfo] = {
        # 字符级 - 打字错误
        'keyboard_typo': PerturbationInfo('keyboard_typo', PerturbationLevel.CHARACTER, PerturbationCategory.TYPO, '键盘距离替换 - 相邻键误触'),
        'ocr_error': PerturbationInfo('ocr_error', PerturbationLevel.CHARACTER, PerturbationCategory.TYPO, 'OCR错误 - 形近字替换'),
        'char_insert': PerturbationInfo('char_insert', PerturbationLevel.CHARACTER, PerturbationCategory.TYPO, '字符插入 - 随机插入字符'),
        'char_delete': PerturbationInfo('char_delete', PerturbationLevel.CHARACTER, PerturbationCategory.TYPO, '字符删除 - 随机删除字符'),
        'char_swap': PerturbationInfo('char_swap', PerturbationLevel.CHARACTER, PerturbationCategory.TYPO, '字符交换 - 相邻字符互换'),
        'char_substitute': PerturbationInfo('char_substitute', PerturbationLevel.CHARACTER, PerturbationCategory.TYPO, '字符替换 - 随机替换字符'),
        
        # 词级 - 词汇变换
        'synonym_replace': PerturbationInfo('synonym_replace', PerturbationLevel.WORD, PerturbationCategory.LEXICAL, '同义词替换 - WordNet同义词'),
        'antonym_replace': PerturbationInfo('antonym_replace', PerturbationLevel.WORD, PerturbationCategory.LEXICAL, '反义词替换 - WordNet反义词'),
        'word_insert': PerturbationInfo('word_insert', PerturbationLevel.WORD, PerturbationCategory.LEXICAL, '词插入 - 随机插入词'),
        'word_delete': PerturbationInfo('word_delete', PerturbationLevel.WORD, PerturbationCategory.LEXICAL, '词删除 - 随机删除词'),
        'word_swap': PerturbationInfo('word_swap', PerturbationLevel.WORD, PerturbationCategory.SYNTACTIC, '词交换 - 词序打乱'),
        'word_split': PerturbationInfo('word_split', PerturbationLevel.WORD, PerturbationCategory.TYPO, '词分割 - 单词拆分'),
        'spelling_error': PerturbationInfo('spelling_error', PerturbationLevel.WORD, PerturbationCategory.TYPO, '拼写错误 - 常见拼写错误'),
        
        # 语义级 - 高级变换
        'contextual_word': PerturbationInfo('contextual_word', PerturbationLevel.SEMANTIC, PerturbationCategory.LEXICAL, '上下文替换 - BERT上下文感知'),
        'reserved_word': PerturbationInfo('reserved_word', PerturbationLevel.SEMANTIC, PerturbationCategory.LEXICAL, '保留词增强 - 保留关键词'),
        
        # 格式级 - 风格变换
        'case_change': PerturbationInfo('case_change', PerturbationLevel.FORMAT, PerturbationCategory.STYLISTIC, '大小写变换 - 随机改变大小写'),
    }
    
    def __init__(self, seed: int = 42):
        """
        初始化文本扰动器
        
        Args:
            seed: 随机种子
        """
        if not HAS_NLPAUG:
            raise ImportError(
                "nlpaug is required for TextPerturber. "
                "Please install it with: pip install nlpaug"
            )
        
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # 缓存增强器以提高性能
        self._augmenters = {}
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        random.seed(seed)
        # 清除缓存的增强器（它们使用了旧的种子）
        self._augmenters.clear()
    
    def _get_aug_fraction(self, severity: int) -> float:
        """
        根据严重度获取增强比例
        
        Args:
            severity: 严重程度 1-5
            
        Returns:
            增强比例 (0.0-1.0)
        """
        # 严重度对应的增强比例
        fraction_map = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
        return fraction_map[severity]
    
    def _get_augmenter(self, perturbation_type: str, severity: int):
        """
        获取或创建增强器
        
        Args:
            perturbation_type: 扰动类型
            severity: 严重程度
            
        Returns:
            nlpaug 增强器实例
        """
        cache_key = f"{perturbation_type}_{severity}"
        
        if cache_key in self._augmenters:
            return self._augmenters[cache_key]
        
        aug_fraction = self._get_aug_fraction(severity)
        aug = None
        
        # 字符级增强器
        if perturbation_type == 'keyboard_typo':
            aug = nac.KeyboardAug(
                aug_char_p=aug_fraction,
                aug_word_p=aug_fraction,
                include_special_char=False,
                include_numeric=False,
            )
        
        elif perturbation_type == 'ocr_error':
            aug = nac.OcrAug(
                aug_char_p=aug_fraction,
                aug_word_p=aug_fraction,
            )
        
        elif perturbation_type == 'char_insert':
            aug = nac.RandomCharAug(
                action='insert',
                aug_char_p=aug_fraction,
                aug_word_p=aug_fraction,
            )
        
        elif perturbation_type == 'char_delete':
            aug = nac.RandomCharAug(
                action='delete',
                aug_char_p=aug_fraction,
                aug_word_p=aug_fraction,
            )
        
        elif perturbation_type == 'char_swap':
            aug = nac.RandomCharAug(
                action='swap',
                aug_char_p=aug_fraction,
                aug_word_p=aug_fraction,
            )
        
        elif perturbation_type == 'char_substitute':
            aug = nac.RandomCharAug(
                action='substitute',
                aug_char_p=aug_fraction,
                aug_word_p=aug_fraction,
            )
        
        # 词级增强器
        elif perturbation_type == 'synonym_replace':
            try:
                aug = naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=aug_fraction,
                )
            except Exception as e:
                warnings.warn(f"Failed to create SynonymAug: {e}. Using fallback.")
                aug = naw.RandomWordAug(action='substitute', aug_p=aug_fraction)
        
        elif perturbation_type == 'antonym_replace':
            try:
                aug = naw.AntonymAug(
                    aug_p=aug_fraction,
                )
            except Exception as e:
                warnings.warn(f"Failed to create AntonymAug: {e}. Using fallback.")
                aug = naw.RandomWordAug(action='substitute', aug_p=aug_fraction)
        
        elif perturbation_type == 'word_insert':
            aug = naw.RandomWordAug(
                action='insert',
                aug_p=aug_fraction,
            )
        
        elif perturbation_type == 'word_delete':
            aug = naw.RandomWordAug(
                action='delete',
                aug_p=aug_fraction,
            )
        
        elif perturbation_type == 'word_swap':
            aug = naw.RandomWordAug(
                action='swap',
                aug_p=aug_fraction,
            )
        
        elif perturbation_type == 'word_split':
            aug = naw.SplitAug(
                aug_p=aug_fraction,
            )
        
        elif perturbation_type == 'spelling_error':
            try:
                aug = naw.SpellingAug(
                    aug_p=aug_fraction,
                )
            except Exception as e:
                warnings.warn(f"Failed to create SpellingAug: {e}. Using KeyboardAug as fallback.")
                aug = nac.KeyboardAug(aug_char_p=aug_fraction, aug_word_p=aug_fraction)
        
        # 语义级增强器
        elif perturbation_type == 'contextual_word':
            try:
                # 使用 distilbert 作为轻量级选项
                aug = naw.ContextualWordEmbsAug(
                    model_path='distilbert-base-uncased',
                    action='substitute',
                    aug_p=aug_fraction,
                    device='cuda',  # 使用 GPU
                )
            except Exception as e:
                warnings.warn(f"Failed to create ContextualWordEmbsAug: {e}. Using SynonymAug as fallback.")
                try:
                    aug = naw.SynonymAug(aug_src='wordnet', aug_p=aug_fraction)
                except:
                    aug = naw.RandomWordAug(action='substitute', aug_p=aug_fraction)
        
        elif perturbation_type == 'reserved_word':
            # 保留特定词不被修改的增强
            aug = naw.ReservedAug(
                reserved_tokens=['color', 'red', 'blue', 'green', 'black', 'white',
                                'size', 'big', 'small', 'long', 'short',
                                'same', 'similar', 'different'],
            )
        
        # 格式级增强器（自定义实现）
        elif perturbation_type == 'case_change':
            aug = CaseChangeAug(severity=severity)
        
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        self._augmenters[cache_key] = aug
        return aug
    
    def apply(self, text: str, perturbation_type: str, severity: int = 3) -> str:
        """
        应用指定类型的扰动
        
        Args:
            text: 输入文本
            perturbation_type: 扰动类型名称
            severity: 严重程度 1-5
            
        Returns:
            扰动后的文本
        """
        if perturbation_type not in self.PERTURBATION_REGISTRY:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}. "
                           f"Available: {list(self.PERTURBATION_REGISTRY.keys())}")
        
        if not 1 <= severity <= 5:
            raise ValueError(f"Severity must be in [1, 5], got {severity}")
        
        if not text or not text.strip():
            return text
        
        try:
            aug = self._get_augmenter(perturbation_type, severity)
            
            # reserved_word 是特殊的，它返回的是处理器而不是增强器
            if perturbation_type == 'reserved_word':
                return text  # 暂时直接返回原文
            
            result = aug.augment(text)
            
            # nlpaug 可能返回列表或字符串
            if isinstance(result, list):
                result = result[0] if result else text
            
            return result if result else text
            
        except Exception as e:
            warnings.warn(f"Augmentation failed for {perturbation_type}: {e}. Returning original text.")
            return text
    
    def apply_random(
        self,
        text: str,
        level: Optional[str] = None,
        category: Optional[str] = None,
        severity: Optional[int] = None
    ) -> Tuple[str, str, int]:
        """
        随机应用一种扰动
        
        Args:
            text: 输入文本
            level: 限制扰动层级 ('character', 'word', 'semantic', 'format')
            category: 限制扰动类别 ('typo', 'lexical', 'syntactic', 'stylistic')
            severity: 指定严重度，None则随机
            
        Returns:
            (perturbed_text, perturbation_type, severity)
        """
        # 筛选可用的扰动类型
        available = list(self.PERTURBATION_REGISTRY.keys())
        
        # 排除可能较慢或需要额外资源的扰动
        slow_augs = ['contextual_word', 'reserved_word']
        available = [a for a in available if a not in slow_augs]
        
        if level:
            level_enum = PerturbationLevel(level)
            available = [k for k, v in self.PERTURBATION_REGISTRY.items() 
                        if v.level == level_enum and k in available]
        
        if category:
            cat_enum = PerturbationCategory(category)
            available = [k for k in available 
                        if self.PERTURBATION_REGISTRY[k].category == cat_enum]
        
        if not available:
            raise ValueError(f"No perturbation types match the given constraints: level={level}, category={category}")
        
        # 随机选择
        perturbation_type = self._rng.choice(available)
        if severity is None:
            severity = self._rng.randint(1, 6)
        
        perturbed = self.apply(text, perturbation_type, severity)
        return perturbed, perturbation_type, severity
    
    def apply_composite(
        self,
        text: str,
        perturbations: List[Tuple[str, int]]
    ) -> str:
        """
        应用复合扰动 (多种扰动叠加)
        
        Args:
            text: 输入文本
            perturbations: 扰动列表 [(type, severity), ...]
            
        Returns:
            扰动后的文本
        """
        result = text
        for perturbation_type, severity in perturbations:
            result = self.apply(result, perturbation_type, severity)
        return result
    
    @classmethod
    def get_all_types(cls) -> Dict[str, List[str]]:
        """
        获取所有扰动类型，按层级分组
        
        Returns:
            {level: [perturbation_types]}
        """
        result = {}
        for name, info in cls.PERTURBATION_REGISTRY.items():
            level = info.level.value
            if level not in result:
                result[level] = []
            result[level].append(name)
        return result
    
    @classmethod
    def get_types_by_category(cls) -> Dict[str, List[str]]:
        """
        获取所有扰动类型，按类别分组
        
        Returns:
            {category: [perturbation_types]}
        """
        result = {}
        for name, info in cls.PERTURBATION_REGISTRY.items():
            cat = info.category.value
            if cat not in result:
                result[cat] = []
            result[cat].append(name)
        return result
    
    @classmethod
    def get_perturbation_info(cls, perturbation_type: str) -> PerturbationInfo:
        """获取扰动类型的详细信息"""
        if perturbation_type not in cls.PERTURBATION_REGISTRY:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        return cls.PERTURBATION_REGISTRY[perturbation_type]


class CaseChangeAug:
    """
    大小写变换增强器 (自定义实现)
    
    nlpaug 没有内置的大小写变换，所以我们自定义实现
    """
    
    def __init__(self, severity: int = 3):
        self.severity = severity
    
    def augment(self, text: str) -> str:
        """应用大小写变换"""
        if not text:
            return text
        
        # 根据严重度选择模式
        mode_probs = {
            1: {'lower': 0.3, 'upper': 0.3, 'random': 0.2, 'none': 0.2},
            2: {'lower': 0.3, 'upper': 0.3, 'random': 0.3, 'none': 0.1},
            3: {'lower': 0.25, 'upper': 0.25, 'random': 0.4, 'none': 0.1},
            4: {'lower': 0.2, 'upper': 0.2, 'random': 0.55, 'none': 0.05},
            5: {'lower': 0.15, 'upper': 0.15, 'random': 0.7, 'none': 0.0},
        }
        
        probs = mode_probs[self.severity]
        mode = random.choices(
            list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]
        
        if mode == 'none':
            return text
        elif mode == 'lower':
            return text.lower()
        elif mode == 'upper':
            return text.upper()
        else:  # random
            return ''.join(
                c.upper() if random.random() < 0.5 else c.lower()
                for c in text
            )


# ============== 便捷函数 ==============

def perturb_text(
    text: str,
    perturbation_type: str,
    severity: int = 3,
    seed: int = 42
) -> str:
    """
    便捷函数：应用单一扰动
    
    Args:
        text: 输入文本
        perturbation_type: 扰动类型
        severity: 严重程度 1-5
        seed: 随机种子
        
    Returns:
        扰动后的文本
    """
    perturber = TextPerturber(seed=seed)
    return perturber.apply(text, perturbation_type, severity)


def list_perturbations() -> None:
    """打印所有可用的扰动类型"""
    print("=" * 60)
    print("Available Text Perturbations (nlpaug-based)")
    print("=" * 60)
    
    types_by_level = TextPerturber.get_all_types()
    for level in ['character', 'word', 'semantic', 'format']:
        print(f"\n{level.upper()} Level:")
        for name in types_by_level.get(level, []):
            info = TextPerturber.PERTURBATION_REGISTRY[name]
            print(f"  - {name}: {info.description}")


if __name__ == '__main__':
    if not HAS_NLPAUG:
        print("ERROR: nlpaug is not installed.")
        print("Please install it with: pip install nlpaug")
        exit(1)
    
    list_perturbations()
    
    print("\n" + "=" * 60)
    print("Demo:")
    print("=" * 60)
    
    perturber = TextPerturber(seed=42)
    
    test_texts = [
        "make it more colorful and add some patterns",
        "the dress should be shorter with a red color",
        "change the background to a beach scene",
    ]
    
    # 测试基础扰动（不需要额外资源）
    basic_perturbations = ['keyboard_typo', 'char_swap', 'word_delete', 'case_change']
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        
        for pert_type in basic_perturbations:
            try:
                result = perturber.apply(text, pert_type, severity=3)
                print(f"  {pert_type}: {result}")
            except Exception as e:
                print(f"  {pert_type}: ERROR - {e}")
