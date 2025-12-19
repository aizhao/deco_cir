"""
图像损坏模块演示脚本
==================
展示如何使用 ImageCorruptor 和 CorruptedCIRRDataset

Usage:
    cd SPRC/src
    python -m robustness.demo_corruption

    # 指定图片路径
    python -m robustness.demo_corruption --image /path/to/image.jpg
    
    # 使用数据集图片
    python -m robustness.demo_corruption --dataset CIRR --num-images 3
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 项目根路径
BASE_PATH = Path(__file__).absolute().parents[2]


def load_sample_images(
    dataset: str = 'CIRR',
    split: str = 'val',
    num_images: int = 3,
    seed: int = 42
) -> List[Tuple[np.ndarray, str]]:
    """
    从数据集加载样本图片
    
    Args:
        dataset: 'CIRR' 或 'FashionIQ'
        split: 数据集划分
        num_images: 加载图片数量
        seed: 随机种子
        
    Returns:
        List of (image_array, image_name)
    """
    np.random.seed(seed)
    images = []
    
    if dataset == 'CIRR':
        # 尝试不同的split
        for try_split in [split, 'val', 'train', 'dev']:
            split_file = BASE_PATH / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{try_split}.json'
            if split_file.exists():
                with open(split_file, 'r') as f:
                    name_to_relpath = json.load(f)
                
                # 随机选择图片
                all_names = list(name_to_relpath.keys())
                selected_indices = np.random.choice(len(all_names), min(num_images, len(all_names)), replace=False)
                
                for idx in selected_indices:
                    name = all_names[idx]
                    image_path = BASE_PATH / 'cirr_dataset' / name_to_relpath[name]
                    if image_path.exists():
                        img = Image.open(image_path).convert('RGB')
                        # 调整大小以便显示
                        img = img.resize((224, 224), Image.BILINEAR)
                        images.append((np.array(img), name))
                
                if images:
                    print(f"Loaded {len(images)} images from CIRR {try_split} split")
                    break
    
    elif dataset == 'FashionIQ':
        for dress_type in ['dress', 'shirt', 'toptee']:
            split_file = BASE_PATH / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json'
            if split_file.exists():
                with open(split_file, 'r') as f:
                    image_names = json.load(f)
                
                # 随机选择
                selected_indices = np.random.choice(len(image_names), min(num_images, len(image_names)), replace=False)
                
                for idx in selected_indices:
                    name = image_names[idx]
                    image_path = BASE_PATH / 'fashionIQ_dataset' / 'images' / f"{name}.png"
                    if image_path.exists():
                        img = Image.open(image_path).convert('RGB')
                        img = img.resize((224, 224), Image.BILINEAR)
                        images.append((np.array(img), name))
                
                if len(images) >= num_images:
                    break
        
        if images:
            print(f"Loaded {len(images)} images from FashionIQ")
    
    return images[:num_images]


def load_image_from_path(image_path: str) -> Tuple[np.ndarray, str]:
    """从指定路径加载图片"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    return np.array(img), path.stem


def create_synthetic_image(h: int = 224, w: int = 224) -> np.ndarray:
    """创建合成测试图像 (彩色渐变)"""
    test_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            test_image[i, j] = [
                int(255 * i / h),           # R: 上到下渐变
                int(255 * j / w),           # G: 左到右渐变
                128                         # B: 固定
            ]
    return test_image


def get_test_images(args) -> List[Tuple[np.ndarray, str]]:
    """根据参数获取测试图片"""
    images = []
    
    # 优先使用指定的图片路径
    if args.image:
        try:
            img, name = load_image_from_path(args.image)
            images.append((img, name))
            print(f"Loaded image from: {args.image}")
        except Exception as e:
            print(f"Failed to load image: {e}")
    
    # 尝试从数据集加载
    if not images and args.dataset:
        images = load_sample_images(
            dataset=args.dataset,
            split=args.split,
            num_images=args.num_images,
            seed=args.seed
        )
    
    # 回退到合成图像
    if not images:
        print("No dataset images found. Using synthetic test image.")
        images = [(create_synthetic_image(), 'synthetic')]
    
    return images


def demo_single_corruptions(test_images: List[Tuple[np.ndarray, str]]):
    """演示单一损坏效果"""
    from robustness.image_corruption import ImageCorruptor, list_corruptions
    
    print("=" * 60)
    print("Demo: Single Corruption Effects")
    print("=" * 60)
    
    # 列出所有可用损坏
    list_corruptions()
    
    # 创建损坏器
    corruptor = ImageCorruptor(seed=42)
    
    # 选择一些代表性损坏进行演示
    demo_corruptions = [
        ('gaussian_noise', 3),
        ('motion_blur', 3),
        ('jpeg_compression', 3),
        ('fog', 3),
        ('brightness', 3),
        ('saturation', 3),
        ('hue_shift', 3),
    ]
    
    # 为每张测试图片生成演示
    for test_image, image_name in test_images:
        print(f"\nProcessing: {image_name}")
        
        # 创建可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # 原图
        axes[0].imshow(test_image)
        axes[0].set_title(f'Original\n({image_name})', fontsize=10)
        axes[0].axis('off')
        
        # 应用各种损坏
        for idx, (corruption_type, severity) in enumerate(demo_corruptions):
            corrupted = corruptor.apply(test_image, corruption_type, severity)
            axes[idx + 1].imshow(corrupted)
            axes[idx + 1].set_title(f'{corruption_type}\n(severity={severity})', fontsize=9)
            axes[idx + 1].axis('off')
        
        plt.suptitle('Single Corruption Effects Demo', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        output_dir = BASE_PATH / 'visualizations' / 'corruption_demo'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'single_corruptions_{image_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def demo_severity_levels(test_images: List[Tuple[np.ndarray, str]]):
    """演示不同严重度的效果"""
    from robustness.image_corruption import ImageCorruptor
    
    print("\n" + "=" * 60)
    print("Demo: Severity Levels")
    print("=" * 60)
    
    corruptor = ImageCorruptor(seed=42)
    
    # 选择一种损坏，展示不同严重度
    corruption_types = ['gaussian_noise', 'motion_blur', 'fog', 'saturation']
    
    for test_image, image_name in test_images:
        print(f"\nProcessing: {image_name}")
        
        fig, axes = plt.subplots(len(corruption_types), 6, figsize=(18, 12))
        
        for row, corruption_type in enumerate(corruption_types):
            # 原图
            axes[row, 0].imshow(test_image)
            axes[row, 0].set_title('Original' if row == 0 else '', fontsize=10)
            axes[row, 0].set_ylabel(corruption_type.replace('_', '\n'), fontsize=11)
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            
            # 各严重度
            for severity in range(1, 6):
                corrupted = corruptor.apply(test_image, corruption_type, severity)
                axes[row, severity].imshow(corrupted)
                if row == 0:
                    axes[row, severity].set_title(f'Severity {severity}', fontsize=10)
                axes[row, severity].axis('off')
        
        plt.suptitle(f'Severity Levels Demo - {image_name}', fontsize=14)
        plt.tight_layout()
        
        output_dir = BASE_PATH / 'visualizations' / 'corruption_demo'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'severity_levels_{image_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def demo_all_corruptions_grid(test_images: List[Tuple[np.ndarray, str]]):
    """生成所有损坏类型的网格图"""
    from robustness.image_corruption import ImageCorruptor
    
    print("\n" + "=" * 60)
    print("Demo: All Corruptions Grid")
    print("=" * 60)
    
    corruptor = ImageCorruptor(seed=42)
    
    # 获取所有损坏类型
    all_types = corruptor.get_all_types()
    all_corruptions = []
    for level in ['pixel', 'region', 'global']:
        all_corruptions.extend(all_types.get(level, []))
    
    for test_image, image_name in test_images:
        print(f"\nProcessing: {image_name}")
        
        # 调整图像大小以便在网格中显示
        h, w = test_image.shape[:2]
        if h > 128 or w > 128:
            img_small = np.array(Image.fromarray(test_image).resize((128, 128), Image.BILINEAR))
        else:
            img_small = test_image
        
        # 计算网格大小
        n_corruptions = len(all_corruptions) + 1  # +1 for original
        n_cols = 7
        n_rows = (n_corruptions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()
        
        # 原图
        axes[0].imshow(img_small)
        axes[0].set_title('Original', fontsize=8)
        axes[0].axis('off')
        
        # 所有损坏
        for idx, corruption_type in enumerate(all_corruptions):
            try:
                corrupted = corruptor.apply(img_small, corruption_type, severity=3)
                axes[idx + 1].imshow(corrupted)
                axes[idx + 1].set_title(corruption_type.replace('_', '\n'), fontsize=6)
            except Exception as e:
                axes[idx + 1].text(0.5, 0.5, f'Error:\n{corruption_type}', 
                                 ha='center', va='center', fontsize=6)
            axes[idx + 1].axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(all_corruptions) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'All Corruption Types (Severity=3) - {image_name}', fontsize=12)
        plt.tight_layout()
        
        output_dir = BASE_PATH / 'visualizations' / 'corruption_demo'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'all_corruptions_grid_{image_name}.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def demo_corruption_by_level(test_images: List[Tuple[np.ndarray, str]]):
    """按层级展示损坏效果"""
    from robustness.image_corruption import ImageCorruptor
    
    print("\n" + "=" * 60)
    print("Demo: Corruptions by Level")
    print("=" * 60)
    
    corruptor = ImageCorruptor(seed=42)
    
    # 按层级组织损坏
    level_corruptions = {
        'Pixel-Level\n(Noise)': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'poisson_noise'],
        'Region-Level\n(Blur)': ['gaussian_blur', 'defocus_blur', 'motion_blur', 'zoom_blur', 'glass_blur'],
        'Region-Level\n(Digital)': ['pixelate', 'jpeg_compression', 'elastic_transform'],
        'Global-Level\n(Weather)': ['snow', 'frost', 'fog', 'rain', 'spatter'],
        'Global-Level\n(Color)': ['brightness', 'contrast', 'saturation', 'hue_shift', 'channel_shuffle'],
        'Global-Level\n(Geometric)': ['scale', 'rotate', 'shear', 'perspective'],
    }
    
    for test_image, image_name in test_images:
        print(f"\nProcessing: {image_name}")
        
        n_levels = len(level_corruptions)
        max_corruptions = max(len(v) for v in level_corruptions.values())
        
        fig, axes = plt.subplots(n_levels, max_corruptions + 1, figsize=(max_corruptions * 2.2 + 2, n_levels * 2.2))
        
        for row, (level_name, corruptions) in enumerate(level_corruptions.items()):
            # 原图
            axes[row, 0].imshow(test_image)
            axes[row, 0].set_ylabel(level_name, fontsize=9)
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            if row == 0:
                axes[row, 0].set_title('Original', fontsize=9)
            
            # 该层级的损坏
            for col, corruption_type in enumerate(corruptions):
                try:
                    corrupted = corruptor.apply(test_image, corruption_type, severity=3)
                    axes[row, col + 1].imshow(corrupted)
                    axes[row, col + 1].set_title(corruption_type.replace('_', '\n'), fontsize=7)
                except Exception as e:
                    axes[row, col + 1].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[row, col + 1].axis('off')
            
            # 隐藏空白格子
            for col in range(len(corruptions) + 1, max_corruptions + 1):
                axes[row, col].axis('off')
        
        plt.suptitle(f'Corruptions by Level (Severity=3) - {image_name}', fontsize=12)
        plt.tight_layout()
        
        output_dir = BASE_PATH / 'visualizations' / 'corruption_demo'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'corruptions_by_level_{image_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def demo_config_presets():
    """演示配置预设"""
    from robustness.corruption_config import CorruptionConfig
    
    print("\n" + "=" * 60)
    print("Demo: Configuration Presets")
    print("=" * 60)
    
    presets = ['clean', 'light', 'moderate', 'severe', 'extreme', 'pixel_only', 'blur_only']
    
    print("\nAvailable presets:")
    for preset in presets:
        config = CorruptionConfig.from_preset(preset)
        print(f"  {preset:15s}: {config.get_description()}")
    
    print("\nAblation config example:")
    config = CorruptionConfig.ablation('gaussian_noise', severity=3)
    print(f"  gaussian_noise s3: {config.get_description()}")
    
    print("\nConfig to dict:")
    print(f"  {config.to_dict()}")


def demo_dataset_integration():
    """演示数据集集成 (需要数据集存在)"""
    from robustness.corruption_config import CorruptionConfig
    from robustness.corrupted_dataset import CorruptedCIRRDataset
    from data_utils import targetpad_transform
    
    print("\n" + "=" * 60)
    print("Demo: Dataset Integration")
    print("=" * 60)
    
    # 检查数据集是否存在
    cirr_path = BASE_PATH / 'cirr_dataset' / 'cirr' / 'captions' / 'cap.rc2.train.json'
    
    if not cirr_path.exists():
        print(f"\nCIRR dataset not found at: {cirr_path}")
        print("Skipping dataset integration demo.")
        print("\nTo run this demo, please download CIRR dataset and place it in:")
        print(f"  {BASE_PATH / 'cirr_dataset'}")
        return
    
    # 创建预处理
    preprocess = targetpad_transform(target_ratio=1.25, dim=224)
    
    # 创建配置
    config = CorruptionConfig(
        image_corruption_prob=1.0,
        image_corruption_types=['gaussian_noise', 'motion_blur'],
        image_severity_range=(2, 4),
        deterministic=True,
        return_corruption_info=True,
    )
    
    # 创建数据集
    print("\nCreating CorruptedCIRRDataset...")
    dataset = CorruptedCIRRDataset(
        split='train',
        mode='relative',
        preprocess=preprocess,
        corruption_config=config,
    )
    
    # 获取一些样本
    print("\nSampling data...")
    for i in range(3):
        sample = dataset[i]
        if sample is not None:
            ref_img, target_img, caption, corruption_info = sample
            print(f"\nSample {i}:")
            print(f"  Reference image shape: {ref_img.shape}")
            print(f"  Target image shape: {target_img.shape}")
            print(f"  Caption: {caption[:50]}...")
            print(f"  Corruption: {corruption_info}")
    
    # 打印统计
    print("\nCorruption statistics:")
    stats = dataset.get_corruption_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Image Corruption Demo')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to a specific image to use for demo')
    parser.add_argument('--dataset', type=str, default='CIRR', choices=['CIRR', 'FashionIQ'],
                       help='Dataset to load sample images from')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split')
    parser.add_argument('--num-images', type=int, default=2,
                       help='Number of sample images to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'single', 'severity', 'grid', 'level', 'config', 'dataset'],
                       help='Which demo to run')
    return parser.parse_args()


def main():
    """运行演示"""
    args = parse_args()
    
    print("=" * 60)
    print("CIR Image Corruption Module Demo")
    print("=" * 60)
    
    # 获取测试图片
    test_images = get_test_images(args)
    print(f"\nUsing {len(test_images)} test image(s)")
    
    # 运行指定的演示
    demos = {
        'single': lambda: demo_single_corruptions(test_images),
        'severity': lambda: demo_severity_levels(test_images),
        'grid': lambda: demo_all_corruptions_grid(test_images),
        'level': lambda: demo_corruption_by_level(test_images),
        'config': demo_config_presets,
        'dataset': demo_dataset_integration,
    }
    
    if args.demo == 'all':
        for name, func in demos.items():
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        try:
            demos[args.demo]()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print(f"Visualizations saved to: {BASE_PATH / 'visualizations' / 'corruption_demo'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
