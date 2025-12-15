"""
可视化CIRR鲁棒性测试配置的样本

用于验证生成的各种配置是否正确：
- Config A (Clean): 原始数据
- Config B (Noisy Label): 标签噪声
- Config C (Cutout): 遮挡
- Config D (Blur): 模糊
- Config E (Mixed): 混合噪声

Usage:
    python src/visualize_robustness_configs.py \
        --configs-dir ../robustness_configs \
        --cirr-root ./cirr_dataset \
        --output-dir ../robustness_vis \
        --num-samples 5
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Error: PIL is required for visualization")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found, will save individual images only")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 导入降质工具
from generate_robustness_configs import ImageDegradation


class RobustnessConfigVisualizer:
    """鲁棒性配置可视化器"""
    
    def __init__(
        self,
        configs_dir: str,
        cirr_root: str,
        output_dir: str,
        seed: int = 42
    ):
        """
        Args:
            configs_dir: 配置文件目录
            cirr_root: CIRR数据集根目录
            output_dir: 可视化输出目录
        """
        self.configs_dir = Path(configs_dir)
        self.cirr_root = Path(cirr_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 配置名称映射
        self.config_names = {
            'config_a_clean': 'Config A: Clean',
            'config_b_noisy_20': 'Config B: Noisy Label 20%',
            'config_b_noisy_50': 'Config B: Noisy Label 50%',
            'config_c_cutout': 'Config C: Cutout',
            'config_d_gaussian_blur': 'Config D: Gaussian Blur',
            'config_d_motion_blur': 'Config D: Motion Blur',
            'config_e_mixed': 'Config E: Mixed'
        }
        
        # 加载原始image_splits用于获取图像路径
        self.image_splits = {}
        for split in ['train', 'val', 'test1']:
            split_path = self.cirr_root / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
            if split_path.exists():
                with open(split_path, 'r') as f:
                    self.image_splits[split] = json.load(f)
    
    def load_config(self, config_name: str, split: str = 'train') -> List[dict]:
        """加载指定配置的标注"""
        config_path = self.configs_dir / config_name / f'cap.rc2.{split}.json'
        if not config_path.exists():
            print(f"Warning: Config not found: {config_path}")
            return []
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_image_path(self, image_name: str, split: str = 'train') -> Optional[Path]:
        """获取图像的完整路径"""
        if split not in self.image_splits:
            return None
        
        if image_name not in self.image_splits[split]:
            # 尝试其他split
            for s in ['train', 'val', 'test1']:
                if s in self.image_splits and image_name in self.image_splits[s]:
                    rel_path = self.image_splits[s][image_name]
                    return self.cirr_root / rel_path
            return None
        
        rel_path = self.image_splits[split][image_name]
        return self.cirr_root / rel_path
    
    def load_image(self, image_name: str, split: str = 'train') -> Optional[Image.Image]:
        """加载图像"""
        img_path = self.get_image_path(image_name, split)
        if img_path is None or not img_path.exists():
            print(f"Warning: Image not found: {image_name}")
            return None
        
        try:
            return Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def apply_degradation(self, image: Image.Image, sample: dict) -> Image.Image:
        """应用降质变换"""
        params = sample.get('degradation_params')
        if params is None:
            return image
        
        if isinstance(params, list):
            # 多个降质（Config E）
            return ImageDegradation.apply_multi_degradation(image, params)
        else:
            # 单个降质
            return ImageDegradation.apply_degradation(image, params)
    
    def add_text_to_image(
        self,
        image: Image.Image,
        text: str,
        position: str = 'top',
        font_size: int = 14,
        bg_color: tuple = (0, 0, 0, 180),
        text_color: tuple = (255, 255, 255)
    ) -> Image.Image:
        """在图像上添加文字"""
        # 转换为RGBA以支持透明度
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # 计算文字边界框
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算位置
        img_width, img_height = image.size
        padding = 5
        
        if position == 'top':
            x = padding
            y = padding
        elif position == 'bottom':
            x = padding
            y = img_height - text_height - padding * 2
        else:
            x, y = padding, padding
        
        # 绘制背景
        bg_box = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
        
        # 创建半透明背景
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bg_box, fill=bg_color)
        image = Image.alpha_composite(image, overlay)
        
        # 绘制文字
        draw = ImageDraw.Draw(image)
        draw.text((x, y), text, fill=text_color, font=font)
        
        return image.convert('RGB')
    
    def visualize_single_sample(
        self,
        sample: dict,
        config_name: str,
        save_path: Path,
        show_original: bool = True
    ):
        """可视化单个样本"""
        reference_name = sample.get('reference')
        target_name = sample.get('target_hard')
        caption = sample.get('caption', '')
        is_noisy = sample.get('is_label_noisy', False)
        original_target = sample.get('original_target', '')
        degradation = sample.get('reference_degradation', 'none')
        
        # 加载图像
        ref_img = self.load_image(reference_name)
        target_img = self.load_image(target_name)
        
        if ref_img is None:
            print(f"Skipping sample: reference image not found")
            return False
        
        # 应用降质
        degraded_ref = self.apply_degradation(ref_img.copy(), sample)
        
        # 创建可视化
        if HAS_MPL:
            fig = plt.figure(figsize=(16, 6))
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8])
            
            # 原始参考图像
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(ref_img)
            ax1.set_title('Original Reference', fontsize=12)
            ax1.axis('off')
            
            # 降质后的参考图像
            ax2 = fig.add_subplot(gs[1])
            ax2.imshow(degraded_ref)
            title2 = f'Degraded Reference\n({degradation})'
            ax2.set_title(title2, fontsize=12)
            ax2.axis('off')
            
            # 目标图像
            ax3 = fig.add_subplot(gs[2])
            if target_img is not None:
                ax3.imshow(target_img)
                if is_noisy:
                    ax3.set_title(f'Target (NOISY)\nOriginal: {original_target[:20]}...', fontsize=12, color='red')
                else:
                    ax3.set_title('Target', fontsize=12)
            else:
                ax3.text(0.5, 0.5, 'Target\nNot Found', ha='center', va='center', fontsize=14)
            ax3.axis('off')
            
            # 信息面板
            ax4 = fig.add_subplot(gs[3])
            ax4.axis('off')
            
            info_text = f"Config: {self.config_names.get(config_name, config_name)}\n\n"
            info_text += f"Reference: {reference_name[:25]}...\n\n"
            info_text += f"Target: {target_name[:25]}...\n\n"
            info_text += f"Caption:\n{caption[:100]}{'...' if len(caption) > 100 else ''}\n\n"
            info_text += f"Degradation: {degradation}\n\n"
            
            if is_noisy:
                info_text += f"⚠️ NOISY LABEL\n"
                info_text += f"Original target:\n{original_target[:25]}..."
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            # 没有matplotlib，拼接多张图像并添加文字
            # 创建拼接图像: [原图 | 降质图 | 目标图]
            images_to_concat = [ref_img, degraded_ref]
            if target_img is not None:
                images_to_concat.append(target_img)
            
            # 统一高度
            max_height = max(img.height for img in images_to_concat)
            resized_images = []
            for img in images_to_concat:
                if img.height != max_height:
                    ratio = max_height / img.height
                    new_width = int(img.width * ratio)
                    img = img.resize((new_width, max_height), Image.LANCZOS)
                resized_images.append(img)
            
            # 拼接
            total_width = sum(img.width for img in resized_images) + 10 * (len(resized_images) - 1)
            concat_img = Image.new('RGB', (total_width, max_height + 80), (255, 255, 255))
            
            x_offset = 0
            for img in resized_images:
                concat_img.paste(img, (x_offset, 40))
                x_offset += img.width + 10
            
            # 添加标题文字
            info_text = f"{self.config_names.get(config_name, config_name)} | {degradation}"
            if is_noisy:
                info_text += f" | ⚠️ NOISY LABEL (Original: {original_target[:20]}...)"
            
            concat_img = self.add_text_to_image(concat_img, info_text, position='top', font_size=16)
            
            # 添加底部说明
            bottom_text = f"Caption: {caption[:80]}..."
            concat_img = self.add_text_to_image(concat_img, bottom_text, position='bottom', font_size=12)
            
            concat_img.save(save_path)
        
        return True
    
    def visualize_config_comparison(
        self,
        sample_idx: int,
        save_path: Path
    ):
        """对比可视化同一样本在不同配置下的效果"""
        # 首先从clean配置加载原始样本
        clean_samples = self.load_config('config_a_clean', 'train')
        if sample_idx >= len(clean_samples):
            print(f"Sample index {sample_idx} out of range")
            return False
        
        original_sample = clean_samples[sample_idx]
        reference_name = original_sample.get('reference')
        ref_img = self.load_image(reference_name)
        
        if ref_img is None:
            return False
        
        # 收集所有配置
        configs_to_compare = [
            'config_a_clean',
            'config_b_noisy_20',
            'config_c_cutout',
            'config_d_gaussian_blur',
            'config_d_motion_blur',
            'config_e_mixed'
        ]
        
        if HAS_MPL:
            n_configs = len(configs_to_compare)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, config_name in enumerate(configs_to_compare):
                samples = self.load_config(config_name, 'train')
                if sample_idx < len(samples):
                    sample = samples[sample_idx]
                    degraded = self.apply_degradation(ref_img.copy(), sample)
                    
                    axes[i].imshow(degraded)
                    
                    title = self.config_names.get(config_name, config_name)
                    degradation = sample.get('reference_degradation', 'none')
                    is_noisy = sample.get('is_label_noisy', False)
                    
                    color = 'red' if is_noisy else 'black'
                    axes[i].set_title(f"{title}\n({degradation})", fontsize=10, color=color)
                else:
                    axes[i].text(0.5, 0.5, 'Config\nNot Found', ha='center', va='center')
                    axes[i].set_title(config_name, fontsize=10)
                
                axes[i].axis('off')
            
            # 添加总标题
            fig.suptitle(
                f"Sample {sample_idx}: {reference_name}\nCaption: {original_sample.get('caption', '')[:80]}...",
                fontsize=12, y=1.02
            )
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return True
        
        return False
    
    def _select_effective_samples(
        self,
        samples: List[dict],
        config_name: str,
        num_samples: int
    ) -> List[int]:
        """
        优先选择有效果的样本进行可视化
        
        对于不同配置类型：
        - Config B (noisy): 选择 is_label_noisy=True 的样本
        - Config C (cutout): 选择有 cutout 降质的样本
        - Config D (blur): 选择有 blur 降质的样本
        - Config E (mixed): 选择有任何降质效果的样本
        - Config A (clean): 随机选择
        """
        effective_indices = []
        
        if 'noisy' in config_name:
            # Config B: 选择有噪声标签的样本
            effective_indices = [
                i for i, s in enumerate(samples) 
                if s.get('is_label_noisy', False)
            ]
        elif 'cutout' in config_name:
            # Config C: 选择有 cutout 的样本
            effective_indices = [
                i for i, s in enumerate(samples)
                if s.get('degradation_params') is not None
            ]
        elif 'blur' in config_name:
            # Config D: 选择有 blur 的样本
            effective_indices = [
                i for i, s in enumerate(samples)
                if s.get('degradation_params') is not None
            ]
        elif 'mixed' in config_name:
            # Config E: 选择有任何降质效果的样本（非clean）
            effective_indices = [
                i for i, s in enumerate(samples)
                if s.get('reference_degradation', 'clean') != 'clean' or s.get('is_label_noisy', False)
            ]
        
        # 如果找到了有效果的样本，从中随机选择
        if effective_indices:
            selected = random.sample(
                effective_indices, 
                min(num_samples, len(effective_indices))
            )
            print(f"  Found {len(effective_indices)} effective samples, selected {len(selected)}")
            return selected
        
        # 否则随机选择
        print(f"  No effective samples found, selecting randomly")
        return random.sample(range(len(samples)), min(num_samples, len(samples)))
    
    def visualize_all_configs(self, num_samples: int = 5):
        """为所有配置生成可视化"""
        print("\n" + "=" * 60)
        print("Generating Visualizations for Robustness Configs")
        print("=" * 60)
        
        # 获取所有可用配置
        available_configs = []
        for config_dir in self.configs_dir.iterdir():
            if config_dir.is_dir() and config_dir.name.startswith('config_'):
                available_configs.append(config_dir.name)
        
        available_configs.sort()
        print(f"Found {len(available_configs)} configs: {available_configs}")
        
        # 为每个配置生成样本可视化
        for config_name in available_configs:
            print(f"\nVisualizing {config_name}...")
            
            config_output_dir = self.output_dir / config_name
            config_output_dir.mkdir(parents=True, exist_ok=True)
            
            samples = self.load_config(config_name, 'train')
            if not samples:
                print(f"  No samples found")
                continue
            
            # 优先选择有效果的样本
            sample_indices = self._select_effective_samples(samples, config_name, num_samples)
            
            for i, idx in enumerate(sample_indices):
                save_path = config_output_dir / f'sample_{i:02d}_idx{idx}.png'
                success = self.visualize_single_sample(
                    samples[idx],
                    config_name,
                    save_path
                )
                if success:
                    print(f"  Saved: {save_path.name}")
        
        # 生成对比可视化
        print("\nGenerating comparison visualizations...")
        comparison_dir = self.output_dir / 'comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择几个样本进行对比
        clean_samples = self.load_config('config_a_clean', 'train')
        if clean_samples:
            comparison_indices = random.sample(range(len(clean_samples)), min(num_samples, len(clean_samples)))
            
            for i, idx in enumerate(comparison_indices):
                save_path = comparison_dir / f'comparison_{i:02d}_idx{idx}.png'
                success = self.visualize_config_comparison(idx, save_path)
                if success:
                    print(f"  Saved: {save_path.name}")
        
        print("\n" + "=" * 60)
        print(f"Visualizations saved to: {self.output_dir}")
        print("=" * 60)
    
    def generate_summary_grid(self, save_path: Optional[Path] = None):
        """生成配置摘要网格图"""
        if not HAS_MPL:
            print("matplotlib required for summary grid")
            return
        
        print("\nGenerating summary grid...")
        
        # 选择一个固定样本
        clean_samples = self.load_config('config_a_clean', 'train')
        if not clean_samples:
            print("No clean samples found")
            return
        
        # 找一个有效的样本
        sample_idx = 0
        ref_img = None
        while sample_idx < len(clean_samples) and ref_img is None:
            reference_name = clean_samples[sample_idx].get('reference')
            ref_img = self.load_image(reference_name)
            if ref_img is None:
                sample_idx += 1
        
        if ref_img is None:
            print("No valid reference image found")
            return
        
        original_sample = clean_samples[sample_idx]
        
        # 配置列表
        configs = [
            ('config_a_clean', 'A: Clean'),
            ('config_b_noisy_20', 'B: Noisy 20%'),
            ('config_b_noisy_50', 'B: Noisy 50%'),
            ('config_c_cutout', 'C: Cutout'),
            ('config_d_gaussian_blur', 'D: Gaussian Blur'),
            ('config_d_motion_blur', 'D: Motion Blur'),
            ('config_e_mixed', 'E: Mixed'),
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # 第一个显示原始图像
        axes[0].imshow(ref_img)
        axes[0].set_title('Original Image', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # 显示各配置效果
        for i, (config_name, display_name) in enumerate(configs):
            ax = axes[i + 1] if i < 7 else None
            if ax is None:
                break
            
            samples = self.load_config(config_name, 'train')
            if sample_idx < len(samples):
                sample = samples[sample_idx]
                degraded = self.apply_degradation(ref_img.copy(), sample)
                ax.imshow(degraded)
                
                # 根据是否有噪声标签设置颜色
                is_noisy = sample.get('is_label_noisy', False)
                color = 'red' if is_noisy else 'black'
                
                title = display_name
                if is_noisy:
                    title += '\n⚠️ Noisy Label'
                
                ax.set_title(title, fontsize=11, color=color)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
                ax.set_title(display_name, fontsize=11)
            
            ax.axis('off')
        
        # 添加标题
        caption = original_sample.get('caption', '')[:60]
        fig.suptitle(
            f"Robustness Config Comparison\nCaption: \"{caption}...\"",
            fontsize=14, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'summary_grid.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Summary grid saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CIRR robustness benchmark configs"
    )
    parser.add_argument(
        '--configs-dir',
        type=str,
        default='../robustness_configs',
        help='Path to robustness configs directory'
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
        default='../robustness_vis',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize per config'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only generate summary grid'
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    
    # 处理相对路径
    if not os.path.isabs(args.configs_dir):
        configs_dir = (script_dir / args.configs_dir).resolve()
    else:
        configs_dir = Path(args.configs_dir)
    
    if not os.path.isabs(args.cirr_root):
        cirr_root = (script_dir / args.cirr_root).resolve()
    else:
        cirr_root = Path(args.cirr_root)
    
    if not os.path.isabs(args.output_dir):
        output_dir = (script_dir / args.output_dir).resolve()
    else:
        output_dir = Path(args.output_dir)
    
    print(f"Configs dir: {configs_dir}")
    print(f"CIRR root: {cirr_root}")
    print(f"Output dir: {output_dir}")
    
    # 检查目录
    if not configs_dir.exists():
        print(f"Error: Configs directory not found: {configs_dir}")
        print("Please run generate_robustness_configs.py first.")
        return
    
    if not cirr_root.exists():
        print(f"Error: CIRR root not found: {cirr_root}")
        return
    
    visualizer = RobustnessConfigVisualizer(
        configs_dir=str(configs_dir),
        cirr_root=str(cirr_root),
        output_dir=str(output_dir),
        seed=args.seed
    )
    
    if args.summary_only:
        visualizer.generate_summary_grid()
    else:
        visualizer.visualize_all_configs(num_samples=args.num_samples)
        visualizer.generate_summary_grid()


if __name__ == "__main__":
    main()

