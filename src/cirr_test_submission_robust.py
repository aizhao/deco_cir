"""
CIRR 测试集鲁棒性评估
====================
在测试集的参考图像上应用图像损坏和/或文本扰动，生成提交文件

Usage:
    # 仅图像损坏
    python cirr_test_submission_robust.py \
        --model-path /path/to/model.pt \
        --image-corruption gaussian_noise \
        --image-severity 3
    
    # 仅文本扰动
    python cirr_test_submission_robust.py \
        --model-path /path/to/model.pt \
        --text-perturbation keyboard_typo \
        --text-severity 3
    
    # 双模态噪声
    python cirr_test_submission_robust.py \
        --model-path /path/to/model.pt \
        --image-corruption gaussian_noise --image-severity 3 \
        --text-perturbation keyboard_typo --text-severity 3
    
    # 批量消融
    python cirr_test_submission_robust.py \
        --model-path /path/to/model.pt \
        --ablation --ablation-mode image
"""

import json
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL.Image

from data_utils import CIRRDataset, targetpad_transform, base_path
from utils import device, extract_index_blip_features
from lavis.models import load_model_and_preprocess
from robustness import ImageCorruptor, TextPerturber


def extract_corrupted_reference_features(
    dataset: CIRRDataset,
    blip_model,
    preprocess,
    corruptor: ImageCorruptor,
    corruption_type: str,
    severity: int,
    batch_size: int = 128,  # 批量处理
) -> Dict[str, torch.Tensor]:
    """
    提取带损坏的参考图像特征（批量处理优化）
    """
    # 先获取干净的 index features
    classic_dataset = CIRRDataset('test1', 'classic', preprocess)
    (index_features, index_features_raw), index_names = extract_index_blip_features(classic_dataset, blip_model)
    
    # 创建基础映射（干净）
    name_to_feat = dict(zip(index_names, index_features_raw))
    
    # 如果不需要损坏，直接返回
    # 返回元组 (index_features, index_features_raw)，与原版 extract_index_blip_features 返回格式一致
    if corruption_type is None:
        return name_to_feat, (index_features, index_features_raw), index_names
    
    # 获取图像路径映射
    with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / 'split.rc2.test1.json') as f:
        name_to_relpath = json.load(f)
    
    # 找出所有需要作为参考的图像名称
    relative_dataset = CIRRDataset('test1', 'relative', preprocess)
    reference_names_set = set()
    for i in range(len(relative_dataset)):
        _, ref_name, _, _ = relative_dataset[i]
        reference_names_set.add(ref_name)
    
    reference_names_list = list(reference_names_set)
    print(f"Corrupting {len(reference_names_list)} reference images with {corruption_type} (severity {severity})")
    
    # 批量处理
    corruptor.set_seed(42)
    
    for batch_start in tqdm(range(0, len(reference_names_list), batch_size), 
                            desc="Extracting corrupted features"):
        batch_names = reference_names_list[batch_start:batch_start + batch_size]
        batch_tensors = []
        
        for ref_name in batch_names:
            img_path = base_path / 'cirr_dataset' / name_to_relpath[ref_name]
            img = PIL.Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            # 应用损坏
            corrupted_np = corruptor.apply(img_np, corruption_type, severity)
            corrupted_img = PIL.Image.fromarray(corrupted_np)
            
            # 预处理
            batch_tensors.append(preprocess(corrupted_img))
        
        # 批量推理
        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            _, raw_feats = blip_model.extract_target_features(batch_tensor, mode="mean")
        
        # 更新映射
        for i, ref_name in enumerate(batch_names):
            name_to_feat[ref_name] = raw_feats[i].clone()
    
    # 返回元组 (index_features, index_features_raw)，与原版格式一致
    return name_to_feat, (index_features, index_features_raw), index_names


def generate_cirr_test_predictions_corrupted(
    blip_model,
    relative_test_dataset: CIRRDataset,
    index_names: List[str],
    index_features: torch.Tensor,
    txt_processors,
    name_to_feat: Dict[str, torch.Tensor],
    text_perturber: Optional[TextPerturber] = None,
    text_perturbation_type: Optional[str] = None,
    text_severity: int = 3,
) -> Tuple[torch.Tensor, List[str], List[List[str]], List[str], List[str]]:
    """
    使用损坏的参考图像特征和/或扰动的文本计算预测
    
    Args:
        text_perturber: 文本扰动器实例
        text_perturbation_type: 文本扰动类型
        text_severity: 文本扰动严重度
    """
    desc = "Computing predictions"
    if text_perturbation_type:
        desc += f" (text: {text_perturbation_type} s{text_severity})"
    print(desc)
    
    relative_test_loader = DataLoader(
        dataset=relative_test_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    pairs_id = []
    group_members = []
    reference_names = []
    distance = []
    captions_all = []
    captions_original = []  # 保存原始 caption 用于 rerank
    
    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(relative_test_loader):
        batch_group_members = np.array(batch_group_members).T.tolist()
        
        # 保存原始 caption
        captions_original.extend(captions)
        
        # 应用文本扰动
        if text_perturber and text_perturbation_type:
            captions = [
                text_perturber.apply(cap, text_perturbation_type, text_severity) 
                for cap in captions
            ]
        
        # 使用 txt_processors 处理
        captions = [txt_processors["eval"](caption) for caption in captions]
        
        with torch.no_grad():
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(
                    itemgetter(*batch_reference_names)(name_to_feat)
                )
            
            batch_distance = blip_model.inference(reference_image_features, index_features[0], captions)
            distance.append(batch_distance)
            captions_all += captions
        
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)
    
    distance = torch.vstack(distance)
    return distance, reference_names, group_members, pairs_id, captions_all


def generate_submission_with_corruption(
    blip_model,
    preprocess,
    txt_processors,
    image_corruption_type: Optional[str] = None,
    image_severity: int = 3,
    text_perturbation_type: Optional[str] = None,
    text_severity: int = 3,
    rerank: bool = False,
):
    """
    生成带损坏的测试提交文件
    
    Args:
        image_corruption_type: 图像损坏类型 (None 表示不损坏)
        image_severity: 图像损坏严重度 (1-5)
        text_perturbation_type: 文本扰动类型 (None 表示不扰动)
        text_severity: 文本扰动严重度 (1-5)
        rerank: 是否重排序
    """
    print(f"\n{'='*60}")
    img_desc = f"img:{image_corruption_type}(s{image_severity})" if image_corruption_type else "img:clean"
    txt_desc = f"txt:{text_perturbation_type}(s{text_severity})" if text_perturbation_type else "txt:clean"
    print(f"Generating submission: {img_desc} | {txt_desc}")
    print(f"{'='*60}")
    
    corruptor = ImageCorruptor(seed=42)
    text_perturber = TextPerturber(seed=42) if text_perturbation_type else None
    
    # 提取特征（对参考图像应用损坏）
    name_to_feat, index_features, index_names = extract_corrupted_reference_features(
        dataset=CIRRDataset('test1', 'classic', preprocess),
        blip_model=blip_model,
        preprocess=preprocess,
        corruptor=corruptor,
        corruption_type=image_corruption_type,
        severity=image_severity,
    )
    
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)
    
    # 生成预测
    predicted_sim, reference_names, group_members, pairs_id, captions_all = \
        generate_cirr_test_predictions_corrupted(
            blip_model, relative_test_dataset, index_names, index_features, 
            txt_processors, name_to_feat,
            text_perturber=text_perturber,
            text_perturbation_type=text_perturbation_type,
            text_severity=text_severity,
        )
    
    # 计算距离和排序
    distances = 1 - predicted_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    
    # Re-rank (optional)
    if rerank:
        print('Reranking...')
        i = 0
        step = 50
        top = 50
        while i < len(sorted_index_names):
            if step + i > len(sorted_index_names):
                step = len(sorted_index_names) - i
            reference_name = reference_names[i: i + step]
            caption = captions_all[i: i + step]
            targets_top100 = sorted_index_names[i: i + step, :top]
            
            if step == 1:
                reference_feats = itemgetter(*reference_name)(name_to_feat).unsqueeze(0)
            else:
                reference_feats = torch.stack(itemgetter(*reference_name)(name_to_feat))
            target_feats = torch.stack(itemgetter(*targets_top100.reshape(-1))(name_to_feat))
            
            with torch.no_grad():
                top100_rank = blip_model.inference_rerank(reference_feats, target_feats, caption)
            distances_top100 = 1 - top100_rank
            distances_top100 = distances_top100.reshape(-1, top)
            sorted_indices_top100 = torch.argsort(distances_top100, dim=-1).cpu()
            
            for j in range(step):
                sorted_index_names[i + j, :top] = sorted_index_names[i + j, :top][sorted_indices_top100[j]]
            i = i + step
    
    # 移除参考图像
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(
            np.array(reference_names), len(index_names)
        ).reshape(len(sorted_index_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0], sorted_index_names.shape[1] - 1
    )
    
    # 计算子集预测
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
    
    # 生成预测字典
    pairid_to_predictions = {
        str(int(pair_id)): prediction[:50].tolist() 
        for (pair_id, prediction) in zip(pairs_id, sorted_index_names)
    }
    pairid_to_group_predictions = {
        str(int(pair_id)): prediction[:3].tolist() 
        for (pair_id, prediction) in zip(pairs_id, sorted_group_names)
    }
    
    # 创建提交文件
    submission = {'version': 'rc2', 'metric': 'recall'}
    group_submission = {'version': 'rc2', 'metric': 'recall_subset'}
    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)
    
    # 保存
    submissions_folder = base_path / "submission" / 'CIRR_robust'
    submissions_folder.mkdir(exist_ok=True, parents=True)
    
    # 构建文件名后缀
    parts = []
    if image_corruption_type:
        parts.append(f"img_{image_corruption_type}_s{image_severity}")
    if text_perturbation_type:
        parts.append(f"txt_{text_perturbation_type}_s{text_severity}")
    
    if parts:
        file_suffix = "_".join(parts)
    else:
        file_suffix = "clean"
    
    if rerank:
        file_suffix += "_rerank"
    
    recall_path = submissions_folder / f"recall_submission_{file_suffix}.json"
    recall_subset_path = submissions_folder / f"recall_subset_submission_{file_suffix}.json"
    
    with open(recall_path, 'w') as f:
        json.dump(submission, f, sort_keys=True)
    with open(recall_subset_path, 'w') as f:
        json.dump(group_submission, f, sort_keys=True)
    
    print(f"Saved: {recall_path}")
    print(f"Saved: {recall_subset_path}")
    
    return recall_path, recall_subset_path


def run_test_ablation(
    blip_model,
    preprocess,
    txt_processors,
    ablation_mode: str = 'image',  # 'image', 'text', 'bimodal'
    image_corruption_types: List[str] = None,
    text_perturbation_types: List[str] = None,
    severities: List[int] = [1, 3, 5],  # 测试集只用3个级别节省时间
    rerank: bool = False,
):
    """
    在测试集上运行消融实验，生成多个提交文件
    
    Args:
        ablation_mode: 消融模式
            - 'image': 仅图像损坏
            - 'text': 仅文本扰动
            - 'bimodal': 双模态 (固定一个，消融另一个)
        image_corruption_types: 图像损坏类型列表
        text_perturbation_types: 文本扰动类型列表
        severities: 严重度列表
        rerank: 是否重排序
    """
    # 默认图像损坏类型
    if image_corruption_types is None:
        image_corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',  # 噪声
            'gaussian_blur', 'motion_blur', 'defocus_blur',   # 模糊
            'fog', 'frost', 'snow',                            # 天气
            'brightness', 'contrast',                          # 亮度
            'jpeg_compression', 'pixelate',                    # 数字
        ]
    
    # 默认文本扰动类型
    if text_perturbation_types is None:
        text_perturbation_types = [
            'keyboard_typo', 'char_swap', 'char_delete',      # 字符级
            'synonym_replace', 'antonym_replace',              # 词级
            'word_delete', 'word_swap',                        # 词级
            'spelling_error',                                  # 拼写
            'case_change',                                     # 格式
        ]
    
    # 先生成干净基线
    print("\n[Baseline] Generating clean submission...")
    generate_submission_with_corruption(
        blip_model, preprocess, txt_processors,
        image_corruption_type=None, image_severity=0,
        text_perturbation_type=None, text_severity=0,
        rerank=rerank
    )
    
    if ablation_mode == 'image':
        # 仅图像消融
        total = len(image_corruption_types) * len(severities)
        current = 0
        
        for corruption_type in image_corruption_types:
            for severity in severities:
                current += 1
                print(f"\n[{current}/{total}] Image: {corruption_type} severity {severity}")
                generate_submission_with_corruption(
                    blip_model, preprocess, txt_processors,
                    image_corruption_type=corruption_type, 
                    image_severity=severity,
                    text_perturbation_type=None,
                    text_severity=0,
                    rerank=rerank
                )
    
    elif ablation_mode == 'text':
        # 仅文本消融
        total = len(text_perturbation_types) * len(severities)
        current = 0
        
        for perturbation_type in text_perturbation_types:
            for severity in severities:
                current += 1
                print(f"\n[{current}/{total}] Text: {perturbation_type} severity {severity}")
                generate_submission_with_corruption(
                    blip_model, preprocess, txt_processors,
                    image_corruption_type=None,
                    image_severity=0,
                    text_perturbation_type=perturbation_type, 
                    text_severity=severity,
                    rerank=rerank
                )
    
    elif ablation_mode == 'bimodal':
        # 双模态消融 - 选择代表性的组合
        combinations = []
        
        # 图像噪声 + 文本打字错误
        for img_type in ['gaussian_noise', 'motion_blur', 'fog']:
            for txt_type in ['keyboard_typo', 'synonym_replace', 'word_delete']:
                for sev in severities:
                    combinations.append((img_type, txt_type, sev, sev))
        
        total = len(combinations)
        current = 0
        
        for img_type, txt_type, img_sev, txt_sev in combinations:
            current += 1
            print(f"\n[{current}/{total}] Bimodal: img={img_type}(s{img_sev}), txt={txt_type}(s{txt_sev})")
            generate_submission_with_corruption(
                blip_model, preprocess, txt_processors,
                image_corruption_type=img_type,
                image_severity=img_sev,
                text_perturbation_type=txt_type,
                text_severity=txt_sev,
                rerank=rerank
            )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentParser.ArgumentTypeError('Boolean value expected.')


def main():
    parser = ArgumentParser()
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--rerank", type=str2bool, default=False)
    
    # 图像损坏参数
    parser.add_argument("--image-corruption", type=str, default=None, 
                        help="Image corruption type (e.g., gaussian_noise)")
    parser.add_argument("--image-severity", type=int, default=3, 
                        help="Image corruption severity (1-5)")
    
    # 文本扰动参数
    parser.add_argument("--text-perturbation", type=str, default=None,
                        help="Text perturbation type (e.g., keyboard_typo)")
    parser.add_argument("--text-severity", type=int, default=3,
                        help="Text perturbation severity (1-5)")
    
    # 消融实验参数
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--ablation-mode", type=str, default='image',
                        choices=['image', 'text', 'bimodal'],
                        help="Ablation mode: image, text, or bimodal")
    parser.add_argument("--image-corruptions", type=str, nargs='+', default=None, 
                        help="List of image corruption types for ablation")
    parser.add_argument("--text-perturbations", type=str, nargs='+', default=None,
                        help="List of text perturbation types for ablation")
    parser.add_argument("--severities", type=int, nargs='+', default=[1, 3, 5],
                        help="List of severities for ablation")
    
    # 兼容旧参数 (deprecated)
    parser.add_argument("--corruption", type=str, default=None, 
                        help="[Deprecated] Use --image-corruption instead")
    parser.add_argument("--severity", type=int, default=None,
                        help="[Deprecated] Use --image-severity instead")
    
    args = parser.parse_args()
    
    # 处理兼容性
    if args.corruption and not args.image_corruption:
        print("Warning: --corruption is deprecated, use --image-corruption instead")
        args.image_corruption = args.corruption
    if args.severity is not None and args.image_severity == 3:
        print("Warning: --severity is deprecated, use --image-severity instead")
        args.image_severity = args.severity
    
    # 加载模型
    print("Loading model...")
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=args.blip_model_name, 
        model_type=args.backbone, 
        is_eval=False, 
        device=device
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    blip_model.eval()
    
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    
    if args.ablation:
        # 运行消融实验
        run_test_ablation(
            blip_model, preprocess, txt_processors,
            ablation_mode=args.ablation_mode,
            image_corruption_types=args.image_corruptions,
            text_perturbation_types=args.text_perturbations,
            severities=args.severities,
            rerank=args.rerank
        )
    else:
        # 单次运行
        generate_submission_with_corruption(
            blip_model, preprocess, txt_processors,
            image_corruption_type=args.image_corruption,
            image_severity=args.image_severity,
            text_perturbation_type=args.text_perturbation,
            text_severity=args.text_severity,
            rerank=args.rerank
        )


if __name__ == '__main__':
    main()