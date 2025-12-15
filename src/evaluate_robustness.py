"""
评估SPRC在各种降质配置上的表现

Usage:
    # 评估单个配置
    python src/evaluate_robustness.py \
        --config config_c_cutout \
        --config-dir /home/caoyu/mnt/zhaoai/robustness_configs \
        --model-path models/your_model/tuned_clip_arithmetic.pth
    
    # 评估所有配置
    python src/evaluate_robustness.py \
        --config all \
        --config-dir /home/caoyu/mnt/zhaoai/robustness_configs \
        --model-path models/your_model/tuned_clip_arithmetic.pth
    
    # 使用预训练模型（不加载checkpoint）
    python src/evaluate_robustness.py \
        --config all \
        --config-dir /home/caoyu/mnt/zhaoai/robustness_configs
"""

import argparse
import json
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from operator import itemgetter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL.Image

from lavis.models import load_model_and_preprocess
from data_utils_degraded import CIRRDegradedDataset, targetpad_transform, ImageDegradation, base_path
from utils import collate_fn, extract_index_blip_features, device


def evaluate_on_config(
    model,
    txt_processors,
    config_dir: str,
    preprocess,
    config_name: str,
):
    """在单个降质配置上评估"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*60}")
    
    # 数据集
    relative_dataset = CIRRDegradedDataset(
        'val', 'relative', preprocess, 
        config_dir=config_dir,
        apply_degradation=False
    )
    classic_dataset = CIRRDegradedDataset(
        'val', 'classic', preprocess,
        config_dir=config_dir,
        apply_degradation=False
    )
    
    # 提取gallery特征
    print("  Extracting gallery features...")
    index_features, index_names = extract_index_blip_features(classic_dataset, model)
    name_to_feat = dict(zip(index_names, index_features[1]))
    
    # 加载降质参数
    with open(Path(config_dir) / 'cap.rc2.val.json') as f:
        triplets = json.load(f)
    name_to_deg_params = {t['reference']: t.get('degradation_params', None) for t in triplets}
    
    # 评估
    relative_loader = DataLoader(
        dataset=relative_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    all_distances = []
    target_names = []
    group_members_all = []
    reference_names = []
    
    print("  Computing predictions...")
    for batch_data in tqdm(relative_loader, desc="  Evaluating", ncols=100):
        batch_refs, batch_targets, captions, batch_groups = batch_data
        batch_groups = np.array(batch_groups).T.tolist()
        captions = [txt_processors["eval"](c) for c in captions]
        
        with torch.no_grad():
            # 加载并降质参考图像
            ref_images = []
            for ref_name in batch_refs:
                img_path = base_path / 'cirr_dataset' / relative_dataset.name_to_relpath[ref_name]
                img = PIL.Image.open(img_path).convert('RGB')
                
                # 应用降质
                deg_params = name_to_deg_params.get(ref_name, None)
                if deg_params:
                    if isinstance(deg_params, list):
                        img = ImageDegradation.apply_multi_degradation(img, deg_params)
                    else:
                        img = ImageDegradation.apply_degradation(img, deg_params)
                
                ref_images.append(preprocess(img))
            
            ref_images = torch.stack(ref_images).to(device)
            
            # 编码
            with torch.cuda.amp.autocast():
                ref_embeds = model.ln_vision(model.visual_encoder(ref_images))
            
            # 计算相似度
            batch_sim = model.inference(ref_embeds, index_features[0], captions)
            all_distances.append(batch_sim.cpu())
        
        target_names.extend(batch_targets)
        group_members_all.extend(batch_groups)
        reference_names.extend(batch_refs)
    
    # 计算metrics
    all_distances = torch.cat(all_distances)
    distances = 1 - all_distances
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    
    # 删除参考图像
    ref_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1)
    )
    sorted_index_names = sorted_index_names[ref_mask].reshape(len(target_names), -1)
    
    # 计算labels
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)-1).reshape(len(target_names), -1)
    )
    
    # Group metrics
    group_members_arr = np.array(group_members_all)
    group_mask = (sorted_index_names[..., None] == group_members_arr[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(len(target_names), -1)
    
    # 计算指标
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100
    
    results = {
        'R@1': recall_at1,
        'R@5': recall_at5,
        'R@10': recall_at10,
        'R@50': recall_at50,
        'R_s@1': group_recall_at1,
        'R_s@2': group_recall_at2,
        'R_s@3': group_recall_at3,
        'arithmetic_mean': mean([recall_at1, recall_at5, recall_at10, recall_at50,
                                 group_recall_at1, group_recall_at2, group_recall_at3]),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPRC on degraded datasets")
    parser.add_argument('--config', type=str, required=True,
                        help='Config name (e.g., config_c_cutout) or "all"')
    parser.add_argument('--config-dir', type=str, default='/home/caoyu/mnt/zhaoai/robustness_configs',
                        help='Root directory of robustness configs')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')
    parser.add_argument('--model-name', type=str, default='blip2_cir_align_prompt',
                        help='Model name')
    parser.add_argument('--backbone', type=str, default='pretrain')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    
    # 加载模型
    print(f"\nLoading model: {args.model_name}")
    model, _, txt_processors = load_model_and_preprocess(
        name=args.model_name, model_type=args.backbone, is_eval=True, device=device
    )
    
    # 加载checkpoint（如果提供）
    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        msg = model.load_state_dict(checkpoint[model.__class__.__name__], strict=False)
        print(f"  Missing keys: {msg.missing_keys[:5]}..." if msg.missing_keys else "  Loaded successfully")
    else:
        print("Using pretrained weights (no checkpoint)")
    
    model.eval()
    preprocess = targetpad_transform(1.25, 224)
    
    # 确定要评估的配置
    config_root = Path(args.config_dir)
    if args.config == 'all':
        configs = sorted([d.name for d in config_root.iterdir() if d.is_dir()])
    else:
        configs = [args.config]
    
    # 评估
    all_results = {}
    for config_name in configs:
        config_path = config_root / config_name
        if not (config_path / 'cap.rc2.val.json').exists():
            print(f"\nSkipping {config_name}: no validation data")
            continue
        
        results = evaluate_on_config(model, txt_processors, str(config_path), preprocess, config_name)
        all_results[config_name] = results
        
        print(f"\n  Results for {config_name}:")
        for k, v in results.items():
            print(f"    {k}: {v:.2f}")
    
    # 汇总输出
    print(f"\n{'='*100}")
    print("ROBUSTNESS EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"{'Config':<30} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'R_s@1':>8} {'Mean':>8}")
    print("-" * 100)
    
    for config_name in configs:
        if config_name not in all_results:
            continue
        r = all_results[config_name]
        print(f"{config_name:<30} {r['R@1']:>8.2f} {r['R@5']:>8.2f} {r['R@10']:>8.2f} "
              f"{r['R@50']:>8.2f} {r['R_s@1']:>8.2f} {r['arithmetic_mean']:>8.2f}")
    
    print("-" * 100)
    
    # 计算性能下降
    if 'config_a_clean' in all_results:
        baseline = all_results['config_a_clean']['arithmetic_mean']
        print(f"\nPerformance drop from clean baseline (Mean):")
        for config_name in configs:
            if config_name in all_results and config_name != 'config_a_clean':
                drop = baseline - all_results[config_name]['arithmetic_mean']
                print(f"  {config_name}: -{drop:.2f}%")
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = Path(args.model_path).stem if args.model_path else 'pretrain'
        output_path = f"robustness_results_{model_suffix}_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()









