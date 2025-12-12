# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
import os

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results,update_train_running_results_dict, set_train_bar_description_dict,set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics


def load_model_with_config(
    name: str,
    model_type: str,
    is_eval: bool = False,
    device: str = "cpu",
    config_overrides: Optional[Dict[str, Any]] = None
):
    """
    加载模型并支持配置覆盖
    
    Args:
        name: 模型名称 (如 'blip2_dual_stream_cir')
        model_type: 模型类型 (如 'pretrain')
        is_eval: 是否为评估模式
        device: 设备
        config_overrides: 要覆盖的配置字典，如 {'semantic_model_type': 'qwen2_vl'}
    
    Returns:
        model, vis_processors, txt_processors
    """
    from lavis.processors import load_processor
    from lavis.models import load_preprocess
    
    model_cls = registry.get_model_class(name)
    
    # 加载默认配置
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    
    # 应用配置覆盖
    if config_overrides:
        for key, value in config_overrides.items():
            # 支持嵌套配置如 'model.semantic_model_type'
            if '.' in key:
                parts = key.split('.')
                current = cfg
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                # 直接在 model 配置下设置
                if hasattr(cfg, 'model'):
                    cfg.model[key] = value
                else:
                    cfg[key] = value
        
        print(f"配置覆盖: {config_overrides}")
    
    # 从修改后的配置创建模型
    model = model_cls.from_config(cfg.model)
    
    if is_eval:
        model.eval()
    
    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()
    
    model = model.to(device)
    
    # 加载预处理器
    if cfg is not None and hasattr(cfg, 'preprocess'):
        preprocess_cfg = cfg.preprocess
        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
    
    return model, vis_processors, txt_processors


def clip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, 
                      **kwargs):
    """
    Fine-tune CLIP on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning leanring rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned CLIP model
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best CLIP model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """

    # 获取配置覆盖参数
    config_overrides = kwargs.get('config_overrides', None)
    
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    # 生成带有配置信息的训练路径名
    model_suffix = blip_model_name
    if config_overrides and 'semantic_model_type' in config_overrides:
        model_suffix += f"_{config_overrides['semantic_model_type']}"
    
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{model_suffix}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"save-memory-in: {save_memory}")
    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    # 加载模型 (支持配置覆盖)
    if config_overrides:
        blip_model, vis_processors, txt_processors = load_model_with_config(
            name=blip_model_name, 
            model_type=backbone, 
            is_eval=False, 
            device=device,
            config_overrides=config_overrides
        )
    else:
        blip_model, vis_processors, txt_processors = load_model_and_preprocess(
            name=blip_model_name, model_type=backbone, is_eval=False, device=device
        )
    
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
        #   'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay':0.05}])
        'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    # scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()


    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150, leave=False, 
                         dynamic_ncols=True, position=0, 
                         desc=f"Epoch {epoch}/{num_epochs}")
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors["eval"](caption) for caption in captions]
            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions})
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
        
        # 监控 Spatial Adapter 门控值
        if hasattr(blip_model, 'get_spatial_adapter_gate_value'):
            gate_value = blip_model.get_spatial_adapter_gate_value()
            if gate_value is not None:
                loss_log_dict['spatial_gate'] = gate_value
                print(f"[Epoch {epoch}] Spatial Adapter Gate Value: {gate_value:.6f}")
        
        # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                        idx_to_dress_mapping):
             
                index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model, save_memory)
                recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                                    index_features, index_names, txt_processors, save_memory)
                
                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)
                torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })

            print(json.dumps(results_dict, indent=4))
          
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)



def clip_finetune_cirr(num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, save_best: bool,
                       **kwargs):
    """
    Fine-tune CLIP on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param blip_model_name: BLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """
    rtc_weights = kwargs['loss_rtc']
    align_weights = kwargs['loss_align']
    
    # 获取配置覆盖参数
    config_overrides = kwargs.get('config_overrides', None)
    
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    # 生成带有配置信息的训练路径名
    model_suffix = blip_model_name
    if config_overrides and 'semantic_model_type' in config_overrides:
        model_suffix += f"_{config_overrides['semantic_model_type']}"
    
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_cirr_{model_suffix}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # 加载模型 (支持配置覆盖)
    if config_overrides:
        blip_model, vis_processors, txt_processors = load_model_with_config(
            name=blip_model_name, 
            model_type=backbone, 
            is_eval=False, 
            device=device,
            config_overrides=config_overrides
        )
    else:
        blip_model, vis_processors, txt_processors = load_model_and_preprocess(
            name=blip_model_name, model_type=backbone, is_eval=False, device=device
        )
    
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    # clip_model.eval().float()
    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs

    # Define the train dataset and the combining function
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)

    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    # debug 1 
    # val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)
    # # 
    # results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features,
    #                                     val_index_names, txt_processors)
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150, leave=False, 
                         dynamic_ncols=True, position=0, 
                         desc=f"Epoch {epoch}/{num_epochs}")
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            # print(scheduler.optimizer.param_groups[0]['lr'])
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions})
                loss = 0.
                for key in loss_dict.keys():
                    if key != 'loss_itc':
                        loss += kwargs[key] * loss_dict[key]
                    else:
                        loss += loss_dict[key]
            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)


        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
        
        # 监控 Spatial Adapter 门控值
        if hasattr(blip_model, 'get_spatial_adapter_gate_value'):
            gate_value = blip_model.get_spatial_adapter_gate_value()
            if gate_value is not None:
                loss_log_dict['spatial_gate'] = gate_value
                print(f"[Epoch {epoch}] Spatial Adapter Gate Value: {gate_value:.6f}")
        
        # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
                # extract target image features
            val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)
            # 
            results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features,
                                                val_index_names, txt_processors)
            group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

            results_dict = {
                'group_recall_at1': group_recall_at1,
                'group_recall_at2': group_recall_at2,
                'group_recall_at3': group_recall_at3,
                'recall_at1': recall_at1,
                'recall_at5': recall_at5,
                'recall_at10': recall_at10,
                'recall_at50': recall_at50,
                'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                'arithmetic_mean': mean(results),
                'harmonic_mean': harmonic_mean(results),
                'geometric_mean': geometric_mean(results)
            }
            print(json.dumps(results_dict, indent=4))
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # if save_training and epoch > 18:
            if save_training:
                if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                    best_arithmetic = results_dict['arithmetic_mean']
                    save_model('tuned_clip_arithmetic', epoch, blip_model, training_path)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--data-path", type=str, default="./cirr_dataset")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str, help="[blip2_cir_cat, blip2_cir, blip2_dual_stream_cir]")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--loss-align", default=0.4, type=float)
    parser.add_argument("--loss-rtc", default=0.4, type=float)
    parser.add_argument("--loss-itm", default=1, type=float)
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--save-memory", dest="save_memory", action='store_true',
                        help="Save only the best model during training")
    
    # ====================================
    # 双流架构配置参数 (用于 blip2_dual_stream_cir)
    # ====================================
    parser.add_argument("--semantic-model-type", type=str, default=None,
                        choices=['lightweight', 'blip2', 'blip2_gen', 'qwen2_vl'],
                        help="语义模型类型: 'lightweight' (简单), 'blip2' (attention), 'blip2_gen' (T5展开, 推荐), 'qwen2_vl' (最强但最重)")
    parser.add_argument("--qwen-model-name", type=str, default=None,
                        choices=['qwen2-vl-2b', 'qwen2-vl-7b', 'qwen2.5-vl-3b', 'qwen2.5-vl-7b'],
                        help="Qwen2-VL 模型名称 (仅 qwen2_vl 模式)")
    parser.add_argument("--t5-model-name", type=str, default=None,
                        choices=['flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'flan-t5-xl'],
                        help="T5 模型名称 (仅 blip2_gen 模式): small(80M), base(250M), large(780M)")
    parser.add_argument("--use-semantic-stream", type=str, default=None,
                        choices=['true', 'false'],
                        help="是否使用语义理解流")
    parser.add_argument("--use-fusion-module", type=str, default=None,
                        choices=['true', 'false'],
                        help="是否使用语义引导融合模块")
    parser.add_argument("--fusion-type", type=str, default=None,
                        choices=['cross_attention', 'gated', 'concat', 'add'],
                        help="融合类型")
    parser.add_argument("--freeze-vlm", type=str, default=None,
                        choices=['true', 'false'],
                        help="是否冻结 VLM 参数 (仅 qwen2_vl 模式)")
    parser.add_argument("--use-image-in-semantic", type=str, default=None,
                        choices=['true', 'false'],
                        help="BLIP2 语义模块是否使用图像引导 (仅 blip2 模式)")
    
    # ====================================
    # 2D-RoPE Spatial Adapter 配置参数 (用于 blip2_cir_align_prompt)
    # ====================================
    parser.add_argument("--use-spatial-adapter", type=str, default=None,
                        choices=['true', 'false'],
                        help="是否启用2D-RoPE空间感知并行分支")
    parser.add_argument("--spatial-adapter-hidden-dim", type=int, default=None,
                        help="Spatial Adapter隐藏层维度 (默认768)")
    parser.add_argument("--spatial-adapter-num-heads", type=int, default=None,
                        help="Spatial Adapter注意力头数 (默认12)")
    parser.add_argument("--spatial-adapter-depth", type=int, default=None,
                        help="Spatial Adapter Transformer层数 (默认2)")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")
    
    # 构建配置覆盖字典
    config_overrides = {}
    if args.semantic_model_type is not None:
        config_overrides['semantic_model_type'] = args.semantic_model_type
    if args.qwen_model_name is not None:
        config_overrides['qwen_model_name'] = args.qwen_model_name
    if args.t5_model_name is not None:
        config_overrides['t5_model_name'] = args.t5_model_name
    if args.use_semantic_stream is not None:
        config_overrides['use_semantic_stream'] = args.use_semantic_stream.lower() == 'true'
    if args.use_fusion_module is not None:
        config_overrides['use_fusion_module'] = args.use_fusion_module.lower() == 'true'
    if args.fusion_type is not None:
        config_overrides['fusion_type'] = args.fusion_type
    if args.freeze_vlm is not None:
        config_overrides['freeze_vlm'] = args.freeze_vlm.lower() == 'true'
    if args.use_image_in_semantic is not None:
        config_overrides['use_image_in_semantic'] = args.use_image_in_semantic.lower() == 'true'
    
    # 2D-RoPE Spatial Adapter 配置
    if args.use_spatial_adapter is not None:
        config_overrides['use_spatial_adapter'] = args.use_spatial_adapter.lower() == 'true'
    if args.spatial_adapter_hidden_dim is not None:
        config_overrides['spatial_adapter_hidden_dim'] = args.spatial_adapter_hidden_dim
    if args.spatial_adapter_num_heads is not None:
        config_overrides['spatial_adapter_num_heads'] = args.spatial_adapter_num_heads
    if args.spatial_adapter_depth is not None:
        config_overrides['spatial_adapter_depth'] = args.spatial_adapter_depth
    
    # 如果没有任何覆盖，设为 None
    if not config_overrides:
        config_overrides = None
    else:
        print(f"=== 模型配置覆盖 ===")
        for key, value in config_overrides.items():
            print(f"  {key}: {value}")
    
    print(f"save-memory: {args.save_memory}")
    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "data_path": args.data_path,
        "loss_rtc": args.loss_rtc,
        "loss_align": args.loss_align,
        "loss_itm": args.loss_itm,
        "save_memory": args.save_memory,
        "config_overrides": config_overrides,  # 添加配置覆盖
    }
    # set_seed(912)
    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        clip_finetune_fiq(**training_hyper_params)


