"""
文本扰动可视化演示
==================
展示各种文本扰动的效果，使用真实数据集中的文本

Usage:
    python demo_text_perturbation.py
    python demo_text_perturbation.py --dataset cirr --num-samples 10
    python demo_text_perturbation.py --output html
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from robustness.text_perturbation import TextPerturber, list_perturbations, HAS_NLPAUG


# 项目根目录
base_path = Path(__file__).absolute().parents[2].absolute()


def load_cirr_captions(split: str = 'val', num_samples: int = 10) -> List[str]:
    """
    从 CIRR 数据集加载 caption
    
    Args:
        split: 数据集划分 ('train', 'val', 'test1')
        num_samples: 加载的样本数量
        
    Returns:
        caption 列表
    """
    caption_path = base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json'
    
    if not caption_path.exists():
        print(f"Warning: {caption_path} not found")
        return get_fallback_captions()
    
    with open(caption_path, 'r') as f:
        data = json.load(f)
    
    captions = [item['caption'] for item in data[:num_samples]]
    return captions


def load_fashioniq_captions(split: str = 'val', dress_type: str = 'dress', num_samples: int = 10) -> List[str]:
    """
    从 FashionIQ 数据集加载 caption
    
    Args:
        split: 数据集划分
        dress_type: 服装类型
        num_samples: 加载的样本数量
        
    Returns:
        caption 列表 (合并两个 caption 为一个)
    """
    caption_path = base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json'
    
    if not caption_path.exists():
        print(f"Warning: {caption_path} not found")
        return get_fallback_captions()
    
    with open(caption_path, 'r') as f:
        data = json.load(f)
    
    captions = []
    for item in data[:num_samples]:
        # FashionIQ 的 caption 是列表，合并为一个字符串
        cap = ' and '.join(item['captions'])
        captions.append(cap)
    
    return captions


def get_fallback_captions() -> List[str]:
    """返回备用的测试 caption"""
    return [
        "make it more colorful and add some patterns",
        "the dress should be shorter with a red color",
        "change the background to a beach scene",
        "show three bottles of soft drink",
        "has more people in a different setting",
        "is a closer view of the same scene",
        "replace the cat with a dog",
        "make the sky blue instead of cloudy",
        "add a hat to the person",
        "change the season to winter with snow",
    ]


def demo_single_perturbation(
    perturber: TextPerturber,
    captions: List[str],
    perturbation_type: str,
    severities: List[int] = [1, 3, 5]
) -> List[Dict[str, Any]]:
    """
    演示单一扰动类型在不同严重度下的效果
    
    Returns:
        结果列表
    """
    results = []
    
    for caption in captions:
        result = {
            'original': caption,
            'perturbation': perturbation_type,
            'results': {}
        }
        
        for severity in severities:
            try:
                perturbed = perturber.apply(caption, perturbation_type, severity)
                result['results'][f'severity_{severity}'] = perturbed
            except Exception as e:
                result['results'][f'severity_{severity}'] = f"ERROR: {e}"
        
        results.append(result)
    
    return results


def print_results_table(results: List[Dict[str, Any]], title: str = ""):
    """打印结果表格"""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
    
    for result in results:
        print(f"\n原文: {result['original']}")
        print(f"扰动类型: {result['perturbation']}")
        print("-" * 60)
        
        for key, value in result['results'].items():
            severity = key.replace('severity_', 'S')
            # 高亮显示变化的部分
            print(f"  {severity}: {value}")


def generate_html_report(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    captions: List[str]
):
    """
    生成 HTML 报告
    
    Args:
        all_results: {perturbation_type: results}
        output_path: 输出文件路径
        captions: 原始 caption 列表
    """
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Text Perturbation Demo</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4CAF50;
            margin-top: 30px;
        }}
        .perturbation-section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .perturbation-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .caption-item {{
            margin: 15px 0;
            padding: 15px;
            background: #fafafa;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .original {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .severity {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            margin-right: 10px;
        }}
        .s1 {{ background: #E8F5E9; color: #2E7D32; }}
        .s3 {{ background: #FFF3E0; color: #E65100; }}
        .s5 {{ background: #FFEBEE; color: #C62828; }}
        .perturbed-text {{
            color: #555;
            margin: 5px 0 5px 30px;
        }}
        .error {{
            color: #C62828;
            font-style: italic;
        }}
        .summary {{
            background: #E3F2FD;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .highlight {{
            background-color: #FFEB3B;
            padding: 0 2px;
        }}
    </style>
</head>
<body>
    <h1>🔤 Text Perturbation Demo (nlpaug)</h1>
    
    <div class="summary">
        <h3>📊 Summary</h3>
        <p><strong>Total Captions:</strong> {num_captions}</p>
        <p><strong>Perturbation Types:</strong> {num_perturbations}</p>
        <p><strong>Severity Levels:</strong> 1 (Light), 3 (Medium), 5 (Heavy)</p>
    </div>
    
    <h2>📝 Original Captions</h2>
    <ol>
        {caption_list}
    </ol>
    
    <h2>🔄 Perturbation Results</h2>
    {perturbation_sections}
    
    <footer style="text-align: center; margin-top: 40px; color: #888;">
        Generated by text_perturbation.py using nlpaug
    </footer>
</body>
</html>"""
    
    # 生成 caption 列表
    caption_list = "\n".join([f"<li>{cap}</li>" for cap in captions])
    
    # 生成扰动部分
    perturbation_sections = ""
    for pert_type, results in all_results.items():
        info = TextPerturber.PERTURBATION_REGISTRY.get(pert_type, None)
        description = info.description if info else ""
        
        section = f"""
        <div class="perturbation-section">
            <div class="perturbation-title">
                {pert_type} - {description}
            </div>
        """
        
        for result in results:
            section += f"""
            <div class="caption-item">
                <div class="original">Original: {result['original']}</div>
            """
            
            for key, value in result['results'].items():
                severity_num = key.replace('severity_', '')
                severity_class = f"s{severity_num}"
                
                if value.startswith("ERROR"):
                    section += f"""
                    <div>
                        <span class="severity {severity_class}">S{severity_num}</span>
                        <span class="perturbed-text error">{value}</span>
                    </div>
                    """
                else:
                    section += f"""
                    <div>
                        <span class="severity {severity_class}">S{severity_num}</span>
                        <span class="perturbed-text">{value}</span>
                    </div>
                    """
            
            section += "</div>"
        
        section += "</div>"
        perturbation_sections += section
    
    # 填充模板
    html_content = html_template.format(
        num_captions=len(captions),
        num_perturbations=len(all_results),
        caption_list=caption_list,
        perturbation_sections=perturbation_sections
    )
    
    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Text Perturbation Demo')
    parser.add_argument('--dataset', type=str, default='cirr', choices=['cirr', 'fashioniq'],
                        help='Dataset to use')
    parser.add_argument('--split', type=str, default='val', help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--output', type=str, default='html', choices=['console', 'html', 'both'],
                        help='Output format')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for HTML report')
    parser.add_argument('--perturbations', type=str, nargs='+', default=None,
                        help='Specific perturbation types to demo')
    args = parser.parse_args()
    
    # 检查 nlpaug
    if not HAS_NLPAUG:
        print("ERROR: nlpaug is not installed.")
        print("Please install it with: pip install nlpaug")
        return
    
    # 打印所有扰动类型
    print("\n" + "="*60)
    print("Available Perturbation Types:")
    print("="*60)
    list_perturbations()
    
    # 加载 caption
    print(f"\n\nLoading captions from {args.dataset} dataset...")
    if args.dataset == 'cirr':
        captions = load_cirr_captions(args.split, args.num_samples)
    else:
        captions = load_fashioniq_captions(args.split, 'dress', args.num_samples)
    
    print(f"Loaded {len(captions)} captions")
    
    # 初始化扰动器
    perturber = TextPerturber(seed=42)
    
    # 选择要演示的扰动类型
    if args.perturbations:
        perturbation_types = args.perturbations
    else:
        # 默认演示基础扰动（避免需要额外资源的）
        perturbation_types = [
            'keyboard_typo',
            'ocr_error', 
            'char_swap',
            'char_delete',
            'synonym_replace',
            'antonym_replace',
            'word_delete',
            'word_swap',
            'spelling_error',
            'case_change',
        ]
    
    # 运行演示
    all_results = {}
    severities = [1, 3, 5]
    
    print(f"\n\nRunning perturbation demo...")
    print(f"Perturbation types: {perturbation_types}")
    print(f"Severities: {severities}")
    
    for pert_type in perturbation_types:
        print(f"  Processing {pert_type}...", end=" ")
        try:
            results = demo_single_perturbation(perturber, captions, pert_type, severities)
            all_results[pert_type] = results
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # 输出结果
    if args.output in ['console', 'both']:
        for pert_type, results in all_results.items():
            info = TextPerturber.PERTURBATION_REGISTRY.get(pert_type, None)
            title = f"{pert_type}: {info.description if info else ''}"
            print_results_table(results, title)
    
    if args.output in ['html', 'both']:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = base_path / 'visualizations' / 'text_perturbation_demo'
        
        output_path = output_dir / f'text_perturbation_demo_{args.dataset}.html'
        generate_html_report(all_results, output_path, captions)
    
    print("\n✅ Demo completed!")


if __name__ == '__main__':
    main()

