#!/usr/bin/env python3
"""
验证数据集中分数的排序
"""

import re

def parse_data_line(line):
    """解析数据行，提取所有人名和分数"""
    scores = []
    names = []
    
    # 提取第一个分数（没有名字的）
    first_score_pattern = r"^(\d+\.\d+),"
    first_match = re.match(first_score_pattern, line)
    if first_match:
        try:
            score = float(first_match.group(1))
            scores.append(score)
            names.append(None)  # 第一个分数没有名字
        except ValueError:
            pass
    
    # 提取所有 'Name': score 格式的数据
    # 匹配模式：'Name': 数字.数字
    name_score_pattern = r"'([^']+)':\s*(\d+\.\d+)"
    matches = re.findall(name_score_pattern, line)
    
    for name, score_str in matches:
        try:
            score = float(score_str)
            scores.append(score)
            names.append(name)
        except ValueError:
            continue
    
    return scores, names

def check_duplicate_names(names):
    """检查人名是否重复"""
    # 过滤掉 None（第一个没有名字的分数）
    valid_names = [name for name in names if name is not None]
    
    # 统计每个名字出现的次数
    name_count = {}
    for i, name in enumerate(names):
        if name is not None:
            if name not in name_count:
                name_count[name] = []
            name_count[name].append(i)
    
    # 找出重复的名字
    duplicates = {name: indices for name, indices in name_count.items() if len(indices) > 1}
    
    return duplicates, len(valid_names), len(set(valid_names))

def verify_sorting(file_path):
    """验证文件中的分数是否按降序排列，并检查人名是否重复"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line = f.read().strip()
        
        scores, names = parse_data_line(line)
        
        if not scores:
            print("错误：未能提取到任何分数")
            return False
        
        print(f"总共提取到 {len(scores)} 个分数")
        print(f"第一个分数: {scores[0]}")
        print(f"最后一个分数: {scores[-1]}")
        print()
        
        # 检查是否按降序排列
        is_descending = True
        violations = []
        
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                is_descending = False
                violations.append({
                    'index': i,
                    'current': scores[i],
                    'next': scores[i + 1],
                    'difference': scores[i + 1] - scores[i]
                })
        
        # 输出排序验证结果
        if is_descending:
            print("✓ 验证通过：所有分数按降序排列（从高到低）")
            print(f"  最高分: {scores[0]}")
            print(f"  最低分: {scores[-1]}")
            print(f"  分数范围: {scores[0] - scores[-1]}")
        else:
            print(f"✗ 验证失败：发现 {len(violations)} 处违反降序排列")
            print("\n前10处违反排序的位置：")
            for i, v in enumerate(violations[:10]):
                print(f"  位置 {v['index']}: {v['current']} < {v['next']} (差值: {v['difference']})")
        
        print()
        
        # 检查人名是否重复
        duplicates, total_names, unique_names = check_duplicate_names(names)
        
        if duplicates:
            print(f"✗ 发现重复人名：共有 {len(duplicates)} 个名字出现多次")
            print(f"  总人次数: {total_names}")
            print(f"  唯一人名数: {unique_names}")
            print(f"  重复人名数: {len(duplicates)}")
            print("\n重复的人名详情（前20个）：")
            for i, (name, indices) in enumerate(list(duplicates.items())[:20]):
                print(f"  \"{name}\": 出现 {len(indices)} 次，位置: {indices}")
                # 显示每个位置的分数
                for idx in indices:
                    print(f"    位置 {idx}: 分数 {scores[idx]}")
        else:
            print("✓ 验证通过：所有人名都是唯一的")
            print(f"  总人次数: {total_names}")
            print(f"  唯一人名数: {unique_names}")
        
        # 统计信息
        print("\n统计信息：")
        unique_scores_count = len(set(scores))
        print(f"  分数总数: {len(scores)}")
        print(f"  唯一分数数量: {unique_scores_count}")
        if unique_scores_count < len(scores):
            duplicates_scores = len(scores) - unique_scores_count
            print(f"  重复分数数量: {duplicates_scores}")
        
        return is_descending and len(duplicates) == 0
        
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
        return False
    except Exception as e:
        print(f"错误：{str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    # 如果提供了文件路径，使用它；否则使用默认路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/home/chency/GraphLLM/1.txt"
    
    verify_sorting(file_path)

