#!/usr/bin/env python3
"""
AMLSim数据加载器测试脚本
用于验证数据加载器的基本功能
"""

import sys
import os
import pandas as pd
from load_data_into_nebulagraph import AMLSimDataLoader

def test_data_files():
    """测试数据文件是否存在且可读"""
    print("测试数据文件...")
    
    loader = AMLSimDataLoader()
    
    # 检查账户文件
    if os.path.exists(loader.accounts_file):
        try:
            accounts_df = pd.read_csv(loader.accounts_file)
            print(f"✓ 账户文件存在，包含 {len(accounts_df)} 条记录")
            print(f"  列名: {list(accounts_df.columns)}")
        except Exception as e:
            print(f"✗ 读取账户文件失败: {e}")
            return False
    else:
        print(f"✗ 账户文件不存在: {loader.accounts_file}")
        return False
    
    # 检查交易文件
    if os.path.exists(loader.transactions_file):
        try:
            transactions_df = pd.read_csv(loader.transactions_file)
            print(f"✓ 交易文件存在，包含 {len(transactions_df)} 条记录")
            print(f"  列名: {list(transactions_df.columns)}")
        except Exception as e:
            print(f"✗ 读取交易文件失败: {e}")
            return False
    else:
        print(f"✗ 交易文件不存在: {loader.transactions_file}")
        return False
    
    return True

def test_data_structure():
    """测试数据结构"""
    print("\n测试数据结构...")
    
    loader = AMLSimDataLoader()
    
    try:
        # 读取数据
        accounts_df = pd.read_csv(loader.accounts_file)
        transactions_df = pd.read_csv(loader.transactions_file)
        
        # 检查必要的列是否存在
        required_account_cols = ['acct_id']
        required_transaction_cols = ['orig_acct', 'bene_acct']
        
        for col in required_account_cols:
            if col in accounts_df.columns:
                print(f"✓ 账户文件包含必要列: {col}")
            else:
                print(f"✗ 账户文件缺少必要列: {col}")
                return False
        
        for col in required_transaction_cols:
            if col in transactions_df.columns:
                print(f"✓ 交易文件包含必要列: {col}")
            else:
                print(f"✗ 交易文件缺少必要列: {col}")
                return False
        
        # 检查数据类型
        print(f"✓ 账户ID类型: {accounts_df['acct_id'].dtype}")
        print(f"✓ 源账户类型: {transactions_df['orig_acct'].dtype}")
        print(f"✓ 目标账户类型: {transactions_df['bene_acct'].dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试数据结构时发生错误: {e}")
        return False

def test_loader_initialization():
    """测试加载器初始化"""
    print("\n测试加载器初始化...")
    
    try:
        loader = AMLSimDataLoader()
        print("✓ 加载器初始化成功")
        print(f"  空间名称: {loader.space_name}")
        print(f"  账户文件: {loader.accounts_file}")
        print(f"  交易文件: {loader.transactions_file}")
        return True
    except Exception as e:
        print(f"✗ 加载器初始化失败: {e}")
        return False

def main():
    """主测试函数"""
    print("AMLSim数据加载器测试")
    print("=" * 50)
    
    tests = [
        ("数据文件测试", test_data_files),
        ("数据结构测试", test_data_structure),
        ("加载器初始化测试", test_loader_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} 通过")
        else:
            print(f"✗ {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！可以运行数据加载器。")
        return 0
    else:
        print("✗ 部分测试失败，请检查配置。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 