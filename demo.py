"""
建模比赛智能分析引擎 - 数据模块演示

运行方式: python demo.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)

from core.workspace_manager import get_workspace_manager, set_workspace_config
set_workspace_config(root_dir=str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from core.data_module import DataModule


def create_sample_data():
    """创建示例数据用于演示"""
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'user_id': range(10000, 11000),           # ID列
        'age': np.random.randint(18, 80, n),        # 数值型
        'gender': np.random.choice(['M', 'F', 'Unknown'], n),  # 类别型
        'income': np.random.lognormal(8.5, 0.5, n),  # 数值型（有偏态）
        'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n),  # 类别型
        'register_date': pd.date_range('2019-01-01', periods=n, freq='D'),  # 日期型
        'last_login': pd.date_range('2023-01-01', periods=n, freq='H'),   # 日期型
        'description': np.random.choice([
            '活跃用户，经常购买电子产品',
            '新注册用户，尚未完成首单',
            'VIP用户，月均消费超过5000元',
            '流失风险用户，建议发送优惠券召回',
            ''
        ], n),  # 文本型
        'is_vip': np.random.choice([0, 1], n),      # 布尔型
        'score': np.random.uniform(0, 100, n),       # 数值型
        'constant_col': 'fixed_value',               # 常量列
        'empty_col': np.nan,                         # 空列
        'target': np.random.randint(0, 2, n)         # 目标变量
    })
    
    # 添加缺失值
    missing_idx_age = np.random.choice(n, 50, replace=False)
    missing_idx_income = np.random.choice(n, 80, replace=False)
    missing_idx_city = np.random.choice(n, 30, replace=False)
    
    df.loc[missing_idx_age, 'age'] = np.nan
    df.loc[missing_idx_income, 'income'] = np.nan
    df.loc[missing_idx_city, 'city'] = np.nan
    
    # 添加异常值
    outlier_idx = np.random.choice(n, 20, replace=False)
    df.loc[outlier_idx, 'income'] = df['income'].max() * 10
    
    return df


def main():
    print("=" * 70)
    print("建模比赛智能分析引擎 - 数据模块演示".center(60))
    print("=" * 70)
    
    # 1. 创建示例数据
    print("\n[1/4] 创建示例数据...")
    df = create_sample_data()
    
    # 保存到工作目录内的临时文件（不占用C盘）
    wm = get_workspace_manager()
    temp_dir = wm.create_temp_dir(prefix='demo')
    data_path = os.path.join(temp_dir, 'sample_data.csv')
    df.to_csv(data_path, index=False, encoding='utf-8')
    print(f"  示例数据已保存: {data_path}")
    print(f"  数据形状: {df.shape}")
    
    # 2. 初始化数据模块并加载数据
    print("\n[2/4] 加载数据...")
    module = DataModule()
    module.load(data_path)
    print(f"  加载完成")
    
    # 3. 分析数据类型
    print("\n[3/4] 分析数据类型...")
    module.analyze()
    
    # 4. 清洗数据
    print("\n[4/4] 清洗数据...")
    module.clean(target_col='target')
    
    # 打印完整报告
    module.print_report()
    
    # 保存报告（自动路由到 workspace/reports/）
    report_path = 'data_report.json'
    saved = module.save_report(report_path)
    if saved:
        print(f"\n详细报告已保存: {saved}")
    
    # 展示清洗前后对比
    print("\n" + "=" * 70)
    print("清洗效果对比".center(60))
    print("=" * 70)
    print(f"\n  原始数据: {module.raw_data.shape}")
    print(f"  清洗后:   {module.cleaned_data.shape}")
    print(f"\n  原始内存: {module.raw_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"  清洗后内存: {module.cleaned_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # 展示各列分析结果
    print("\n" + "=" * 70)
    print("各列分析结果".center(60))
    print("=" * 70)
    for col, profile in module.profiles.items():
        print(f"\n  {col:15s} -> {profile.inferred_type.value:10s} "
              f"(缺失: {profile.null_rate:.1%}, "
              f"唯一值: {profile.unique_count})")
    
    # 清理临时文件（WorkspaceManager 退出时自动清理）
    # shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "=" * 70)
    print("演示完成！".center(60))
    print("=" * 70)


if __name__ == '__main__':
    main()
