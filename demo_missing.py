"""
缺失值智能分析引擎 - 演示脚本

运行方式: python demo_missing.py
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)

from core.workspace_manager import get_workspace_manager, set_workspace_config
set_workspace_config(root_dir=str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from core.auto_pipeline import AutoMissingPipeline, PipelineConfig
from core.missing_engine import MissingPattern, export_missing_report


def create_demo_data():
    """
    创建包含多种缺失模式的演示数据
    """
    np.random.seed(42)
    n = 500
    n_train = 350
    
    df = pd.DataFrame({
        'user_id': range(10000, 10000 + n),
        'age': np.random.randint(18, 65, n),
        'gender': np.random.choice(['M', 'F'], n),
        'income': np.random.lognormal(8.5, 0.5, n),
        'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n),
        'married': np.random.choice(['是', '否'], n, p=[0.6, 0.4]),
        'has_child': np.nan,
        'spouse_income': np.nan,
        'child_education': np.nan,
        'last_login': pd.date_range('2023-01-01', periods=n, freq='H'),
        'credit_score': np.random.uniform(300, 850, n),
        'comment': np.random.choice([
            '体验很好', '需要改进', '', '一般般', '非常满意', ''
        ], n),
        'almost_empty': np.nan,
        'target': np.nan
    })
    
    # ========== 1. 目标缺失：训练集有标签，测试集无标签 ==========
    df.loc[:n_train-1, 'target'] = np.random.choice([0, 1], n_train)
    
    # ========== 2. 结构性缺失：婚姻-配偶收入 ==========
    for i in range(n):
        if df.loc[i, 'married'] == '是':
            df.loc[i, 'spouse_income'] = np.random.lognormal(8.3, 0.4)
            df.loc[i, 'has_child'] = np.random.choice(['是', '否'], p=[0.7, 0.3])
        else:
            df.loc[i, 'spouse_income'] = np.nan
            df.loc[i, 'has_child'] = '否'
    
    # ========== 3. 结构性缺失：有子女-教育支出 ==========
    for i in range(n):
        if df.loc[i, 'has_child'] == '是':
            df.loc[i, 'child_education'] = np.random.lognormal(9, 0.3)
        else:
            df.loc[i, 'child_education'] = np.nan
    
    # ========== 4. 真缺失：随机缺失 ==========
    # 年龄随机缺失（用户没填）
    missing_age = np.random.choice(n, 25, replace=False)
    df.loc[missing_age, 'age'] = np.nan
    
    # 收入随机缺失（采集失败）
    missing_income = np.random.choice(n, 30, replace=False)
    df.loc[missing_income, 'income'] = np.nan
    
    # 信用分随机缺失
    missing_credit = np.random.choice(n, 40, replace=False)
    df.loc[missing_credit, 'credit_score'] = np.nan
    
    # ========== 5. 空列：几乎全空 ==========
    df.loc[np.random.choice(n, 3, replace=False), 'almost_empty'] = 'xxx'
    
    return df


def main():
    print("=" * 70)
    print("  Modeling Competition Intelligence Engine".center(60))
    print("  Missing Value Analysis Demo".center(60))
    print("=" * 70)
    
    # 1. 创建数据
    print("\n[1/5] Creating demo dataset with multiple missing patterns...")
    df = create_demo_data()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Missing overview:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"    {col:20s}: {missing:3d} ({missing/len(df):.1%})")
    
    # 2. 运行自动流程（标准模式）
    print("\n[2/5] Running auto pipeline (standard mode)...")
    config = PipelineConfig(
        structural_threshold=0.85,
        drop_col_threshold=0.95
    )
    pipeline = AutoMissingPipeline(config)
    train_df, test_df, report = pipeline.run(df)
    
    # 3. 打印报告
    print("\n[3/5] Analysis report:")
    pipeline.print_report()
    
    # 4. 验证处理结果
    print("\n[4/5] Processing results verification:")
    print(f"  Train set: {train_df.shape}")
    print(f"  Test set:  {test_df.shape if test_df is not None else None}")
    
    # 检查各列处理效果
    print(f"\n  Column processing status:")
    for col in ['age', 'income', 'spouse_income', 'child_education', 
                'credit_score', 'comment', 'almost_empty', 'target']:
        if col in train_df.columns:
            missing = train_df[col].isnull().sum()
            print(f"    {col:20s}: {missing} missing in train")
        else:
            print(f"    {col:20s}: DROPPED")
    
    # 5. 导出报告（自动保存到 workspace/reports/，不占用C盘）
    print("\n[5/5] Exporting report...")
    wm = get_workspace_manager()
    report_path = 'missing_report.json'
    saved_path = export_missing_report(report, report_path)
    
    # 显示关键洞察
    print(f"\n  Key insights:")
    for col, profile in report.column_profiles.items():
        if profile.pattern == MissingPattern.STRUCTURAL:
            print(f"    [Structural] {col}")
            for rule in profile.structural_rules[:1]:
                print(f"      -> When {rule.condition_col}={rule.condition_value}, missing rate={rule.confidence:.1%}")
        elif profile.pattern == MissingPattern.TARGET_MISSING:
            print(f"    [Target] {col}: {profile.missing_count} values to predict")
        elif profile.pattern == MissingPattern.TRUE_MISSING:
            print(f"    [True Missing] {col}: strategy={profile.recommended_strategy.value}")
    
    if saved_path:
        print(f"\n  Report saved to: {saved_path}")
    
    # 清理（WorkspaceManager 退出时自动清理 temp）
    
    print("\n" + "=" * 70)
    print("  Demo completed!".center(60))
    print("=" * 70)


if __name__ == '__main__':
    main()
