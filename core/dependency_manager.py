"""
可选依赖管理器

检测项目可选依赖的缺失状态，并提供安装到项目目录的功能。
"""

import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils.helpers import log_info, log_warning, log_error

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_THIRD_PARTY_DIR = PROJECT_ROOT / 'third_party'

# 可选依赖白名单
OPTIONAL_DEPENDENCIES = {
    'xgboost': {
        'name': 'XGBoost',
        'description': '高性能梯度提升框架',
        'pip_name': 'xgboost',
        'module': 'xgboost',
    },
    'lightgbm': {
        'name': 'LightGBM',
        'description': '微软轻量级梯度提升框架',
        'pip_name': 'lightgbm',
        'module': 'lightgbm',
    },
    'catboost': {
        'name': 'CatBoost',
        'description': 'Yandex 类别特征梯度提升框架',
        'pip_name': 'catboost',
        'module': 'catboost',
    },
    'prophet': {
        'name': 'Prophet',
        'description': 'Facebook 时间序列预测库',
        'pip_name': 'prophet',
        'module': 'prophet',
    },
    'plotly': {
        'name': 'Plotly',
        'description': '交互式可视化图表库',
        'pip_name': 'plotly',
        'module': 'plotly',
    },
}


def _ensure_path(target_dir: Path):
    """确保指定目录在 sys.path 中"""
    path_str = str(target_dir.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def check_package(package_key: str, target_dir: Path = None) -> bool:
    """检查某个可选依赖是否已安装"""
    if target_dir:
        _ensure_path(target_dir)
    info = OPTIONAL_DEPENDENCIES.get(package_key)
    if not info:
        return False
    try:
        importlib.import_module(info['module'])
        return True
    except ImportError:
        return False


def get_missing_dependencies(target_dir: Path = None) -> List[Dict]:
    """获取所有缺失的可选依赖列表"""
    missing = []
    for key, info in OPTIONAL_DEPENDENCIES.items():
        installed = check_package(key, target_dir)
        missing.append({
            'key': key,
            'name': info['name'],
            'description': info['description'],
            'pip_name': info['pip_name'],
            'installed': installed,
        })
    return missing


def install_dependency(package_key: str, target_dir: Path = None) -> Tuple[bool, str, str]:
    """
    安装指定依赖到指定目录（默认项目 third_party）
    
    Returns:
        (success, stdout, stderr)
    """
    info = OPTIONAL_DEPENDENCIES.get(package_key)
    if not info:
        return False, '', f'未知依赖: {package_key}'
    
    target = target_dir or DEFAULT_THIRD_PARTY_DIR
    target.mkdir(parents=True, exist_ok=True)
    pip_name = info['pip_name']
    
    log_info(f'开始安装 {pip_name} 到 {target}', category='DependencyManager')
    
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        pip_name,
        '--target', str(target),
        '--upgrade',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            encoding='utf-8',
            errors='replace',
        )
        success = result.returncode == 0
        if success:
            log_info(f'{pip_name} 安装成功', category='DependencyManager')
            _ensure_path(target)
        else:
            log_warning(f'{pip_name} 安装失败: {result.stderr[:500]}', category='DependencyManager')
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        log_error(f'{pip_name} 安装超时', category='DependencyManager')
        return False, '', '安装超时（超过10分钟）'
    except Exception as e:
        log_error(f'{pip_name} 安装异常: {e}', category='DependencyManager')
        return False, '', str(e)


def install_all_missing(target_dir: Path = None) -> Dict[str, Tuple[bool, str, str]]:
    """安装所有缺失的依赖"""
    results = {}
    missing = [d for d in get_missing_dependencies(target_dir) if not d['installed']]
    for dep in missing:
        results[dep['key']] = install_dependency(dep['key'], target_dir)
    return results


# 启动时自动将默认目录加入路径
_ensure_path(DEFAULT_THIRD_PARTY_DIR)
