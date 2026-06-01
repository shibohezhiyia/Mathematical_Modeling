"""
工作空间管理器

核心职责：
1. 统一管理所有磁盘IO，确保所有文件操作限制在项目工作目录内
2. 提供 allow_disk_write 开关，关闭时禁止一切磁盘写入
3. 自动拦截并重定向C盘路径到工作目录
4. 隔离临时文件、缓存、报告到 workspace/ 子目录

使用方式：
    from core.workspace_manager import get_workspace_manager, WorkspaceManager
    
    # 获取全局管理器（默认允许写入，根目录为当前工作目录）
    wm = get_workspace_manager()
    
    # 创建临时目录（自动在项目目录内）
    temp = wm.create_temp_dir()
    
    # 安全写入文件（C盘路径自动重定向）
    safe_path = wm.safe_path('report.json')
    with open(safe_path, 'w') as f:
        f.write(data)
    
    # 关闭磁盘写入
    wm.set_allow_disk_write(False)
"""

import os
import sys
import uuid
import shutil
import atexit
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

import pandas as pd

from utils.helpers import log_info, log_warning, log_error


class WorkspaceManager:
    """
    工作空间管理器
    
    所有磁盘操作必须经过此管理器，确保：
    - 不写入系统盘（C盘）
    - 不写入用户不知情的位置
    - 提供明确的开关控制
    """
    
    def __init__(self,
                 root_dir: Optional[str] = None,
                 allow_disk_write: bool = True) -> None:
        """
        Args:
            root_dir: 工作根目录，默认当前工作目录
            allow_disk_write: 是否允许磁盘写入
        """
        self._allow_disk_write = allow_disk_write
        
        # 确定工作根目录（绝对路径）
        if root_dir:
            self.root_dir = os.path.abspath(root_dir)
        else:
            self.root_dir = os.path.abspath(os.getcwd())
        
        # 工作空间子目录（全部在项目目录内）
        self.workspace_dir = os.path.join(self.root_dir, 'data')
        self.temp_dir = os.path.join(self.workspace_dir, 'temp')
        self.cache_dir = os.path.join(self.workspace_dir, 'cache')
        self.report_dir = os.path.join(self.workspace_dir, 'reports')
        self.data_dir = os.path.join(self.workspace_dir, 'raw')
        
        # 注册清理钩子
        atexit.register(self._cleanup_on_exit)
        
        if self._allow_disk_write:
            self._ensure_dirs()
            log_info(f"[Workspace] 工作空间初始化: {self.workspace_dir}")
        else:
            log_info("[Workspace] 工作空间初始化（磁盘写入已禁用）")
    
    # ------------------------------------------------------------------
    # 开关控制
    # ------------------------------------------------------------------
    
    @property
    def allow_disk_write(self) -> bool:
        return self._allow_disk_write
    
    def set_allow_disk_write(self, enabled: bool) -> None:
        """动态开关磁盘写入权限"""
        old = self._allow_disk_write
        self._allow_disk_write = enabled
        if enabled and not old:
            self._ensure_dirs()
            log_info("[Workspace] 磁盘写入已启用")
        elif not enabled and old:
            log_info("[Workspace] 磁盘写入已禁用")
    
    def check_permission(self, operation: str = "写入") -> bool:
        """检查是否有磁盘写入权限"""
        if not self._allow_disk_write:
            log_warning(f"[Workspace] 磁盘{operation}被拒绝（allow_disk_write=False）")
            return False
        return True
    
    # ------------------------------------------------------------------
    # 路径安全
    # ------------------------------------------------------------------
    
    def safe_path(self, path: str, subdir: str = 'reports') -> str:
        """
        将任意路径转换为工作目录内的安全路径
        
        规则：
        1. 相对路径 → 放入 workspace/<subdir>/
        2. C盘绝对路径 → 提取文件名，放入 workspace/<subdir>/
        3. 其他盘绝对路径 → 保持原样（但会警告）
        4. 已在工作目录内的路径 → 保持原样
        
        Args:
            path: 原始路径
            subdir: 目标子目录（reports/temp/cache/data）
            
        Returns:
            安全路径（绝对路径）
        """
        if not path:
            return path
        
        # 相对路径 → 直接放入目标子目录
        if not os.path.isabs(path):
            target_base = {
                'reports': self.report_dir,
                'temp': self.temp_dir,
                'cache': self.cache_dir,
                'data': self.data_dir,
            }.get(subdir, self.report_dir)
            safe = os.path.normpath(os.path.join(target_base, path))
            return safe
        
        abs_path = os.path.abspath(path)
        
        # 已经在工作目录内 → 保持
        if self._is_under_workspace(abs_path):
            return abs_path
        
        # C盘路径（Windows）→ 强制重定向
        if self._is_system_drive(abs_path):
            target_base = {
                'reports': self.report_dir,
                'temp': self.temp_dir,
                'cache': self.cache_dir,
                'data': self.data_dir,
            }.get(subdir, self.report_dir)
            filename = os.path.basename(abs_path)
            safe = os.path.join(target_base, filename)
            log_warning(
                f"[Workspace] 检测到C盘路径，已重定向: "
                f"{path} → {safe}"
            )
            return safe
        
        # 其他绝对路径（D盘等）→ 保持但警告
        log_warning(
            f"[Workspace] 使用非工作目录路径: {abs_path}，"
            f"建议将数据放入项目目录"
        )
        return abs_path
    
    def _is_system_drive(self, path: str) -> bool:
        """检查是否为系统盘（Windows C盘 / Linux / 等）"""
        if os.name == 'nt':
            # Windows: 检查是否为 C:\
            drive = os.path.splitdrive(path)[0].upper()
            return drive == 'C:'
        else:
            # Linux/macOS: 检查系统目录
            system_paths = ['/usr', '/bin', '/sbin', '/lib', '/etc', '/var', '/tmp']
            path_norm = os.path.normpath(path)
            return any(path_norm.startswith(sp) for sp in system_paths)
    
    def _is_under_workspace(self, path: str) -> bool:
        """检查路径是否已在工作目录内"""
        try:
            Path(path).relative_to(self.root_dir)
            return True
        except ValueError:
            return False
    
    # ------------------------------------------------------------------
    # 目录操作
    # ------------------------------------------------------------------
    
    def create_temp_dir(self, prefix: str = 'tmp') -> str:
        """创建工作目录内的临时目录（替代 tempfile.mkdtemp）"""
        if not self.check_permission("创建临时目录"):
            raise PermissionError(
                "磁盘写入已禁用，无法创建临时目录。"
                "请设置 allow_disk_write=True 或手动管理内存数据"
            )
        
        name = f"{prefix}_{uuid.uuid4().hex[:8]}"
        d = os.path.join(self.temp_dir, name)
        os.makedirs(d, exist_ok=True)
        log_info(f"[Workspace] 创建临时目录: {d}")
        return d
    
    def create_cache_dir(self, name: str = '') -> str:
        """创建缓存目录"""
        if not self.check_permission("创建缓存目录"):
            raise PermissionError("磁盘写入已禁用")
        
        d = os.path.join(self.cache_dir, name) if name else self.cache_dir
        os.makedirs(d, exist_ok=True)
        return d
    
    def get_data_path(self, filename: str) -> str:
        """获取数据目录内的文件路径"""
        return os.path.join(self.data_dir, filename)
    
    def get_report_path(self, filename: str) -> str:
        """获取报告目录内的文件路径"""
        return os.path.join(self.report_dir, filename)
    
    # ------------------------------------------------------------------
    # 文件写入包装
    # ------------------------------------------------------------------
    
    def write_text(self, path: str, content: str, subdir: str = 'reports') -> Optional[str]:
        """安全写入文本文件"""
        if not self.check_permission("写入"):
            return None
        
        safe_path = self.safe_path(path, subdir)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        log_info(f"[Workspace] 写入文件: {safe_path}")
        return safe_path
    
    def write_bytes(self, path: str, content: bytes, subdir: str = 'reports') -> Optional[str]:
        """安全写入二进制文件"""
        if not self.check_permission("写入"):
            return None
        
        safe_path = self.safe_path(path, subdir)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        with open(safe_path, 'wb') as f:
            f.write(content)
        
        return safe_path
    
    def save_dataframe(self, df: pd.DataFrame, path: str, subdir: str = 'reports', **kwargs) -> Optional[str]:
        """安全保存DataFrame"""
        if not self.check_permission("保存"):
            return None
        
        safe_path = self.safe_path(path, subdir)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        ext = os.path.splitext(safe_path)[1].lower()
        if ext == '.csv':
            df.to_csv(safe_path, index=False, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(safe_path, index=False, **kwargs)
        elif ext == '.parquet':
            df.to_parquet(safe_path, **kwargs)
        elif ext == '.json':
            df.to_json(safe_path, **kwargs)
        else:
            df.to_csv(safe_path, index=False, **kwargs)
        
        log_info(f"[Workspace] 保存DataFrame: {safe_path}")
        return safe_path
    
    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    
    def clear_temp(self) -> None:
        """清空临时目录"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            os.makedirs(self.temp_dir, exist_ok=True)
            log_info("[Workspace] 临时目录已清空")
    
    def clear_cache(self) -> None:
        """清空缓存目录"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            log_info("[Workspace] 缓存目录已清空")
    
    def clear_all(self) -> None:
        """清空整个工作空间"""
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
            if self._allow_disk_write:
                self._ensure_dirs()
            log_info("[Workspace] 工作空间已清空")
    
    def _ensure_dirs(self) -> None:
        """确保所有子目录存在"""
        for d in [self.workspace_dir, self.temp_dir, self.cache_dir, 
                  self.report_dir, self.data_dir]:
            os.makedirs(d, exist_ok=True)
    
    def _cleanup_on_exit(self) -> None:
        """退出时自动清理临时目录"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass
    
    # ------------------------------------------------------------------
    # 信息
    # ------------------------------------------------------------------
    
    def get_info(self) -> Dict[str, Any]:
        """获取工作空间信息"""
        info = {
            'root_dir': self.root_dir,
            'allow_disk_write': self._allow_disk_write,
            'workspace_dir': self.workspace_dir,
        }
        
        for name, path in [
            ('temp', self.temp_dir),
            ('cache', self.cache_dir),
            ('reports', self.report_dir),
            ('data', self.data_dir),
        ]:
            if os.path.exists(path):
                size = self._get_dir_size(path)
                files = sum(1 for _, _, files in os.walk(path) for _ in files)
                info[f'{name}_size_mb'] = round(size / (1024 * 1024), 2)
                info[f'{name}_files'] = files
            else:
                info[f'{name}_size_mb'] = 0
                info[f'{name}_files'] = 0
        
        return info
    
    def _get_dir_size(self, path: str) -> int:
        """计算目录大小（字节）"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except:
                    pass
        return total
    
    def print_info(self) -> None:
        """打印工作空间信息"""
        info = self.get_info()
        print("=" * 50)
        print("工作空间信息".center(40))
        print("=" * 50)
        print(f"根目录: {info['root_dir']}")
        print(f"磁盘写入: {'允许' if info['allow_disk_write'] else '禁止'}")
        print(f"工作空间: {info['workspace_dir']}")
        for key in ['temp', 'cache', 'reports', 'data']:
            size = info.get(f'{key}_size_mb', 0)
            files = info.get(f'{key}_files', 0)
            print(f"  {key:8s}: {size:8.2f} MB ({files} 文件)")
        print("=" * 50)


# =============================================================================
# 全局单例
# =============================================================================

_workspace_manager: Optional[WorkspaceManager] = None


def get_workspace_manager(
    root_dir: Optional[str] = None,
    allow_disk_write: bool = True,
    force_new: bool = False
) -> WorkspaceManager:
    """
    获取全局 WorkspaceManager 实例
    
    Args:
        root_dir: 工作根目录
        allow_disk_write: 是否允许磁盘写入
        force_new: 是否强制创建新实例
        
    Returns:
        WorkspaceManager 实例
    """
    global _workspace_manager
    if _workspace_manager is None or force_new:
        _workspace_manager = WorkspaceManager(
            root_dir=root_dir,
            allow_disk_write=allow_disk_write
        )
    return _workspace_manager


def set_workspace_config(
    root_dir: Optional[str] = None,
    allow_disk_write: Optional[bool] = None
) -> None:
    """
    修改全局工作空间配置
    
    示例：
        # 关闭磁盘写入（纯内存模式）
        set_workspace_config(allow_disk_write=False)
        
        # 更改工作目录
        set_workspace_config(root_dir='D:/my_project')
    """
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager(
            root_dir=root_dir,
            allow_disk_write=allow_disk_write if allow_disk_write is not None else True
        )
    else:
        if root_dir is not None:
            _workspace_manager.root_dir = os.path.abspath(root_dir)
            _workspace_manager.workspace_dir = os.path.join(_workspace_manager.root_dir, 'data')
            _workspace_manager.temp_dir = os.path.join(_workspace_manager.workspace_dir, 'temp')
            _workspace_manager.cache_dir = os.path.join(_workspace_manager.workspace_dir, 'cache')
            _workspace_manager.report_dir = os.path.join(_workspace_manager.workspace_dir, 'reports')
            _workspace_manager.data_dir = os.path.join(_workspace_manager.workspace_dir, 'raw')
            _workspace_manager._ensure_dirs()
        
        if allow_disk_write is not None:
            _workspace_manager.set_allow_disk_write(allow_disk_write)
