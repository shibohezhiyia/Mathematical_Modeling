"""
工作空间管理器测试
"""
import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.workspace_manager import (
    WorkspaceManager, get_workspace_manager, set_workspace_config
)


class TestWorkspaceManager(unittest.TestCase):
    """测试工作空间管理器"""
    
    def test_default_paths_under_project(self):
        """默认路径应在项目目录内"""
        wm = WorkspaceManager()
        self.assertTrue(os.path.isabs(wm.root_dir))
        self.assertTrue(wm.workspace_dir.startswith(wm.root_dir))
        self.assertTrue(wm.temp_dir.startswith(wm.workspace_dir))
        self.assertTrue(wm.cache_dir.startswith(wm.workspace_dir))
        self.assertTrue(wm.report_dir.startswith(wm.workspace_dir))
    
    def test_no_system_drive_write(self):
        """不应允许写入系统盘"""
        wm = WorkspaceManager()
        
        # Windows C盘路径应被重定向
        if os.name == 'nt':
            safe = wm.safe_path('C:/some/path/report.json')
            self.assertFalse(safe.upper().startswith('C:\\'))
            self.assertTrue(safe.startswith(wm.workspace_dir))
    
    def test_safe_path_relative(self):
        """相对路径应放入workspace"""
        wm = WorkspaceManager()
        safe = wm.safe_path('my_report.json')
        self.assertTrue(safe.startswith(wm.report_dir))
    
    def test_safe_path_already_inside(self):
        """已在工作目录内的路径应保持"""
        wm = WorkspaceManager()
        inside = os.path.join(wm.root_dir, 'existing_file.csv')
        safe = wm.safe_path(inside)
        self.assertEqual(safe, inside)
    
    def test_allow_disk_write_switch(self):
        """磁盘写入开关"""
        wm = WorkspaceManager(allow_disk_write=True)
        self.assertTrue(wm.allow_disk_write)
        
        wm.set_allow_disk_write(False)
        self.assertFalse(wm.allow_disk_write)
        
        # 关闭后写入应失败
        with self.assertRaises(PermissionError):
            wm.create_temp_dir()
    
    def test_write_text_disabled(self):
        """关闭时write_text返回None"""
        wm = WorkspaceManager(allow_disk_write=False)
        result = wm.write_text('test.txt', 'hello')
        self.assertIsNone(result)
    
    def test_create_temp_dir(self):
        """创建临时目录"""
        wm = WorkspaceManager()
        d = wm.create_temp_dir(prefix='test')
        self.assertTrue(os.path.exists(d))
        self.assertTrue(d.startswith(wm.temp_dir))
        self.assertTrue(os.path.basename(d).startswith('test_'))
    
    def test_save_dataframe(self):
        """保存DataFrame"""
        import pandas as pd
        wm = WorkspaceManager()
        df = pd.DataFrame({'A': [1, 2, 3]})
        path = wm.save_dataframe(df, 'test_data.csv')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        
        # 读取验证
        df2 = pd.read_csv(path)
        self.assertEqual(len(df2), 3)
    
    def test_get_info(self):
        """获取信息"""
        wm = WorkspaceManager()
        info = wm.get_info()
        self.assertIn('root_dir', info)
        self.assertIn('allow_disk_write', info)
        self.assertIn('temp_size_mb', info)
    
    def test_clear_temp(self):
        """清空临时目录"""
        wm = WorkspaceManager()
        d = wm.create_temp_dir()
        with open(os.path.join(d, 'file.txt'), 'w') as f:
            f.write('test')
        
        wm.clear_temp()
        self.assertFalse(os.path.exists(d))
    
    def test_global_singleton(self):
        """全局单例"""
        wm1 = get_workspace_manager(force_new=True)
        wm2 = get_workspace_manager()
        self.assertIs(wm1, wm2)
    
    def test_set_config(self):
        """动态修改配置"""
        set_workspace_config(allow_disk_write=False)
        wm = get_workspace_manager()
        self.assertFalse(wm.allow_disk_write)
        
        # 恢复
        set_workspace_config(allow_disk_write=True)
        self.assertTrue(wm.allow_disk_write)


class TestWorkspaceIntegration(unittest.TestCase):
    """集成测试：验证各模块使用WorkspaceManager"""
    
    def test_data_module_report_respects_switch(self):
        """DataModule.save_report 应遵守磁盘开关"""
        from core.data_module import DataModule
        import pandas as pd
        
        wm = get_workspace_manager(force_new=True)
        wm.set_allow_disk_write(False)
        
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        module = DataModule()
        module.raw_data = df
        module.analyze()
        
        result = module.save_report('test_report.json')
        self.assertIsNone(result)
        
        # 恢复
        wm.set_allow_disk_write(True)


if __name__ == '__main__':
    unittest.main()
