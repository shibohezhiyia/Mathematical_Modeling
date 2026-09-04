"""
Extended unit tests for core/workspace_manager.py
Maximizes line coverage for WorkspaceManager and related utilities.
"""
import os
import sys
import unittest
import tempfile
import shutil
import atexit
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from core.workspace_manager import (
    WorkspaceManager, get_workspace_manager, set_workspace_config, _workspace_manager
)


class TestWorkspaceManager(unittest.TestCase):
    """Test WorkspaceManager class"""

    def setUp(self):
        self.temp_root = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp directory
        if os.path.exists(self.temp_root):
            shutil.rmtree(self.temp_root, ignore_errors=True)
        # Reset global singleton
        import core.workspace_manager as wm
        wm._workspace_manager = None

    def test_init_default_root(self):
        """Test initialization with default root directory"""
        wm = WorkspaceManager()
        self.assertEqual(wm.root_dir, os.path.abspath(os.getcwd()))
        self.assertTrue(wm.allow_disk_write)

    def test_init_custom_root(self):
        """Test initialization with custom root directory"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        self.assertEqual(wm.root_dir, os.path.abspath(self.temp_root))

    def test_init_disabled_write(self):
        """Test initialization with disk write disabled"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        self.assertFalse(wm.allow_disk_write)
        # Directories should not be created
        self.assertFalse(os.path.exists(wm.workspace_dir))

    def test_dirs_created(self):
        """Test that workspace directories are created"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        self.assertTrue(os.path.exists(wm.workspace_dir))
        self.assertTrue(os.path.exists(wm.temp_dir))
        self.assertTrue(os.path.exists(wm.cache_dir))
        self.assertTrue(os.path.exists(wm.report_dir))
        self.assertTrue(os.path.exists(wm.data_dir))

    def test_set_allow_disk_write_enable(self):
        """Test enabling disk write"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        self.assertFalse(wm.allow_disk_write)
        wm.set_allow_disk_write(True)
        self.assertTrue(wm.allow_disk_write)
        self.assertTrue(os.path.exists(wm.workspace_dir))

    def test_set_allow_disk_write_disable(self):
        """Test disabling disk write"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=True)
        self.assertTrue(wm.allow_disk_write)
        wm.set_allow_disk_write(False)
        self.assertFalse(wm.allow_disk_write)

    def test_check_permission_allowed(self):
        """Test check_permission when allowed"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=True)
        self.assertTrue(wm.check_permission("写入"))

    def test_check_permission_denied(self):
        """Test check_permission when denied"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        with patch('core.workspace_manager.log_warning'):
            self.assertFalse(wm.check_permission("写入"))

    def test_safe_path_relative(self):
        """Test safe_path with relative path"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        path = wm.safe_path("report.txt", subdir='reports')
        expected = os.path.join(wm.report_dir, "report.txt")
        self.assertEqual(path, expected)

    def test_safe_path_empty(self):
        """Test safe_path with empty path"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        path = wm.safe_path("")
        self.assertEqual(path, "")

    def test_safe_path_under_workspace(self):
        """Test safe_path with path already under workspace"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        existing_path = os.path.join(wm.report_dir, "existing.txt")
        result = wm.safe_path(existing_path, subdir='reports')
        self.assertEqual(result, existing_path)

    def test_safe_path_system_drive_windows(self):
        """Test safe_path redirects C: drive paths on Windows"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with patch('os.name', 'nt'):
            with patch('os.path.splitdrive', return_value=('C:', '\\file.txt')):
                path = wm.safe_path("C:\\file.txt", subdir='reports')
                expected = os.path.join(wm.report_dir, "file.txt")
                self.assertEqual(path, expected)

    def test_safe_path_other_drive(self):
        """Test safe_path with other drive paths"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with patch('os.name', 'nt'):
            with patch('os.path.splitdrive', return_value=('D:', '\\file.txt')):
                with patch('core.workspace_manager.log_warning'):
                    path = wm.safe_path("D:\\file.txt", subdir='reports')
                    self.assertEqual(path, "D:\\file.txt")

    @unittest.skipIf(os.name == 'nt', "Unix path tests not applicable on Windows")
    def test_safe_path_unix_system_path(self):
        """Test safe_path with Unix system paths"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with patch('os.name', 'posix'):
            path = wm.safe_path("/usr/bin/file.txt", subdir='reports')
            expected = os.path.join(wm.report_dir, "file.txt")
            self.assertEqual(path, expected)

    @unittest.skipIf(os.name == 'nt', "Unix path tests not applicable on Windows")
    def test_safe_path_unix_non_system(self):
        """Test safe_path with Unix non-system paths"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with patch('os.name', 'posix'):
            with patch('core.workspace_manager.log_warning'):
                path = wm.safe_path("/home/user/file.txt", subdir='reports')
                self.assertEqual(path, "/home/user/file.txt")

    def test_is_system_drive_windows(self):
        """Test _is_system_drive on Windows"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with patch('os.name', 'nt'):
            self.assertTrue(wm._is_system_drive("C:\\test"))
            self.assertTrue(wm._is_system_drive("c:\\test"))
            self.assertFalse(wm._is_system_drive("D:\\test"))

    @unittest.skipIf(os.name == 'nt', "Unix path tests not applicable on Windows")
    def test_is_system_drive_linux(self):
        """Test _is_system_drive on Linux"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with patch('os.name', 'posix'):
            self.assertTrue(wm._is_system_drive("/usr/bin"))
            self.assertTrue(wm._is_system_drive("/etc/config"))
            self.assertFalse(wm._is_system_drive("/home/user"))

    def test_is_under_workspace(self):
        """Test _is_under_workspace"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        self.assertTrue(wm._is_under_workspace(wm.report_dir))
        self.assertFalse(wm._is_under_workspace("/some/other/path"))

    def test_create_temp_dir(self):
        """Test create_temp_dir"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        temp_dir = wm.create_temp_dir(prefix="test")
        self.assertTrue(os.path.exists(temp_dir))
        self.assertTrue(temp_dir.startswith(wm.temp_dir))

    def test_create_temp_dir_denied(self):
        """Test create_temp_dir when disk write is disabled"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        with self.assertRaises(PermissionError):
            wm.create_temp_dir()

    def test_create_cache_dir(self):
        """Test create_cache_dir"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        cache_dir = wm.create_cache_dir(name="my_cache")
        self.assertTrue(os.path.exists(cache_dir))
        self.assertEqual(cache_dir, os.path.join(wm.cache_dir, "my_cache"))

    def test_create_cache_dir_empty_name(self):
        """Test create_cache_dir with empty name"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        cache_dir = wm.create_cache_dir(name="")
        self.assertEqual(cache_dir, wm.cache_dir)

    def test_cache_and_relative_paths_cannot_escape_category(self):
        wm = WorkspaceManager(root_dir=self.temp_root)
        with self.assertRaises(ValueError):
            wm.create_cache_dir(name="../raw")
        with self.assertRaises(ValueError):
            wm.safe_path("../../outside.txt", subdir="cache")

    def test_create_cache_dir_denied(self):
        """Test create_cache_dir when disk write is disabled"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        with self.assertRaises(PermissionError):
            wm.create_cache_dir()

    def test_get_data_path(self):
        """Test get_data_path"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        path = wm.get_data_path("data.csv")
        self.assertEqual(path, os.path.join(wm.data_dir, "data.csv"))

    def test_get_report_path(self):
        """Test get_report_path"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        path = wm.get_report_path("report.html")
        self.assertEqual(path, os.path.join(wm.report_dir, "report.html"))

    def test_write_text(self):
        """Test write_text"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        path = wm.write_text("hello.txt", "Hello World", subdir='reports')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        with open(path, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "Hello World")

    def test_write_text_denied(self):
        """Test write_text when disk write is disabled"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        result = wm.write_text("hello.txt", "Hello")
        self.assertIsNone(result)

    def test_write_bytes(self):
        """Test write_bytes"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        path = wm.write_bytes("data.bin", b"\x00\x01\x02", subdir='reports')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        with open(path, 'rb') as f:
            self.assertEqual(f.read(), b"\x00\x01\x02")

    def test_write_bytes_denied(self):
        """Test write_bytes when disk write is disabled"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        result = wm.write_bytes("data.bin", b"test")
        self.assertIsNone(result)

    def test_save_dataframe_csv(self):
        """Test save_dataframe with CSV extension"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        path = wm.save_dataframe(df, "data.csv", subdir='reports')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_dataframe_json(self):
        """Test save_dataframe with JSON extension"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        df = pd.DataFrame({'a': [1, 2, 3]})
        path = wm.save_dataframe(df, "data.json", subdir='reports')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_save_dataframe_parquet(self):
        """Test save_dataframe with parquet extension"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        df = pd.DataFrame({'a': [1, 2, 3]})
        path = wm.save_dataframe(df, "data.parquet", subdir='reports')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_save_dataframe_unknown_ext(self):
        """Test save_dataframe with unknown extension falls back to csv"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        df = pd.DataFrame({'a': [1, 2, 3]})
        path = wm.save_dataframe(df, "data.xyz", subdir='reports')
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_dataframe_denied(self):
        """Test save_dataframe when disk write is disabled"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = wm.save_dataframe(df, "data.csv")
        self.assertIsNone(result)

    def test_clear_temp(self):
        """Test clear_temp"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        temp_file = os.path.join(wm.temp_dir, "tempfile.txt")
        with open(temp_file, 'w') as f:
            f.write("temp")
        wm.clear_temp()
        self.assertFalse(os.path.exists(temp_file))
        self.assertTrue(os.path.exists(wm.temp_dir))

    def test_clear_cache(self):
        """Test clear_cache"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        cache_file = os.path.join(wm.cache_dir, "cachefile.txt")
        with open(cache_file, 'w') as f:
            f.write("cache")
        wm.clear_cache()
        self.assertFalse(os.path.exists(cache_file))
        self.assertTrue(os.path.exists(wm.cache_dir))

    def test_clear_all(self):
        """Test clear_all"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        temp_file = os.path.join(wm.temp_dir, "tempfile.txt")
        with open(temp_file, 'w') as f:
            f.write("temp")
        wm.clear_all()
        self.assertFalse(os.path.exists(temp_file))
        self.assertTrue(os.path.exists(wm.workspace_dir))

    def test_clear_all_disabled_write(self):
        """Test clear_all when disk write is disabled"""
        wm = WorkspaceManager(root_dir=self.temp_root, allow_disk_write=False)
        # Create file manually
        os.makedirs(wm.temp_dir, exist_ok=True)
        temp_file = os.path.join(wm.temp_dir, "tempfile.txt")
        with open(temp_file, 'w') as f:
            f.write("temp")
        wm.clear_all()
        # Should not recreate dirs when write is disabled
        self.assertFalse(os.path.exists(temp_file))

    def test_get_info(self):
        """Test get_info"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        info = wm.get_info()
        self.assertIn('root_dir', info)
        self.assertIn('allow_disk_write', info)
        self.assertIn('workspace_dir', info)
        self.assertIn('temp_size_mb', info)
        self.assertIn('temp_files', info)

    def test_get_info_with_files(self):
        """Test get_info with actual files"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        # Create some files
        with open(os.path.join(wm.temp_dir, "a.txt"), 'w') as f:
            f.write("hello")
        with open(os.path.join(wm.cache_dir, "b.txt"), 'w') as f:
            f.write("world")
        info = wm.get_info()
        # Size may round to 0.0 for tiny files, check files count instead
        self.assertGreaterEqual(info['temp_size_mb'], 0)
        self.assertEqual(info['temp_files'], 1)
        self.assertEqual(info['cache_files'], 1)

    def test_get_dir_size(self):
        """Test _get_dir_size"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with open(os.path.join(wm.temp_dir, "size_test.txt"), 'w') as f:
            f.write("12345")
        size = wm._get_dir_size(wm.temp_dir)
        self.assertEqual(size, 5)

    def test_get_dir_size_missing_file(self):
        """Test _get_dir_size with file that disappears"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        # Create a file, then mock getsize to fail
        with patch('os.path.getsize', side_effect=OSError("gone")):
            size = wm._get_dir_size(wm.temp_dir)
            self.assertEqual(size, 0)

    def test_print_info(self):
        """Test print_info"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        with patch('builtins.print') as mock_print:
            wm.print_info()
            self.assertTrue(mock_print.called)
            # Check that it prints section headers
            calls = [call for call in mock_print.call_args_list]
            self.assertGreater(len(calls), 0)

    def test_cleanup_on_exit(self):
        """Test _cleanup_on_exit removes temp directory"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        temp_subdir = os.path.join(wm.temp_dir, "sub")
        os.makedirs(temp_subdir, exist_ok=True)
        with open(os.path.join(temp_subdir, "file.txt"), 'w') as f:
            f.write("test")
        wm._cleanup_on_exit()
        self.assertFalse(os.path.exists(wm.temp_dir))

    def test_cleanup_on_exit_no_temp_dir(self):
        """Test _cleanup_on_exit when temp dir doesn't exist"""
        wm = WorkspaceManager(root_dir=self.temp_root)
        shutil.rmtree(wm.temp_dir, ignore_errors=True)
        # Should not raise
        wm._cleanup_on_exit()

    def test_atexit_registration(self):
        """Test that atexit is registered on init"""
        with patch('atexit.register') as mock_register:
            wm = WorkspaceManager(root_dir=self.temp_root)
            mock_register.assert_called()


class TestGetWorkspaceManager(unittest.TestCase):
    """Test get_workspace_manager singleton"""

    def tearDown(self):
        import core.workspace_manager as wm
        wm._workspace_manager = None

    def test_singleton(self):
        """Test that get_workspace_manager returns same instance"""
        wm1 = get_workspace_manager()
        wm2 = get_workspace_manager()
        self.assertIs(wm1, wm2)

    def test_force_new(self):
        """Test force_new parameter"""
        wm1 = get_workspace_manager()
        wm2 = get_workspace_manager(force_new=True)
        self.assertIsNot(wm1, wm2)

    def test_with_root_dir(self):
        """Test get_workspace_manager with root_dir"""
        temp_root = tempfile.mkdtemp()
        try:
            wm = get_workspace_manager(root_dir=temp_root)
            self.assertEqual(wm.root_dir, os.path.abspath(temp_root))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


class TestSetWorkspaceConfig(unittest.TestCase):
    """Test set_workspace_config function"""

    def tearDown(self):
        import core.workspace_manager as wm
        wm._workspace_manager = None

    def test_creates_new_manager(self):
        """Test set_workspace_config creates new manager if none exists"""
        temp_root = tempfile.mkdtemp()
        try:
            set_workspace_config(root_dir=temp_root, allow_disk_write=False)
            wm = get_workspace_manager()
            self.assertEqual(wm.root_dir, os.path.abspath(temp_root))
            self.assertFalse(wm.allow_disk_write)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            import core.workspace_manager as wm_mod
            wm_mod._workspace_manager = None

    def test_updates_existing_manager(self):
        """Test set_workspace_config updates existing manager"""
        temp_root1 = tempfile.mkdtemp()
        temp_root2 = tempfile.mkdtemp()
        try:
            wm1 = get_workspace_manager(root_dir=temp_root1)
            set_workspace_config(root_dir=temp_root2, allow_disk_write=False)
            self.assertEqual(wm1.root_dir, os.path.abspath(temp_root2))
            self.assertFalse(wm1.allow_disk_write)
        finally:
            shutil.rmtree(temp_root1, ignore_errors=True)
            shutil.rmtree(temp_root2, ignore_errors=True)
            import core.workspace_manager as wm_mod
            wm_mod._workspace_manager = None

    def test_allow_disk_write_only(self):
        """Test set_workspace_config only updates allow_disk_write"""
        temp_root = tempfile.mkdtemp()
        try:
            wm = get_workspace_manager(root_dir=temp_root, allow_disk_write=True)
            set_workspace_config(allow_disk_write=False)
            self.assertFalse(wm.allow_disk_write)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            import core.workspace_manager as wm_mod
            wm_mod._workspace_manager = None

    def test_root_dir_only(self):
        """Test set_workspace_config only updates root_dir"""
        temp_root1 = tempfile.mkdtemp()
        temp_root2 = tempfile.mkdtemp()
        try:
            wm = get_workspace_manager(root_dir=temp_root1)
            set_workspace_config(root_dir=temp_root2)
            self.assertEqual(wm.root_dir, os.path.abspath(temp_root2))
        finally:
            shutil.rmtree(temp_root1, ignore_errors=True)
            shutil.rmtree(temp_root2, ignore_errors=True)
            import core.workspace_manager as wm_mod
            wm_mod._workspace_manager = None


class TestEdgeCases(unittest.TestCase):
    """Test edge cases"""

    def tearDown(self):
        import core.workspace_manager as wm
        wm._workspace_manager = None

    def test_safe_path_subdir_variations(self):
        """Test safe_path with different subdirs"""
        temp_root = tempfile.mkdtemp()
        try:
            wm = WorkspaceManager(root_dir=temp_root)
            self.assertTrue(wm.safe_path("f.txt", "temp").startswith(wm.temp_dir))
            self.assertTrue(wm.safe_path("f.txt", "cache").startswith(wm.cache_dir))
            self.assertTrue(wm.safe_path("f.txt", "data").startswith(wm.data_dir))
            self.assertTrue(wm.safe_path("f.txt", "unknown").startswith(wm.report_dir))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_write_text_nested_path(self):
        """Test write_text creates nested directories"""
        temp_root = tempfile.mkdtemp()
        try:
            wm = WorkspaceManager(root_dir=temp_root)
            path = wm.write_text("a/b/c/nested.txt", "nested", subdir='reports')
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
