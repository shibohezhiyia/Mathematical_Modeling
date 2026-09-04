"""
Unit tests for core/model_versioning.py
"""
import os
import tempfile
import unittest

import core.model_versioning as mv
from core.experiment_tracker import DB_PATH as EXP_DB_PATH


class TestModelVersioning(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test_mv.db')
        self._orig = mv.DB_PATH
        mv.DB_PATH = self.db_path
        mv._init_table()

    def tearDown(self):
        mv.DB_PATH = self._orig
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.tmpdir)

    def test_save_and_load_snapshot(self):
        model = {'weights': [1, 2, 3]}
        sid = mv.save_snapshot(1, 'sgd', model, metadata={'acc': 0.9})
        self.assertIsInstance(sid, int)
        loaded = mv.load_snapshot(sid)
        self.assertEqual(loaded['weights'], [1, 2, 3])

    def test_list_snapshots(self):
        mv.save_snapshot(1, 'a', [1])
        mv.save_snapshot(1, 'b', [2])
        rows = mv.list_snapshots(experiment_id=1)
        self.assertEqual(len(rows), 2)

    def test_delete_snapshot(self):
        sid = mv.save_snapshot(1, 'x', [1])
        self.assertTrue(mv.delete_snapshot(sid))
        self.assertIsNone(mv.load_snapshot(sid))

    def test_snapshot_owner_isolation(self):
        sid = mv.save_snapshot(1, 'private', {'value': 7}, owner_id='session-a')

        self.assertEqual(mv.load_snapshot(sid, owner_id='session-a'), {'value': 7})
        self.assertIsNone(mv.load_snapshot(sid, owner_id='session-b'))
        self.assertEqual(len(mv.list_snapshots(owner_id='session-a')), 1)
        self.assertEqual(mv.list_snapshots(owner_id='session-b'), [])
        self.assertFalse(mv.delete_snapshot(sid, owner_id='session-b'))
        self.assertTrue(mv.delete_snapshot(sid, owner_id='session-a'))


if __name__ == '__main__':
    unittest.main()
