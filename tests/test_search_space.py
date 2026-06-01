"""
测试 SearchSpace 丰富格式支持

覆盖:
  - 向后兼容: 列表/单值格式
  - 丰富格式: float/int/categorical/bool + log scale + labels
  - 条件参数: conditional dependencies
  - 采样: sample/sample_many/build_candidates/to_optuna
"""

import unittest
import numpy as np
from core.search_space import Parameter, SearchSpace


class TestParameter(unittest.TestCase):
    def test_float_uniform(self):
        p = Parameter(name='C', type='float', low=0.01, high=10.0)
        rng = np.random.RandomState(42)
        for _ in range(10):
            v = p.sample(rng)
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.01)
            self.assertLessEqual(v, 10.0)
    
    def test_float_log(self):
        p = Parameter(name='lr', type='float', low=1e-5, high=1.0, scale='log')
        rng = np.random.RandomState(42)
        values = [p.sample(rng) for _ in range(100)]
        self.assertTrue(all(v > 0 for v in values))
        self.assertTrue(all(v >= 1e-5 and v <= 1.0 for v in values))
        # 对数采样应该偏向小值
        self.assertLess(np.median(values), 0.1)
    
    def test_int(self):
        p = Parameter(name='max_depth', type='int', low=3, high=10)
        rng = np.random.RandomState(42)
        for _ in range(20):
            v = p.sample(rng)
            self.assertIsInstance(v, int)
            self.assertIn(v, range(3, 11))
    
    def test_categorical(self):
        p = Parameter(name='kernel', type='categorical', choices=['rbf', 'linear', 'poly'])
        rng = np.random.RandomState(42)
        for _ in range(20):
            v = p.sample(rng)
            self.assertIn(v, ['rbf', 'linear', 'poly'])
    
    def test_bool(self):
        p = Parameter(name='fit_intercept', type='bool')
        rng = np.random.RandomState(42)
        results = [p.sample(rng) for _ in range(50)]
        self.assertTrue(any(results))
        self.assertTrue(any(not r for r in results))
    
    def test_build_candidates_float_log(self):
        p = Parameter(name='lr', type='float', low=1e-5, high=1.0, scale='log')
        cand = p.build_candidates(n=5)
        self.assertEqual(len(cand), 5)
        self.assertTrue(all(v > 0 for v in cand))
        # 对数尺度: 值应该呈几何级数
        ratios = [cand[i+1] / cand[i] for i in range(len(cand)-1)]
        self.assertTrue(all(r > 1 for r in ratios))
    
    def test_build_candidates_int(self):
        p = Parameter(name='depth', type='int', low=3, high=10)
        cand = p.build_candidates(n=5)
        self.assertTrue(all(isinstance(v, int) for v in cand))
        self.assertTrue(all(3 <= v <= 10 for v in cand))
    
    def test_condition_active(self):
        p = Parameter(name='max_depth', type='int', low=3, high=10,
                      condition={'param': 'booster', 'values': ['gbtree']})
        self.assertTrue(p.is_active({'booster': 'gbtree'}))
        self.assertFalse(p.is_active({'booster': 'dart'}))
        self.assertFalse(p.is_active({}))
    
    def test_condition_sample(self):
        p = Parameter(name='max_depth', type='int', low=3, high=10,
                      condition={'param': 'booster', 'values': ['gbtree']})
        rng = np.random.RandomState(42)
        self.assertIsNotNone(p.sample(rng, {'booster': 'gbtree'}))
        self.assertIsNone(p.sample(rng, {'booster': 'dart'}))


class TestSearchSpace(unittest.TestCase):
    def test_backward_compat_list(self):
        space = SearchSpace({'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1.0]})
        rng = np.random.RandomState(42)
        params = space.sample(rng=rng)
        self.assertIn(params['penalty'], ['l1', 'l2'])
        self.assertIn(params['C'], [0.01, 0.1, 1.0])
    
    def test_rich_format_log_scale(self):
        space = SearchSpace({
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'scale': 'log'},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        })
        rng = np.random.RandomState(42)
        params = space.sample(rng=rng)
        self.assertIn('learning_rate', params)
        self.assertIn('max_depth', params)
        self.assertIsInstance(params['learning_rate'], float)
        self.assertIsInstance(params['max_depth'], int)
        self.assertGreaterEqual(params['learning_rate'], 1e-5)
        self.assertLessEqual(params['learning_rate'], 1e-1)
        self.assertIn(params['max_depth'], range(3, 11))
    
    def test_categorical_with_labels(self):
        space = SearchSpace({
            'booster': {'type': 'categorical', 'choices': ['gbtree', 'dart'],
                        'labels': ['梯度提升树', 'Dropouts']}
        })
        p = space.get_param('booster')
        self.assertEqual(p.labels, ['梯度提升树', 'Dropouts'])
        rng = np.random.RandomState(42)
        params = space.sample(rng=rng)
        self.assertIn(params['booster'], ['gbtree', 'dart'])
    
    def test_conditional_params(self):
        space = SearchSpace({
            'booster': {'type': 'categorical', 'choices': ['gbtree', 'dart']},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10,
                          'condition': {'param': 'booster', 'values': ['gbtree']}},
            'sample_type': {'type': 'categorical', 'choices': ['uniform', 'weighted'],
                            'condition': {'param': 'booster', 'values': ['dart']}},
        })
        rng = np.random.RandomState(42)
        for _ in range(20):
            params = space.sample(rng=rng)
            self.assertIn('booster', params)
            if params['booster'] == 'gbtree':
                self.assertIn('max_depth', params)
                self.assertNotIn('sample_type', params)
            else:
                self.assertIn('sample_type', params)
                self.assertNotIn('max_depth', params)
    
    def test_build_candidates(self):
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'scale': 'log'},
            'depth': {'type': 'int', 'low': 3, 'high': 10},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear']},
        })
        cand = space.build_candidates(n=5)
        self.assertIn('lr', cand)
        self.assertIn('depth', cand)
        self.assertIn('kernel', cand)
        self.assertEqual(len(cand['lr']), 5)
        self.assertTrue(all(isinstance(v, float) for v in cand['lr']))
    
    def test_sample_many(self):
        space = SearchSpace({
            'C': {'type': 'float', 'low': 0.01, 'high': 10.0},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear']},
        })
        params_list = space.sample_many(n=10, random_state=42)
        self.assertEqual(len(params_list), 10)
        self.assertTrue(all('C' in p and 'kernel' in p for p in params_list))
    
    def test_get_active_params(self):
        space = SearchSpace({
            'booster': {'type': 'categorical', 'choices': ['gbtree', 'dart']},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10,
                          'condition': {'param': 'booster', 'values': ['gbtree']}},
        })
        active_gbtree = space.get_active_params({'booster': 'gbtree'})
        self.assertIn('booster', active_gbtree)
        self.assertIn('max_depth', active_gbtree)
        
        active_dart = space.get_active_params({'booster': 'dart'})
        self.assertIn('booster', active_dart)
        self.assertNotIn('max_depth', active_dart)
    
    def test_repr(self):
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log'},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear']},
        })
        s = repr(space)
        self.assertIn('SearchSpace', s)
        self.assertIn('lr', s)
        self.assertIn('kernel', s)
    
    def test_to_dict_roundtrip(self):
        config = {
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log'},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'booster': {'type': 'categorical', 'choices': ['gbtree', 'dart'],
                        'labels': ['GB', 'DART']},
            'use_gpu': {'type': 'bool'},
        }
        space = SearchSpace(config)
        d = space.to_dict()
        self.assertEqual(d['lr']['scale'], 'log')
        self.assertEqual(d['booster']['labels'], ['GB', 'DART'])
        self.assertEqual(d['use_gpu']['type'], 'bool')


class TestSearchSpaceWithOptuna(unittest.TestCase):
    def test_to_optuna_float_log(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("Optuna not installed")
        
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'scale': 'log'},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear']},
        })
        
        def objective(trial):
            params = space.to_optuna(trial)
            self.assertIn('lr', params)
            self.assertIn('max_depth', params)
            self.assertIn('kernel', params)
            self.assertIsInstance(params['lr'], float)
            self.assertIsInstance(params['max_depth'], int)
            return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)
    
    def test_to_optuna_conditional(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("Optuna not installed")
        
        space = SearchSpace({
            'booster': {'type': 'categorical', 'choices': ['gbtree', 'dart']},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10,
                          'condition': {'param': 'booster', 'values': ['gbtree']}},
        })
        
        def objective(trial):
            params = space.to_optuna(trial)
            self.assertIn('booster', params)
            if params['booster'] == 'gbtree':
                self.assertIn('max_depth', params)
            return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)


if __name__ == '__main__':
    unittest.main()
