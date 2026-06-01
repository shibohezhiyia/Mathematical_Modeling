"""
OptimizerFactory 与附加优化器单元测试

覆盖:
  - OptimizerFactory.create 所有策略
  - OptimizerFactory.list_strategies
  - RandomSearchOptimizer.optimize (有/无搜索空间)
  - HyperbandOptimizer.optimize
  - GeneticAlgorithmOptimizer.optimize
  - 错误处理 (未知策略)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimizer_factory import (
    OptimizerFactory,
    RandomSearchOptimizer,
    HyperbandOptimizer,
    GeneticAlgorithmOptimizer,
)
from core.modeling_engine import TaskType


class TestOptimizerFactoryStrategies(unittest.TestCase):
    """测试策略字典与列表"""

    def test_strategies_contains_all_expected(self):
        expected = {'bayesian', 'tpe', 'cmaes', 'rl', 'random', 'hyperband', 'genetic'}
        self.assertTrue(expected.issubset(set(OptimizerFactory.STRATEGIES.keys())))

    def test_list_strategies(self):
        strategies = OptimizerFactory.list_strategies()
        self.assertIsInstance(strategies, list)
        for s in ['bayesian', 'tpe', 'cmaes', 'rl', 'random', 'hyperband', 'genetic']:
            self.assertIn(s, strategies)


class TestOptimizerFactoryCreate(unittest.TestCase):
    """测试 OptimizerFactory.create 各策略实例化"""

    def test_create_random(self):
        opt = OptimizerFactory.create('random', n_trials=10)
        self.assertIsInstance(opt, RandomSearchOptimizer)
        self.assertEqual(opt.n_trials, 10)

    def test_create_hyperband(self):
        opt = OptimizerFactory.create('hyperband', n_trials=20, eta=3)
        self.assertIsInstance(opt, HyperbandOptimizer)
        self.assertEqual(opt.n_trials, 20)
        self.assertEqual(opt.eta, 3)

    def test_create_genetic(self):
        opt = OptimizerFactory.create('genetic', n_trials=30, population_size=10)
        self.assertIsInstance(opt, GeneticAlgorithmOptimizer)
        self.assertEqual(opt.n_trials, 30)
        self.assertEqual(opt.population_size, 10)

    def test_create_bayesian(self):
        opt = OptimizerFactory.create('bayesian', n_trials=15)
        from core.hyperparameter_optimizer import BayesianOptimizer
        self.assertIsInstance(opt, BayesianOptimizer)
        self.assertEqual(opt.n_trials, 15)
        self.assertEqual(opt.sampler_type, 'tpe')

    def test_create_tpe(self):
        opt = OptimizerFactory.create('tpe', n_trials=15)
        from core.hyperparameter_optimizer import BayesianOptimizer
        self.assertIsInstance(opt, BayesianOptimizer)
        self.assertEqual(opt.sampler_type, 'tpe')

    def test_create_cmaes(self):
        opt = OptimizerFactory.create('cmaes', n_trials=15)
        from core.hyperparameter_optimizer import BayesianOptimizer
        self.assertIsInstance(opt, BayesianOptimizer)
        self.assertEqual(opt.sampler_type, 'cmaes')

    def test_create_rl(self):
        opt = OptimizerFactory.create('rl', n_trials=10)
        from core.reinforcement_learning import RLOptimizer
        self.assertIsInstance(opt, RLOptimizer)
        self.assertEqual(opt.n_trials, 10)

    def test_create_unknown_strategy_raises(self):
        with self.assertRaises(ValueError) as ctx:
            OptimizerFactory.create('unknown_strategy')
        self.assertIn('未知优化策略', str(ctx.exception))

    def test_create_case_insensitive(self):
        opt = OptimizerFactory.create('RANDOM')
        self.assertIsInstance(opt, RandomSearchOptimizer)
        opt2 = OptimizerFactory.create('  Bayesian  ')
        from core.hyperparameter_optimizer import BayesianOptimizer
        self.assertIsInstance(opt2, BayesianOptimizer)

    def test_create_passes_kwargs(self):
        opt = OptimizerFactory.create('random', n_trials=5, cv_folds=2, random_state=123)
        self.assertEqual(opt.cv_folds, 2)
        self.assertEqual(opt.random_state, 123)


class TestRandomSearchOptimizer(unittest.TestCase):
    """测试 RandomSearchOptimizer.optimize"""

    def setUp(self):
        self.X = pd.DataFrame({'f1': np.random.randn(30), 'f2': np.random.randn(30)})
        self.y_cls = pd.Series(np.random.choice([0, 1], 30))
        self.y_reg = pd.Series(np.random.randn(30))

    def test_optimize_with_search_space(self):
        optimizer = RandomSearchOptimizer(n_trials=3, cv_folds=2, random_state=42)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION)
        self.assertEqual(result.model_key, 'lr')
        self.assertIsInstance(result.best_params, dict)
        self.assertEqual(result.sampler_type, 'random')
        self.assertGreaterEqual(result.n_trials, 0)
        self.assertIsInstance(result.optimization_history, list)

    def test_optimize_without_search_space(self):
        """模型无搜索空间时应直接返回默认参数"""
        optimizer = RandomSearchOptimizer(n_trials=3, cv_folds=2, random_state=42)
        # linear regression 默认无超参搜索空间
        result = optimizer.optimize('linear', self.X, self.y_reg, TaskType.REGRESSION)
        self.assertEqual(result.model_key, 'linear')
        self.assertEqual(result.n_trials, 0)
        self.assertEqual(result.best_score, 0.0)
        self.assertIsInstance(result.best_params, dict)

    def test_optimize_custom_search_space(self):
        optimizer = RandomSearchOptimizer(n_trials=3, cv_folds=2, random_state=42)
        custom_space = {'C': [0.01, 0.1, 1.0]}
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION,
                                        custom_search_space=custom_space)
        self.assertEqual(result.model_key, 'lr')
        self.assertIsInstance(result.best_params, dict)
        self.assertIn('C', result.best_params)

    def test_optimize_task_type_string(self):
        optimizer = RandomSearchOptimizer(n_trials=2, cv_folds=2, random_state=42)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, 'classification')
        self.assertEqual(result.model_key, 'lr')

    def test_optimize_with_failures(self):
        """部分 trial 失败时应继续"""
        optimizer = RandomSearchOptimizer(n_trials=5, cv_folds=2, random_state=42)
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError('mock failure')
            return 0.85

        with patch.object(optimizer, '_evaluate_model', side_effect=side_effect):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION)
        self.assertLessEqual(result.n_trials, 3)


class TestHyperbandOptimizer(unittest.TestCase):
    """测试 HyperbandOptimizer.optimize"""

    def setUp(self):
        self.X = pd.DataFrame({'f1': np.random.randn(30), 'f2': np.random.randn(30)})
        self.y_cls = pd.Series(np.random.choice([0, 1], 30))
        self.y_reg = pd.Series(np.random.randn(30))

    def test_optimize_with_search_space(self):
        optimizer = HyperbandOptimizer(n_trials=27, cv_folds=2, random_state=42, max_resource=9)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION)
        self.assertEqual(result.model_key, 'lr')
        self.assertEqual(result.sampler_type, 'hyperband')
        self.assertIsInstance(result.best_params, dict)
        self.assertIsInstance(result.optimization_history, list)
        self.assertGreaterEqual(result.n_trials, 0)

    def test_optimize_without_search_space(self):
        optimizer = HyperbandOptimizer(n_trials=27, cv_folds=2, random_state=42)
        result = optimizer.optimize('linear', self.X, self.y_reg, TaskType.REGRESSION)
        self.assertEqual(result.model_key, 'linear')
        self.assertEqual(result.n_trials, 0)
        self.assertEqual(result.best_score, 0.0)

    def test_optimize_task_type_string(self):
        optimizer = HyperbandOptimizer(n_trials=9, cv_folds=2, random_state=42, max_resource=3)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, 'classification')
        self.assertEqual(result.model_key, 'lr')

    def test_apply_resource_tree_models(self):
        optimizer = HyperbandOptimizer(n_trials=9, cv_folds=2, random_state=42)
        params = {'n_estimators': 200, 'max_depth': 10}
        result = optimizer._apply_resource(params, 5, 'rf')
        self.assertEqual(result['n_estimators'], 50)
        self.assertEqual(result['max_depth'], 10)

    def test_apply_resource_neural(self):
        optimizer = HyperbandOptimizer(n_trials=9, cv_folds=2, random_state=42)
        params = {'epochs': 100, 'max_epochs': 200}
        result = optimizer._apply_resource(params, 10, 'mlp')
        self.assertEqual(result['epochs'], 10)
        self.assertEqual(result['max_epochs'], 10)

    def test_apply_resource_max_iter(self):
        optimizer = HyperbandOptimizer(n_trials=9, cv_folds=2, random_state=42)
        params = {'max_iter': 1000}
        result = optimizer._apply_resource(params, 3, 'lr')
        self.assertEqual(result['max_iter'], 30)

    def test_apply_resource_iterations(self):
        optimizer = HyperbandOptimizer(n_trials=9, cv_folds=2, random_state=42)
        params = {'iterations': 200}
        result = optimizer._apply_resource(params, 5, 'catboost')
        self.assertEqual(result['iterations'], 50)

    def test_apply_resource_no_match(self):
        optimizer = HyperbandOptimizer(n_trials=9, cv_folds=2, random_state=42)
        params = {'C': 1.0}
        result = optimizer._apply_resource(params, 5, 'lr')
        self.assertEqual(result['C'], 1.0)


class TestGeneticAlgorithmOptimizer(unittest.TestCase):
    """测试 GeneticAlgorithmOptimizer.optimize"""

    def setUp(self):
        self.X = pd.DataFrame({'f1': np.random.randn(30), 'f2': np.random.randn(30)})
        self.y_cls = pd.Series(np.random.choice([0, 1], 30))
        self.y_reg = pd.Series(np.random.randn(30))

    def test_optimize_with_search_space(self):
        optimizer = GeneticAlgorithmOptimizer(n_trials=20, population_size=4, cv_folds=2, random_state=42)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION)
        self.assertEqual(result.model_key, 'lr')
        self.assertEqual(result.sampler_type, 'genetic')
        self.assertIsInstance(result.best_params, dict)
        self.assertIsInstance(result.optimization_history, list)

    def test_optimize_without_search_space(self):
        optimizer = GeneticAlgorithmOptimizer(n_trials=20, population_size=4, cv_folds=2, random_state=42)
        result = optimizer.optimize('linear', self.X, self.y_reg, TaskType.REGRESSION)
        self.assertEqual(result.model_key, 'linear')
        self.assertEqual(result.n_trials, 0)
        self.assertEqual(result.best_score, 0.0)

    def test_optimize_task_type_string(self):
        optimizer = GeneticAlgorithmOptimizer(n_trials=20, population_size=4, cv_folds=2, random_state=42)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, 'classification')
        self.assertEqual(result.model_key, 'lr')

    def test_random_chromosome_with_search_space_object(self):
        from core.search_space import SearchSpace
        optimizer = GeneticAlgorithmOptimizer(random_state=42)
        space = SearchSpace({'C': {'type': 'float', 'low': 0.01, 'high': 1.0}})
        chrom = optimizer._random_chromosome(space)
        self.assertEqual(len(chrom), 1)
        self.assertIsInstance(chrom[0], (int, np.integer))

    def test_random_chromosome_with_dict(self):
        optimizer = GeneticAlgorithmOptimizer(random_state=42)
        space = {'C': [0.01, 0.1, 1.0], 'penalty': ['l1', 'l2']}
        chrom = optimizer._random_chromosome(space)
        self.assertEqual(len(chrom), 2)

    def test_chromosome_to_params_with_search_space_object(self):
        from core.search_space import SearchSpace
        optimizer = GeneticAlgorithmOptimizer(random_state=42)
        space = SearchSpace({'C': {'type': 'float', 'low': 0.01, 'high': 1.0}})
        cand = space.build_candidates(n=8)
        chrom = [0]
        params = optimizer._chromosome_to_params(chrom, space)
        self.assertIn('C', params)
        self.assertEqual(params['C'], cand['C'][0])

    def test_chromosome_to_params_with_dict(self):
        optimizer = GeneticAlgorithmOptimizer(random_state=42)
        space = {'C': [0.01, 0.1, 1.0]}
        chrom = [1]
        params = optimizer._chromosome_to_params(chrom, space)
        self.assertEqual(params['C'], 0.1)

    def test_tournament_select(self):
        optimizer = GeneticAlgorithmOptimizer(random_state=42)
        population = [[0, 1], [2, 3], [4, 5], [6, 7]]
        fitness = [0.1, 0.9, 0.5, 0.3]
        winner = optimizer._tournament_select(population, fitness, k=2)
        self.assertIn(winner, population)

    def test_crossover(self):
        optimizer = GeneticAlgorithmOptimizer(random_state=42)
        p1 = [0, 1, 2, 3]
        p2 = [4, 5, 6, 7]
        c1, c2 = optimizer._crossover(p1, p2)
        self.assertEqual(len(c1), 4)
        self.assertEqual(len(c2), 4)
        # 单点交叉，验证存在某个 point 使得前缀和后缀分别来自不同父代
        valid_point_found = False
        for point in range(1, 4):
            if (c1[:point] == p1[:point] and c1[point:] == p2[point:] and
                    c2[:point] == p2[:point] and c2[point:] == p1[point:]):
                valid_point_found = True
                break
        self.assertTrue(valid_point_found, f"c1={c1}, c2={c2} 不是合法的单点交叉结果")

    def test_mutate_with_search_space_object(self):
        from core.search_space import SearchSpace
        optimizer = GeneticAlgorithmOptimizer(random_state=42, mutation_rate=1.0)
        space = SearchSpace({'C': {'type': 'float', 'low': 0.01, 'high': 1.0}})
        chrom = [0]
        mutated = optimizer._mutate(chrom, space)
        self.assertEqual(len(mutated), 1)

    def test_mutate_with_dict(self):
        optimizer = GeneticAlgorithmOptimizer(random_state=42, mutation_rate=1.0)
        space = {'C': [0.01, 0.1, 1.0]}
        chrom = [0]
        mutated = optimizer._mutate(chrom, space)
        self.assertEqual(len(mutated), 1)


class TestOptimizerIntegration(unittest.TestCase):
    """集成测试：使用 mock 评估测试完整 optimize 流程（保证速度）"""

    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
        })
        self.y_cls = pd.Series(np.random.choice([0, 1], 50))
        self.y_reg = pd.Series(np.random.randn(50))

    def test_random_search_full_flow_classification(self):
        optimizer = RandomSearchOptimizer(n_trials=3, cv_folds=2, random_state=42)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION)
        self.assertEqual(result.model_key, 'lr')
        self.assertEqual(result.n_trials, 3)
        self.assertIsInstance(result.best_score, float)
        self.assertEqual(result.best_score, 0.85)

    def test_random_search_full_flow_regression(self):
        optimizer = RandomSearchOptimizer(n_trials=3, cv_folds=2, random_state=42)
        with patch.object(optimizer, '_evaluate_model', return_value=-0.5):
            result = optimizer.optimize('ridge', self.X, self.y_reg, TaskType.REGRESSION)
        self.assertEqual(result.model_key, 'ridge')
        self.assertEqual(result.n_trials, 3)

    def test_hyperband_full_flow(self):
        optimizer = HyperbandOptimizer(n_trials=9, cv_folds=2, random_state=42, max_resource=3)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION)
        self.assertEqual(result.model_key, 'lr')
        self.assertGreaterEqual(result.n_trials, 0)

    def test_genetic_full_flow(self):
        optimizer = GeneticAlgorithmOptimizer(n_trials=8, population_size=4, cv_folds=2, random_state=42)
        with patch.object(optimizer, '_evaluate_model', return_value=0.85):
            result = optimizer.optimize('lr', self.X, self.y_cls, TaskType.CLASSIFICATION)
        self.assertEqual(result.model_key, 'lr')
        self.assertGreaterEqual(result.n_trials, 0)


if __name__ == '__main__':
    unittest.main()
