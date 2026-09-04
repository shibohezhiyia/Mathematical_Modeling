"""
Unit tests for core/problem_solver.py (universal version)
"""
import unittest

from core.problem_solver import analyze_problem, generate_modeling_report


class TestProblemSolverUniversal(unittest.TestCase):
    def test_interception_screening(self):
        desc = '无人机投放烟幕干扰弹在来袭武器和保护目标之间形成遮蔽，使得有效遮蔽时间尽可能长'
        result = analyze_problem(desc)
        self.assertIn(result['task_type'], ['optimization', 'differential_equations', 'simulation', 'evaluation_ranking'])
        self.assertGreater(result['confidence'], 0)
        self.assertTrue(len(result['objectives']) > 0)
        self.assertTrue(len(result['steps']) > 5)
        self.assertTrue(len(result['code_framework']) > 100)

    def test_traffic_optimization(self):
        desc = '城市交通拥堵严重，如何优化信号灯配时方案，使得车辆平均通行时间最短'
        result = analyze_problem(desc)
        self.assertEqual(result['task_type'], 'optimization')
        self.assertIn('数学优化模型', result['model_class'])

    def test_epidemic_prediction(self):
        desc = '预测未来30天内新冠疫情感染人数的变化趋势，建立传播模型'
        result = analyze_problem(desc)
        self.assertIn(result['task_type'], ['prediction_forecast', 'differential_equations'])

    def test_sales_forecast(self):
        desc = '根据历史销售数据，预测下季度各产品的销量'
        result = analyze_problem(desc)
        self.assertEqual(result['task_type'], 'prediction_forecast')

    def test_data_requirements_is_not_misclassified_as_optimization(self):
        desc = (
            '为了更好地制定上述补货和定价决策，还需要采集哪些数据？'
            '这些数据对解决上述问题有什么帮助？请给出意见和理由。'
        )
        result = analyze_problem(desc)
        self.assertEqual(result['task_type'], 'data_requirements')
        self.assertEqual(result['task_graph'][0]['task_type'], 'data_requirements')
        self.assertEqual(result['task_graph'][0]['depends_on'], [])

    def test_bare_motion_speed_does_not_force_optimization(self):
        desc = '物体以恒定速度运动，建立位置随时间变化的微分方程并计算轨迹'
        result = analyze_problem(desc)
        self.assertEqual(result['task_type'], 'differential_equations')

    def test_carbon_evaluation(self):
        desc = '对10个城市的碳排放水平进行综合评价和排名'
        result = analyze_problem(desc)
        self.assertEqual(result['task_type'], 'evaluation_ranking')

    def test_report_generation(self):
        report = generate_modeling_report('优化资源配置，使得总成本最小')
        self.assertIn('任务类型', report)
        self.assertIn('建模步骤', report)
        self.assertIn('Python 代码框架', report)


if __name__ == '__main__':
    unittest.main()
