'''
Copyright 2022 Airbus SAS
Modifications on 2023/01/24-2024/05/16 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_optimization_plugins.sos_processes.test.test_sellar_opt_w_func_manager.usecase import (
    Study,
)

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for optimization scenario
"""


class TestSoSOptimScenarioWithFuncManager(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def test_18_optim_scenario_optim_algo_projected_gradient_func_manager(self):
        self.name = 'Test12'
        self.ee = ExecutionEngine(self.name)

        builder = self.ee.factory.get_builder_from_process('sostrades_optimization_plugins.sos_processes.test',
                                                           'test_sellar_opt_w_func_manager'
                                                           )
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        usecase = Study(execution_engine=self.ee)
        usecase.study_name = self.name

        values_dict = usecase.setup_usecase()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        full_values_dict.update({
            f"{self.name}.SellarOptimScenario.{'max_iter'}": 67,
            f"{self.name}.SellarOptimScenario.{'algo'}": 'ProjectedGradient',
        })
        self.ee.load_study_from_input_dict(full_values_dict)

        self.ee.execute()

        proxy_optim = self.ee.root_process.proxy_disciplines[0]
        filters = proxy_optim.get_chart_filter_list()
        graph_list = proxy_optim.get_post_processing_list(filters)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass

if '__main__' == __name__:
    cls = TestSoSOptimScenarioWithFuncManager()
    cls.setUp()

