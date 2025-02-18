'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/05/16 Copyright 2023 Capgemini

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
import logging
import unittest

import numpy as np
import pandas as pd
from numpy import arange
from sostrades_core.execution_engine.execution_engine import ExecutionEngine

from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)
from sostrades_optimization_plugins.sos_processes.test.test_sellar_opt_w_func_manager.usecase import (
    Study,
)


class TestFuncManager(unittest.TestCase):
    """
    FunctionManager test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'FuncManagerTest'
        self.func_manager = FunctionManager()

    def tearDown(self):
        pass

    def test_01_instantiate_func_manager(self):
        fail = True
        try:
            func_manager = FunctionManager()
            fail = False
        except:
            fail = True
        self.assertFalse(fail)

    def test_02_objective_functions(self):
        obj1 = 1.5
        obj2 = 1.

        self.func_manager.add_function(
            'obj1', None, FunctionManager.OBJECTIVE, weight=0.8)
        self.func_manager.update_function_value('obj1', obj1)

        self.func_manager.add_function(
            'obj2', None, FunctionManager.OBJECTIVE, weight=0.2)
        self.func_manager.update_function_value('obj2', obj2)

        self.func_manager.build_aggregated_functions(eps=1e-3)
        self.assertEqual(self.func_manager.mod_obj,
                         100 * (0.8 * obj1 + 0.2 * obj2))

    def test_03_ineq_functions(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT
        EQ_CONSTRAINT = self.func_manager.EQ_CONSTRAINT

        obj1 = 1.5
        obj2 = 1.
        cst1 = np.array([10., 200., -30.])
        cst2 = np.array([40000., 1., -10000.])
        cst3 = np.array([-10., 0.2, -5.])
        eqcst1 = np.array([-10., 10., -5.])
        eqcst2 = np.array([0.001, 0.001, 0.00001])

        self.func_manager.add_function(
            'obj1', None, FunctionManager.OBJECTIVE, weight=0.8)
        self.func_manager.update_function_value('obj1', obj1)

        self.func_manager.add_function(
            'obj2', None, FunctionManager.OBJECTIVE, weight=0.2)
        self.func_manager.update_function_value('obj2', obj2)

        self.func_manager.add_function(
            'cst1', None, FunctionManager.INEQ_CONSTRAINT)
        self.func_manager.update_function_value('cst1', cst1)

        self.func_manager.add_function(
            'cst2', None, FunctionManager.INEQ_CONSTRAINT)
        self.func_manager.update_function_value('cst2', cst2)

        self.func_manager.add_function(
            'cst3', None, FunctionManager.INEQ_CONSTRAINT)
        self.func_manager.update_function_value('cst3', cst3)

        self.func_manager.add_function(
            'eqcst1', None, FunctionManager.EQ_CONSTRAINT)
        self.func_manager.update_function_value('eqcst1', eqcst1)

        self.func_manager.add_function(
            'eqcst2', None, FunctionManager.EQ_CONSTRAINT)
        self.func_manager.update_function_value('eqcst2', eqcst2)

        self.func_manager.configure_smooth_log(True, 1e4)
        self.func_manager.set_aggregation_mods('sum', 'sum')
        self.func_manager.build_aggregated_functions(eps=1e-3)

        self.assertAlmostEqual(
            self.func_manager.aggregated_functions[OBJECTIVE], (0.8 * obj1 + 0.2 * obj2), delta=1e-6)
        self.assertGreater(
            self.func_manager.aggregated_functions[INEQ_CONSTRAINT], 0.)
        self.assertGreater(
            self.func_manager.aggregated_functions[EQ_CONSTRAINT], 0.)

        res = 100. * (self.func_manager.aggregated_functions[OBJECTIVE] +
                      self.func_manager.aggregated_functions[INEQ_CONSTRAINT] +
                      self.func_manager.aggregated_functions[EQ_CONSTRAINT])

        self.assertEqual(self.func_manager.mod_obj, res)

    def test_04_vest_obj(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        obj1 = np.array([1., 2., 3.])
        self.func_manager.add_function(
            'obj1', None, OBJECTIVE, weight=0.8)
        self.func_manager.update_function_value('obj1', obj1)
        self.func_manager.build_aggregated_functions(eps=1e-3)
        self.assertEqual(
            self.func_manager.aggregated_functions[OBJECTIVE], obj1.sum() * 0.8)

    def test_05_instantiate_func_manager_disc(self):
        try:
            FunctionManagerDisc(self.name, logging.getLogger(__name__))
            fail = False
        except:
            fail = True
        self.assertFalse(fail)

    def test_06_configure_func_manager_disc(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT
        EQ_CONSTRAINT = self.func_manager.EQ_CONSTRAINT
        OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name,
                   'ns_functions_2': self.name + '.' + 'FunctionManager2'}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        base_df = pd.DataFrame({'years': arange(10, 13)})
        obj1 = base_df.copy()
        obj1['obj1_values'] = 1.5
        obj2 = base_df.copy()
        obj2['obj2_values'] = 1.
        cst1 = base_df.copy()
        cst1['cst1_values'] = np.array([10., 200., -30.])
        cst2 = base_df.copy()
        cst2['cst2_values'] = np.array([40000., 1., -10000.])
        # cst2['cst2_valuesss'] = np.array([400000., 1., -10000.])

        cst3 = base_df.copy()
        cst3['cst3_values'] = np.array([-10., 0.2, -5.])
        eqcst1 = base_df.copy()
        eqcst1['eqcst1_values'] = np.array([-10., 10., -5.])
        eqcst2 = base_df.copy()
        eqcst2['eqcst2_values'] = np.array([0.001, 0.001, 0.00001])

        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
        func_df['variable'] = ['cst1', 'cst2', 'cst3',
                               'eqcst1', 'eqcst2', 'obj1', 'obj2']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT,
                            INEQ_CONSTRAINT, EQ_CONSTRAINT, EQ_CONSTRAINT, OBJECTIVE, OBJECTIVE]
        func_df['aggr'] = "sum"
        func_df['parent'] = 'obj'
        func_df['weight'] = [1., 1., 1., 1, 1, 0.8, 0.2]
        func_df['namespace'] = ['ns_functions'] * 6 + ['ns_functions_2']
        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'cst1'] = cst1
        values_dict[prefix + 'cst2'] = cst2
        values_dict[prefix + 'cst3'] = cst3
        values_dict[prefix + 'eqcst1'] = eqcst1
        values_dict[prefix + 'eqcst2'] = eqcst2
        values_dict[prefix + 'obj1'] = obj1
        values_dict[self.name + '.FunctionManager2.' + 'obj2'] = obj2

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

        # -- execution
        ee.execute()
        # -- retrieve outputs
        disc = ee.dm.get_disciplines_with_name(
            f'{self.name}.{func_mng_name}')[0]
        outputs = disc.get_sosdisc_outputs()

        # -- check outputs with reference data
        o1 = obj1['obj1_values'].to_numpy().sum()
        o2 = obj2['obj2_values'].to_numpy().sum()
        self.assertAlmostEqual(outputs[OBJECTIVE][0], 0.8 * o1 + 0.2 * o2)
        self.assertGreater(outputs[INEQ_CONSTRAINT][0], 0.)
        self.assertGreater(outputs[EQ_CONSTRAINT][0], 0.)

        res = 100. * (outputs[OBJECTIVE][0] +
                      outputs[INEQ_CONSTRAINT][0] +
                      outputs[EQ_CONSTRAINT][0])

        self.assertEqual(outputs[OBJECTIVE_LAGR][0], res)
        del (ee)
        del (self)

    def test_07_jacobian_func_manager_disc(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT
        EQ_CONSTRAINT = self.func_manager.EQ_CONSTRAINT
        OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        base_df = pd.DataFrame({'years': arange(10, 13)})
        obj1 = base_df.copy()
        obj1['obj1_values'] = 1.5
        obj2 = base_df.copy()
        obj2['obj2_values'] = 1.
        cst1 = base_df.copy()
        cst1['cst1_values'] = np.array([10., 200., -30.])
        cst2 = base_df.copy()
        cst2['cst2_values'] = np.array([40000., 1., -10000.])

        cst3 = base_df.copy()
        cst3['cst3_values'] = np.array([-10., 0.2, -5.])
        cst0 = base_df.copy()
        cst0['cst0_values'] = np.array([-10., 0.2, -5.])
        eqcst1 = base_df.copy()
        eqcst1['eqcst1_values'] = np.array([-10., 10., -5.])
        eqcst2 = base_df.copy()
        eqcst2['eqcst2_values'] = np.array([0.001, 0.001, 0.00001])

        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
        func_df['variable'] = ['cst0','cst1', 'cst2', 'cst3',
                               'eqcst1', 'eqcst2', 'obj1', 'obj2']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT, INEQ_CONSTRAINT,
                            INEQ_CONSTRAINT, EQ_CONSTRAINT, EQ_CONSTRAINT, OBJECTIVE, OBJECTIVE]
        func_df['weight'] = [1, 1., 1., 1., 1, 1, 0.8, 0.2]
        func_df['aggr'] = [FunctionManager.INEQ_NEGATIVE_WHEN_SATIFIED] + [FunctionManager.AGGR_TYPE_SUM] * 7
        func_df['parent'] = 'obj'
        func_df['namespace'] = ''


        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'cst1'] = cst1
        values_dict[prefix + 'cst2'] = cst2
        values_dict[prefix + 'cst3'] = cst3
        values_dict[prefix + 'cst0'] = cst0
        values_dict[prefix + 'eqcst1'] = eqcst1
        values_dict[prefix + 'eqcst2'] = eqcst2
        values_dict[prefix + 'obj1'] = obj1
        values_dict[prefix + 'obj2'] = obj2

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

        # -- execution
        ee.execute()
        # -- retrieve outputs
        disc = ee.dm.get_disciplines_with_name(
            f'{self.name}.{func_mng_name}')[0]
        outputs = disc.get_sosdisc_outputs()

        # -- check outputs with reference data
        o1 = obj1['obj1_values'].to_numpy().sum()
        o2 = obj2['obj2_values'].to_numpy().sum()
        self.assertAlmostEqual(outputs[OBJECTIVE][0], 0.8 * o1 + 0.2 * o2)
        self.assertGreater(outputs[INEQ_CONSTRAINT][0], 0.)
        self.assertGreater(outputs[EQ_CONSTRAINT][0], 0.)

        res = 100. * (outputs[OBJECTIVE][0] +
                      outputs[INEQ_CONSTRAINT][0] +
                      outputs[EQ_CONSTRAINT][0])

        self.assertEqual(outputs[OBJECTIVE_LAGR][0], res)

        disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        assert disc_techno.check_jacobian(
            input_data=disc_techno.local_data,
            threshold=1e-5, inputs=[
                'FuncManagerTest.FunctionManager.cst0',
                'FuncManagerTest.FunctionManager.cst1', 'FuncManagerTest.FunctionManager.cst2',
                                    'FuncManagerTest.FunctionManager.cst3', 'FuncManagerTest.FunctionManager.obj1',
                                    'FuncManagerTest.FunctionManager.obj2'],
            outputs=['FuncManagerTest.FunctionManager.objective_lagrangian'], derr_approx='complex_step')

    def test_08_jacobian_func_manager_disc2(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT
        EQ_CONSTRAINT = self.func_manager.EQ_CONSTRAINT
        OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        base_df = pd.DataFrame({'years': arange(10, 11)})
        obj1 = base_df.copy()
        obj1['obj1_values'] = 100
        obj2 = base_df.copy()
        obj2['obj2_values'] = 20
        cst1 = base_df.copy()
        cst1['cst1_values'] = -40
        cst2 = base_df.copy()
        cst2['cst2_values'] = 0.000001
        cst3 = base_df.copy()
        cst3['cst3_values'] = 4000
        cst4 = base_df.copy()
        cst4['cst4_values'] = -0.01
        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
        func_df['variable'] = ['cst1', 'cst2', 'cst3', 'cst4', 'obj1', 'obj2']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT,
                            INEQ_CONSTRAINT, INEQ_CONSTRAINT, OBJECTIVE, OBJECTIVE]
        func_df['weight'] = [-0.5, -1., -1., -1., 0.8, 0.2]

        func_df['aggr'] = "sum"
        func_df['parent'] = 'obj'
        func_df['namespace'] = ''

        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'cst1'] = cst1
        values_dict[prefix + 'cst2'] = cst2
        values_dict[prefix + 'cst3'] = cst3
        values_dict[prefix + 'cst4'] = cst4
        values_dict[prefix + 'obj1'] = obj1
        values_dict[prefix + 'obj2'] = obj2

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

        # -- execution
        ee.execute()
        # -- retrieve outputs
        disc = ee.dm.get_disciplines_with_name(
            f'{self.name}.{func_mng_name}')[0]
        outputs = disc.get_sosdisc_outputs()

        # -- check outputs with reference data
        o1 = obj1['obj1_values'].to_numpy().sum()
        o2 = obj2['obj2_values'].to_numpy().sum()
        self.assertAlmostEqual(outputs[OBJECTIVE][0], 0.8 * o1 + 0.2 * o2)
        self.assertGreater(outputs[INEQ_CONSTRAINT][0], 0.)

        res = 100. * (outputs[OBJECTIVE][0] +
                      outputs[INEQ_CONSTRAINT][0] +
                      outputs[EQ_CONSTRAINT][0])

        self.assertEqual(outputs[OBJECTIVE_LAGR][0], res)

        disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        assert disc_techno.check_jacobian(
            input_data=disc_techno.local_data,
            threshold=1e-5, inputs=['FuncManagerTest.FunctionManager.cst1', 'FuncManagerTest.FunctionManager.cst2',
                                    'FuncManagerTest.FunctionManager.cst3', 'FuncManagerTest.FunctionManager.cst4',
                                    'FuncManagerTest.FunctionManager.obj1', 'FuncManagerTest.FunctionManager.obj2'],
            outputs=['FuncManagerTest.FunctionManager.objective_lagrangian'], derr_approx='complex_step')

    def test_09_inf_nan_values(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT
        EQ_CONSTRAINT = self.func_manager.EQ_CONSTRAINT

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        base_df = pd.DataFrame({'years': arange(10, 13)})
        obj1 = base_df.copy()
        obj1['obj1_values'] = 1.5
        obj2 = base_df.copy()
        obj2['obj2_values'] = 1.
        cst1 = base_df.copy()
        cst1['cst1_values'] = np.array([np.nan, 200., -30.])
        cst2 = base_df.copy()
        cst2['cst2_values'] = np.array([40000., 1., -10000.])
        cst3 = base_df.copy()
        cst3['cst3_values'] = np.array([-10., 0.2, -5.])
        eqcst1 = base_df.copy()
        eqcst1['eqcst1_values'] = np.array([-10., 10., -5.])
        eqcst2 = base_df.copy()
        eqcst2['eqcst2_values'] = np.array([0.001, np.inf, 0.00001])

        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
        func_df['variable'] = ['cst1', 'cst2', 'cst3',
                               'eqcst1', 'eqcst2', 'obj1', 'obj2']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT,
                            INEQ_CONSTRAINT, EQ_CONSTRAINT, EQ_CONSTRAINT, OBJECTIVE, OBJECTIVE]
        func_df['weight'] = [1, 1, 1, 1, 1, 0.8, 0.2]
        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'cst1'] = cst1
        values_dict[prefix + 'cst2'] = cst2
        values_dict[prefix + 'cst3'] = cst3
        values_dict[prefix + 'eqcst1'] = eqcst1
        values_dict[prefix + 'eqcst2'] = eqcst2
        values_dict[prefix + 'obj1'] = obj1
        values_dict[prefix + 'obj2'] = obj2

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

    def test_10_jacobian_func_manager_disc_different_aggr(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT
        EQ_CONSTRAINT = self.func_manager.EQ_CONSTRAINT

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        base_df = pd.DataFrame({'years': arange(10, 13)})
        obj1 = base_df.copy()
        obj1['obj1_values'] = 1.5
        obj2 = base_df.copy()
        obj2['obj2_values'] = 1.
        cst1 = base_df.copy()
        cst1['cst1_values'] = np.array([10., 200., -30.])
        cst2 = base_df.copy()
        cst2['cst2_values'] = np.array([40000., 1., -10000.])

        cst3 = base_df.copy()
        cst3['cst3_values'] = np.array([-10., 0.2, -5.])
        eqcst1 = base_df.copy()
        eqcst1['eqcst1_values'] = np.array([-10., 10., -5.])
        eqcst2 = base_df.copy()
        eqcst2['eqcst2_values'] = np.array([0.001, 0.001, 0.00001])

        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight', 'aggr'])
        func_df['variable'] = ['cst1', 'cst2', 'cst3',
                               'eqcst1', 'eqcst2', 'obj1', 'obj2']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT,
                            INEQ_CONSTRAINT, EQ_CONSTRAINT, EQ_CONSTRAINT, OBJECTIVE, OBJECTIVE]
        func_df['weight'] = [1., 1., 1., 1, 1, 0.8, 0.2]
        func_df['aggr'] = "sum"
        func_df['namespace'] = ''
        func_df['parent'] = ['ineqcst', 'ineqcst', 'ineqcst', 'eqcst', 'eqcst', 'obj', 'obj']
        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'cst1'] = cst1
        values_dict[prefix + 'cst2'] = cst2
        values_dict[prefix + 'cst3'] = cst3
        values_dict[prefix + 'eqcst1'] = eqcst1
        values_dict[prefix + 'eqcst2'] = eqcst2
        values_dict[prefix + 'obj1'] = obj1
        values_dict[prefix + 'obj2'] = obj2

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

        # -- execution
        ee.execute()
        # -- retrieve outputs
        disc = ee.dm.get_disciplines_with_name(
            f'{self.name}.{func_mng_name}')[0]
        outputs = disc.get_sosdisc_outputs()

        # -- check outputs with reference data
        o1 = obj1['obj1_values'].to_numpy().sum()
        o2 = obj2['obj2_values'].to_numpy().sum()

        res = 100. * (outputs[OBJECTIVE][0] +
                      outputs[INEQ_CONSTRAINT][0] +
                      outputs[EQ_CONSTRAINT][0])

        disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        assert disc_techno.check_jacobian(
            input_data=disc_techno.local_data,
            threshold=1e-5, inputs=['FuncManagerTest.FunctionManager.cst1', 'FuncManagerTest.FunctionManager.cst2',
                                    'FuncManagerTest.FunctionManager.cst3', 'FuncManagerTest.FunctionManager.obj1',
                                    'FuncManagerTest.FunctionManager.obj2'],
            outputs=['FuncManagerTest.FunctionManager.objective_lagrangian'], derr_approx='complex_step')

        # get charts
        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)

    def test_11_jacobian_eq_delta_and_lin_to_quad(self):
        OBJECTIVE = self.func_manager.OBJECTIVE
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT
        EQ_CONSTRAINT = self.func_manager.EQ_CONSTRAINT

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        base_df = pd.DataFrame({'years': arange(10, 13)})
        obj1 = base_df.copy()
        obj1['obj1_values'] = 1.5
        obj2 = base_df.copy()
        obj2['obj2_values'] = 1.
        ineq_cst = base_df.copy()
        ineq_cst['ineq_cst_values'] = np.array([10., -2000., -30.])
        ineq_cst0 = np.array([10., 2000., -30.])
        eqcst_delta = base_df.copy()
        eqcst_delta['eqcst_delta_values'] = np.array([400., 1., -10.])
        eqcst_delta2 = base_df.copy()
        eqcst_delta2['eqcst_delta2_values'] = np.array([0.0001, 1., -0.00003])
        eqcst_delta_array = np.array([-10., -200000., -5.])
        eqcst_lintoquad = base_df.copy()
        eqcst_lintoquad['eqcst_lintoquad_values'] = np.array([-1., 2., 0.03])
        eqcst_lintoquad_array = np.array([-0.2, -50., 100.])

        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight', 'aggr'])
        func_df['variable'] = ['ineq_cst', 'ineq_cst0', 'eqcst_delta', 'eqcst_delta2',
                               'eqcst_delta_array', 'eqcst_lintoquad', 'eqcst_lintoquad_array',
                               'obj1', 'obj2']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT,
                            EQ_CONSTRAINT, EQ_CONSTRAINT, EQ_CONSTRAINT, EQ_CONSTRAINT, EQ_CONSTRAINT,
                            OBJECTIVE, OBJECTIVE]
        func_df['weight'] = [0.5, -1., -0.2, 0.2, 1.2, -1.0, 0.01, 0.8, 0.2]
        func_df['aggr'] = ['sum', 'sum', 'sum', 'sum', 'delta', 'lin_to_quad', 'lin_to_quad', 'smax', 'sum']
        func_df['parent'] = 'parent'
        func_df['namespace'] = ''
        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'ineq_cst'] = ineq_cst
        values_dict[prefix + 'ineq_cst0'] = ineq_cst0
        values_dict[prefix + 'eqcst_delta'] = eqcst_delta
        values_dict[prefix + 'eqcst_delta2'] = eqcst_delta2
        values_dict[prefix + 'eqcst_delta_array'] = eqcst_delta_array
        values_dict[prefix + 'eqcst_lintoquad'] = eqcst_lintoquad
        values_dict[prefix + 'eqcst_lintoquad_array'] = eqcst_lintoquad_array
        values_dict[prefix + 'obj1'] = obj1
        values_dict[prefix + 'obj2'] = obj2
        values_dict[prefix + 'aggr_mod_eq'] = 'sum'
        values_dict[prefix + 'aggr_mod_ineq'] = 'smooth_max'

        ee.load_study_from_input_dict(values_dict)

        ee.dm.set_data(prefix + 'ineq_cst0', 'type', 'array')
        ee.dm.set_data(prefix + 'eqcst_delta_array', 'type', 'array')
        ee.dm.set_data(prefix + 'eqcst_lintoquad_array', 'type', 'array')

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

        # -- execution
        ee.execute()
        # -- retrieve outputs
        disc = ee.dm.get_disciplines_with_name(
            f'{self.name}.{func_mng_name}')[0]
        outputs = disc.get_sosdisc_outputs()

        # -- check outputs with reference data
        o1 = obj1['obj1_values'].to_numpy().sum()
        o2 = obj2['obj2_values'].to_numpy().sum()

        res = 100. * (outputs[OBJECTIVE][0] +
                      outputs[INEQ_CONSTRAINT][0] +
                      outputs[EQ_CONSTRAINT][0])

        disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        assert disc_techno.check_jacobian(
            input_data=disc_techno.local_data,
            threshold=1e-8, inputs=['FuncManagerTest.FunctionManager.ineq_cst',
                                    'FuncManagerTest.FunctionManager.ineq_cst0',
                                    'FuncManagerTest.FunctionManager.eqcst_delta',
                                    'FuncManagerTest.FunctionManager.eqcst_delta2',
                                    'FuncManagerTest.FunctionManager.eqcst_delta_array',
                                    'FuncManagerTest.FunctionManager.eqcst_lintoquad',
                                    'FuncManagerTest.FunctionManager.eqcst_lintoquad_array',
                                    'FuncManagerTest.FunctionManager.obj1',
                                    'FuncManagerTest.FunctionManager.obj2'],
            outputs=['FuncManagerTest.FunctionManager.objective_lagrangian',
                     'FuncManagerTest.FunctionManager.eq_constraint',
                     'FuncManagerTest.FunctionManager.ineq_constraint',
                     ],
            step=1e-15, derr_approx='complex_step')

    def test_12_test_number_iteration_output_optim_df(self):
        self.name = 'Test12'
        self.ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + 'FunctionManager2'}
        self.ee.ns_manager.add_ns_def(ns_dict)
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

        self.ee.load_study_from_input_dict(full_values_dict)

        self.ee.execute()

        funcmanager = self.ee.root_process.proxy_disciplines[0].proxy_disciplines[0].proxy_disciplines[-1]
        optim_iter = funcmanager.discipline_wrapp.discipline.sos_wrapp.iter
        optim_name = "SellarOptimScenario"
        optim_output_df = self.ee.dm.get_value(
            f'{self.name}.{optim_name}.SellarCoupling.FunctionManager.{FunctionManagerDisc.OPTIM_OUTPUT_DF}')
        self.assertEqual(optim_iter + 1, len(optim_output_df))
        # get charts
        func_disc = self.ee.dm.get_disciplines_with_name(f'{self.name}.{optim_name}.SellarCoupling.FunctionManager')[0]
        filter = func_disc.get_chart_filter_list()
        graph_list = func_disc.get_post_processing_list(filter)

    def test_13_jacobian_func_manager_disc_ineq_constraint_negative_when_satisfied(self):
        INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT

        # -- init the case
        func_mng_name = 'FunctionManager'
        prefix = self.name + '.' + func_mng_name + '.'

        ee = ExecutionEngine(self.name)
        ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                   'ns_optim': self.name + '.' + func_mng_name}
        ee.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
        fm_builder = ee.factory.get_builder_from_module(
            'FunctionManager', mod_list)
        ee.factory.set_builders_to_coupling_builder(fm_builder)
        ee.configure()

        # -- i/o setup
        base_df = pd.DataFrame({'years': arange(10, 13)})
        cst0 = base_df.copy()
        cst0['cst0_values'] = np.array([-10., 1, -5.])

        # -- ~GUI inputs: selection of functions

        func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
        func_df['variable'] = ['cst0']
        func_df['ftype'] = [INEQ_CONSTRAINT]
        func_df['weight'] = [2.]
        func_df['aggr'] = [FunctionManager.INEQ_NEGATIVE_WHEN_SATIFIED]
        func_df['parent'] = 'obj'
        func_df['namespace'] = ''


        values_dict = {}
        values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

        # -- data to simulate disciplinary chain outputs
        values_dict[prefix + 'cst0'] = cst0

        ee.load_study_from_input_dict(values_dict)

        ee.display_treeview_nodes(True)

        # -- execution
        ee.execute()
        # -- retrieve outputs
        disc = ee.dm.get_disciplines_with_name(
            f'{self.name}.{func_mng_name}')[0]
        disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

        assert disc_techno.check_jacobian(
            input_data=disc_techno.local_data,
            threshold=1e-5, inputs=['FuncManagerTest.FunctionManager.cst0'],
            outputs=['FuncManagerTest.FunctionManager.objective_lagrangian'], derr_approx='complex_step', step=1e-15)

    def test_14_jacobian_func_manager_disc_ineq_constraint_positive_when_satisfied(self):
            INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT

            # -- init the case
            func_mng_name = 'FunctionManager'
            prefix = self.name + '.' + func_mng_name + '.'

            ee = ExecutionEngine(self.name)
            ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                       'ns_optim': self.name + '.' + func_mng_name}
            ee.ns_manager.add_ns_def(ns_dict)

            mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
            fm_builder = ee.factory.get_builder_from_module(
                'FunctionManager', mod_list)
            ee.factory.set_builders_to_coupling_builder(fm_builder)
            ee.configure()

            # -- i/o setup
            base_df = pd.DataFrame({'years': arange(10, 13)})
            cst0 = base_df.copy()
            cst0['cst0_values'] = np.array([-10., 0.2, -5.])

            # -- ~GUI inputs: selection of functions

            func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
            func_df['variable'] = ['cst0']
            func_df['ftype'] = [INEQ_CONSTRAINT]
            func_df['weight'] = [3.]
            func_df['aggr'] = [FunctionManager.INEQ_POSITIVE_WHEN_SATIFIED]
            func_df['parent'] = 'obj'
            func_df['namespace'] = ''

            values_dict = {}
            values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

            # -- data to simulate disciplinary chain outputs
            values_dict[prefix + 'cst0'] = cst0

            ee.load_study_from_input_dict(values_dict)

            ee.display_treeview_nodes(True)

            # -- execution
            ee.execute()
            # -- retrieve outputs
            disc = ee.dm.get_disciplines_with_name(
                f'{self.name}.{func_mng_name}')[0]
            disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

            assert disc_techno.check_jacobian(
                input_data=disc_techno.local_data,
                threshold=1e-5, inputs=['FuncManagerTest.FunctionManager.cst0'],
                outputs=['FuncManagerTest.FunctionManager.objective_lagrangian'], derr_approx='complex_step')

    def test_16_jacobian_func_manager_disc_ineq_constraint_negative_when_satisfied_square(self):
            INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT

            # -- init the case
            func_mng_name = 'FunctionManager'
            prefix = self.name + '.' + func_mng_name + '.'

            ee = ExecutionEngine(self.name)
            ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                       'ns_optim': self.name + '.' + func_mng_name}
            ee.ns_manager.add_ns_def(ns_dict)

            mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
            fm_builder = ee.factory.get_builder_from_module(
                'FunctionManager', mod_list)
            ee.factory.set_builders_to_coupling_builder(fm_builder)
            ee.configure()

            # -- i/o setup
            base_df = pd.DataFrame({'years': arange(10, 13)})
            cst0 = base_df.copy()
            cst0['cst0_values'] = np.array([-10., 0.2, -5.])

            # -- ~GUI inputs: selection of functions

            func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
            func_df['variable'] = ['cst0']
            func_df['ftype'] = [INEQ_CONSTRAINT]
            func_df['weight'] = [2.5]
            func_df['aggr'] = [FunctionManager.INEQ_NEGATIVE_WHEN_SATIFIED_AND_SQUARE_IT]
            func_df['parent'] = 'obj'
            func_df['namespace'] = ''

            values_dict = {}
            values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

            # -- data to simulate disciplinary chain outputs
            values_dict[prefix + 'cst0'] = cst0

            ee.load_study_from_input_dict(values_dict)

            ee.display_treeview_nodes(True)

            # -- execution
            ee.execute()
            # -- retrieve outputs
            disc = ee.dm.get_disciplines_with_name(
                f'{self.name}.{func_mng_name}')[0]
            disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

            assert disc_techno.check_jacobian(
                input_data=disc_techno.local_data,
                threshold=1e-5, inputs=['FuncManagerTest.FunctionManager.cst0'],
                outputs=['FuncManagerTest.FunctionManager.objective_lagrangian'], derr_approx='complex_step')

    def test_17_jacobian_func_manager_disc_ineq_constraint_positive_when_satisfied_square(self):
            INEQ_CONSTRAINT = self.func_manager.INEQ_CONSTRAINT

            # -- init the case
            func_mng_name = 'FunctionManager'
            prefix = self.name + '.' + func_mng_name + '.'

            ee = ExecutionEngine(self.name)
            ns_dict = {'ns_functions': self.name + '.' + func_mng_name,
                       'ns_optim': self.name + '.' + func_mng_name}
            ee.ns_manager.add_ns_def(ns_dict)

            mod_list = 'sostrades_optimization_plugins.models.func_manager.func_manager_disc.FunctionManagerDisc'
            fm_builder = ee.factory.get_builder_from_module(
                'FunctionManager', mod_list)
            ee.factory.set_builders_to_coupling_builder(fm_builder)
            ee.configure()

            # -- i/o setup
            base_df = pd.DataFrame({'years': arange(10, 13)})
            cst0 = base_df.copy()
            cst0['cst0_values'] = np.array([-10., 0.2, -5.])

            # -- ~GUI inputs: selection of functions

            func_df = pd.DataFrame(columns=['variable', 'ftype', 'weight'])
            func_df['variable'] = ['cst0']
            func_df['ftype'] = [INEQ_CONSTRAINT]
            func_df['weight'] = [2.]
            func_df['aggr'] = [FunctionManager.INEQ_POSITIVE_WHEN_SATIFIED_AND_SQUARE_IT]
            func_df['parent'] = 'obj'
            func_df['namespace'] = ''

            values_dict = {}
            values_dict[prefix + FunctionManagerDisc.FUNC_DF] = func_df

            # -- data to simulate disciplinary chain outputs
            values_dict[prefix + 'cst0'] = cst0

            ee.load_study_from_input_dict(values_dict)

            ee.display_treeview_nodes(True)

            # -- execution
            ee.execute()
            # -- retrieve outputs
            disc = ee.dm.get_disciplines_with_name(
                f'{self.name}.{func_mng_name}')[0]
            disc_techno = ee.root_process.proxy_disciplines[0].discipline_wrapp.discipline

            assert disc_techno.check_jacobian(
                input_data=disc_techno.local_data,
                threshold=1e-5, inputs=['FuncManagerTest.FunctionManager.cst0'],
                outputs=['FuncManagerTest.FunctionManager.objective_lagrangian'], derr_approx='complex_step')
