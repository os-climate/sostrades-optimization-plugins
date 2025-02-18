'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
import pandas as pd
from numpy import array
from sostrades_core.study_manager.study_manager import StudyManager

from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)
        self.optim_name = "SellarOptimScenario"
        self.subcoupling_name = "SellarCoupling"
        self.coupling_name = "Sellar_Problem"

    def setup_usecase(self):

        INEQ_CONSTRAINT = FunctionManager.INEQ_CONSTRAINT
        OBJECTIVE = FunctionManager.OBJECTIVE
        ns = f'{self.study_name}'
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [5.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        #                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        disc_dict = {}
        # Optim inputs
        disc_dict[f'{ns}.{self.optim_name}.algo'] = "SLSQP"
        disc_dict[f'{ns}.{self.optim_name}.design_space'] = dspace
        # TODO: what's wrong with IDF
        disc_dict[f'{ns}.{self.optim_name}.formulation'] = 'DisciplinaryOpt'
        # f'{ns}.{optim_name}.obj'
        disc_dict[f'{ns}.{self.optim_name}.objective_name'] = 'objective_lagrangian'
        disc_dict[f'{ns}.{self.optim_name}.ineq_constraints'] = [
        ]
        # f'{ns}.{self.optim_name}.c_1', f'{ns}.{self.optim_name}.c_2']

        disc_dict[f'{ns}.{self.optim_name}.algo_options'] = {"ftol_rel": 1e-10,
                                                               "ineq_tolerance": 2e-3,
                                                               "normalize_design_space": False}

        # Sellar inputs
        disc_dict[f'{ns}.{self.optim_name}.{self.subcoupling_name}.x'] = array([1.])
        disc_dict[f'{ns}.{self.optim_name}.{self.subcoupling_name}.z'] = array([1., 1.])
        disc_dict[f'{ns}.{self.optim_name}.{self.subcoupling_name}.{self.coupling_name}.local_dv'] = 10.
        disc_dict[f'{ns}.{self.optim_name}.{self.subcoupling_name}.sub_mda_class'] = 'PureNewtonRaphson'
        disc_dict[f'{ns}.{self.optim_name}.max_iter'] = 2

        func_df = pd.DataFrame(
            columns=['variable', 'ftype', 'weight', AGGR_TYPE])
        func_df['variable'] = ['c_1', 'c_2', 'obj']
        func_df['parent'] = "parent"
        func_df['namespace'] = "ns_functions"
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT, OBJECTIVE]
        func_df['weight'] = [200, 0.000001, 0.1]
        func_df[AGGR_TYPE] = [AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM]
        func_mng_name = 'FunctionManager'

        prefix = self.study_name + f'.{self.optim_name}.' + f'{self.subcoupling_name}.' + func_mng_name + '.'
        values_dict = {}
        values_dict[prefix +
                    FunctionManagerDisc.FUNC_DF] = func_df

        disc_dict.update(values_dict)
        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
