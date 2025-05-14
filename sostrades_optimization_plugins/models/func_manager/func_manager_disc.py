'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/04-2024/05/17 Copyright 2023 Capgemini

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
import csv
import logging
import time
import warnings
from copy import deepcopy
from math import isnan

import numpy as np
import pandas as pd
from numpy import asarray, float64, ndarray
from plotly import graph_objects as go
from sostrades_core.execution_engine.optim_manager_disc import OptimManagerDisc
from sostrades_core.tools.base_functions.exp_min import (
    compute_dfunc_with_exp_min,
    compute_func_with_exp_min,
)
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)

from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.tools.cst_manager.func_manager_common import (
    derivative_d_keep_negative_only,
    derivative_d_keep_positive_only,
    derivative_d_keep_positive_only_square,
    get_dsmooth_dvariable,
    keep_negative_only,
    keep_positive_only,
    keep_positive_only_square,
    smooth_maximum,
)

warnings.simplefilter(action='ignore', category=FutureWarning)


class FunctionManagerDisc(OptimManagerDisc):
    """
    Constraints aggregation discipline
    """

    # ontology information
    _ontology_data = {
        'label': 'Core Function Manager',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-chart-line fa-fw',
        'version': '',
    }
    MOD_SUFFIX = '_mod'
    INEQ_CONSTRAINT = FunctionManager.INEQ_CONSTRAINT
    EQ_CONSTRAINT = FunctionManager.EQ_CONSTRAINT
    OBJECTIVE = FunctionManager.OBJECTIVE
    OBJECTIVE_LAGR = OBJECTIVE + '_lagrangian'

    PARAMETER_LIST = [OBJECTIVE_LAGR, OBJECTIVE,
                      INEQ_CONSTRAINT, EQ_CONSTRAINT]
    FUNC_DF = 'function_df'
    VARIABLE = 'variable'
    FTYPE = 'ftype'
    WEIGHT = 'weight'
    INDEX = 'index'
    NAMESPACE_VARIABLE = 'namespace'
    COMPONENT = 'component'
    AGGR_TYPE = 'aggr'
    PARENT = 'parent'
    CHILDREN = 'children'
    OPTIM_OUTPUT_DF = 'optim_output_df'
    EXPORT_CSV = 'export_csv'

    AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
    AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX

    NS_OPTIM = 'ns_optim'
    DESC_IN = {FUNC_DF: {'type': 'dataframe',
                         'dataframe_descriptor': {VARIABLE: ('string', None, True),  # input function
                                                  FTYPE: ('string', None, True),
                                                  WEIGHT: ('float', None, True),
                                                  AGGR_TYPE: ('string', None, True),
                                                  PARENT: ('string', None, True),
                                                  # index of the dataframe
                                                  # INDEX: ('int', None, True),
                                                  NAMESPACE_VARIABLE: ('string', None, True),
                                                  },  # col name of the dataframe
                         'dataframe_edition_locked': False,
                         'structuring': True
                         },
               EXPORT_CSV: {'type': 'bool', 'default': False},
               'smooth_log': {'type': 'bool', 'default': False, 'user_level': 3},
               'eps2': {'type': 'float', 'default': 1e10, 'user_level': 3},
               'aggr_mod_ineq': {'type': 'string', 'default': AGGR_TYPE_SUM, 'user_level': 3},
               'aggr_mod_eq': {'type': 'string', 'default': AGGR_TYPE_SUM, 'user_level': 3},
               }
    DESC_OUT = {OPTIM_OUTPUT_DF: {'type': 'dataframe'}}

    def __init__(self, sos_name, logger: logging.Logger):
        '''
        constructor
        '''
        super().__init__(sos_name=sos_name, logger=logger)
        self.function_dict = None
        self.func_manager = FunctionManager()
        self.__formulation = None
        self.old_iter = -1

    def set_optim_formulation(self, formulation):
        '''

        Set the formulation object (sent by proxyoptim and built by mdoscenario of gemseo
        '''
        self.__formulation = formulation

    def get_current_iter(self):
        '''
        Get current iter from the optimisation problem in the formulation object built by GEMSEO
        If no formulation then we compute in the old way, iter is the number of method calls
        '''
        # if self.__formulation is not None:
        #     return self.__formulation.optimization_problem.current_iter
        # else:
        self.iter += 1
        return self.iter

    def setup_sos_disciplines(self):
        '''

        Specific configure for the func manager discipline

        '''
        dynamic_inputs, dynamic_outputs = {}, {}

        # initialization of func_manager
        self.func_manager.reinit()

        inputs_dict = self.get_sosdisc_inputs()

        if 'eps2' in inputs_dict.keys():
            self.func_manager.configure_smooth_log(inputs_dict['smooth_log'], inputs_dict['eps2'])

        # retrieve all the function descriptions
        if self.FUNC_DF in inputs_dict.keys():
            func_df = inputs_dict[self.FUNC_DF]
            if func_df is not None:
                if len(func_df) == 0:
                    # otherwise it will cause problems in gemseo gradients chain rule computation
                    raise Exception(f"{self.FUNC_DF} can not be empty.")

                list_var = list(func_df[self.VARIABLE])

                # create func_dict from func_df
                func_dict = {}

                # loop on variables (function names),
                # to get the indices (INDEX) and columns (COMPONENT)
                for var in list_var:
                    ftype = func_df.loc[func_df[self.VARIABLE]
                                        == var, self.FTYPE].values[0]
                    weight = func_df.loc[func_df[self.VARIABLE]
                                         == var, self.WEIGHT].values[0]
                    func_dict[var] = {self.FTYPE: ftype, self.WEIGHT: weight}

                    if self.INDEX in func_df:
                        index = func_df.loc[func_df[self.VARIABLE]
                                            == var, self.INDEX].values[0]
                        if index is not None and not isinstance(index, str):
                            if not isnan(index):
                                func_dict[var].update({self.INDEX: index})

                    if self.COMPONENT in func_df:
                        component = func_df.loc[func_df[self.VARIABLE]
                                                == var, self.COMPONENT].values[0]
                        if component is not None:
                            func_dict[var].update({self.COMPONENT: component})

                self.function_dict = func_dict
                # -- update all i/o function per function
                for f, metadata in self.function_dict.items():
                    # TODO: improve by retrieving desc_i/o info from disciplines
                    # instead of assuming that they all are dataframes

                    # get namespace of the variable
                    # if namespace column is in the dataframe
                    if self.NAMESPACE_VARIABLE in func_df.columns:
                        namespace = func_df.loc[func_df[self.VARIABLE]
                                                == f, self.NAMESPACE_VARIABLE].values[0]
                        if not isinstance(namespace, str) or namespace == '':
                            namespace = 'ns_functions'
                    # default namespace is ns_functions
                    else:
                        namespace = 'ns_functions'

                    namespaces = self.dm.get_all_namespaces_from_var_name(f)

                    if namespaces != []:
                        variable_full_name = namespaces[0]
                        var_type = self.dm.get_data(variable_full_name)['type']
                        dynamic_inputs[f] = {
                            self.TYPE: var_type, self.STRUCTURING: True, self.VISIBILITY: self.SHARED_VISIBILITY,
                            self.NAMESPACE: namespace}
                    else:
                        dynamic_inputs[f] = {
                            self.TYPE: 'dataframe', self.DATAFRAME_DESCRIPTOR: {},
                            self.DYNAMIC_DATAFRAME_COLUMNS: True, self.STRUCTURING: True,
                            self.VISIBILITY: self.SHARED_VISIBILITY,
                            self.NAMESPACE: namespace}

                    # -- output update : constr aggregation and scalarized objective
                    out_name = self.__build_mod_names(f)
                    dynamic_outputs[out_name] = {self.TYPE: 'array', self.VISIBILITY: self.SHARED_VISIBILITY,
                                                 self.NAMESPACE: self.NS_OPTIM}
                    # -- add function to the FuncManager
                    ftype = metadata[FunctionManager.FTYPE]
                    w = metadata.get(FunctionManager.WEIGHT, None)

                    aggr_type = self.AGGR_TYPE_SUM
                    if self.AGGR_TYPE in func_df.columns:
                        aggr_type = func_df.loc[func_df[self.VARIABLE]
                                                == f, self.AGGR_TYPE].values[0]
                        if pd.isnull(aggr_type):
                            aggr_type = self.AGGR_TYPE_SUM
                    if w is None:
                        w = 1.
                    self.func_manager.add_function(
                        f, value=None, ftype=ftype, weight=w, aggr_type=aggr_type)

        # -- output update : aggregation of ineq constraints
        dynamic_outputs[self.INEQ_CONSTRAINT] = {'type': 'array', self.VISIBILITY: self.SHARED_VISIBILITY,
                                                 'namespace': self.NS_OPTIM}

        # -- output update : aggregation of eq constraints
        dynamic_outputs[self.EQ_CONSTRAINT] = {'type': 'array', self.VISIBILITY: self.SHARED_VISIBILITY,
                                               'namespace': self.NS_OPTIM}

        # -- output update : scalarization of objective
        dynamic_outputs[self.OBJECTIVE] = {'type': 'array', self.VISIBILITY: self.SHARED_VISIBILITY,
                                           'namespace': self.NS_OPTIM}

        # -- output update : lagrangian penalization
        dynamic_outputs[self.OBJECTIVE_LAGR] = {'type': 'array', self.VISIBILITY: self.SHARED_VISIBILITY,
                                                'namespace': self.NS_OPTIM}
        self.iter, self.last_len_database = -1, 0
        self.old_optim_output_df = pd.DataFrame()
        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)
        self.inst_desc_in = dynamic_inputs
        self.inst_desc_out = dynamic_outputs
        # self.DESC_IN.update(proxy.inst_desc_in)
        # self.DESC_OUT.update(proxy.inst_desc_out)

    def run(self):
        '''
        computes the scalarization
        '''

        current_iter = self.get_current_iter()
        # --initialize csv
        if current_iter == 0 and self.get_sosdisc_inputs(self.EXPORT_CSV):
            self.initialize_csv_files()

        aggr_mod_ineq, aggr_mod_eq = self.get_sosdisc_inputs(['aggr_mod_ineq', 'aggr_mod_eq'])
        self.func_manager.set_aggregation_mods(aggr_mod_ineq, aggr_mod_eq)
        # -- update function values
        for f in self.function_dict.keys():
            fvalue_df = self.get_sosdisc_inputs(f)
            self.check_isnan_inf(f, fvalue_df)
            weight = self.function_dict[f][self.WEIGHT]
            if weight != 0:
                self.check_value_range(fvalue_df, fname=f, ftype=self.function_dict[f]['ftype'].upper())
            # conversion dataframe > array:
            f_arr = self.convert_df_to_array(f, fvalue_df)
            # update func manager with value as array
            self.func_manager.update_function_value(f, f_arr)

        # -- build aggregation functions
        self.func_manager.build_aggregated_functions(eps=1e-3)  # alpha=3

        # # -- store output values

        dict_out = self.build_dict_output_values()

        if self.get_sosdisc_inputs(self.EXPORT_CSV):
            self.write_in_csv_files(current_iter, dict_out)

        # To store all results of the optim in a dataframe
        self.build_new_output_optim_df(current_iter, dict_out)

        dict_out[self.OPTIM_OUTPUT_DF] = self.old_optim_output_df

        self.store_sos_outputs_values(dict_out)

    def build_dict_output_values(self):
        '''
        Build the dictionary of outputs for the discipline

        '''
        dict_out = {}
        for f in self.function_dict.keys():
            val = self.func_manager.mod_functions[f][self.VALUE]
            out_name = self.__build_mod_names(f)

            dict_out[out_name] = np.array(
                [val])
        dict_out[self.INEQ_CONSTRAINT] = np.array(
            [self.func_manager.aggregated_functions[self.INEQ_CONSTRAINT]])
        dict_out[self.EQ_CONSTRAINT] = np.array(
            [self.func_manager.aggregated_functions[self.EQ_CONSTRAINT]])
        dict_out[self.OBJECTIVE] = np.array(
            [self.func_manager.aggregated_functions[self.OBJECTIVE]])
        dict_out[self.OBJECTIVE_LAGR] = np.array([self.func_manager.mod_obj])

        return dict_out

    def build_new_output_optim_df(self, current_iter, dict_out):
        '''

        Build new_output_optim_df if the current iteration is new
        Only first run of the optim iteration will be triggered

        '''
        if current_iter == self.old_iter + 1:
            full_end_df = pd.DataFrame({key: [value]
                                        for key, value in dict_out.items()})
            full_end_df.insert(loc=0, column='iteration',
                               value=[current_iter])
            self.old_iter = current_iter
            new_optim_output_df = pd.concat([
                self.old_optim_output_df, full_end_df])
            self.old_optim_output_df = new_optim_output_df

    def initialize_csv_files(self):
        '''

        With export csv option some CSV files are initialized to follow the optimisation wrt func manager outputs

        '''
        s_name = [full_name for full_name in self.local_data if 'export_csv' in full_name][0].split('.')[0]
        t = time.localtime()

        self.csvfile = open(
            f'mod_{s_name}_funcmanager_test.csv', 'w')
        self.csvfile2 = open(
            f'aggr_{s_name}_funcmanager_test.csv', 'w')
        self.writer = csv.writer(
            self.csvfile, lineterminator='\n', delimiter=',')
        self.writer2 = csv.writer(
            self.csvfile2, lineterminator='\n', delimiter=',')

    def write_in_csv_files(self, current_iter, dict_out):
        '''

        With export csv option CSV files are written and flushed to follow the optimisation wrt func manager outputs

        '''
        msg2_list = [self.INEQ_CONSTRAINT, self.EQ_CONSTRAINT, self.OBJECTIVE, self.OBJECTIVE_LAGR]

        msg1 = [str(current_iter)]
        msg1.extend([str(val) for key, val in dict_out.items() if key not in msg2_list])
        header = ["iteration "]
        header.extend([key for key in dict_out if key not in msg2_list])
        header2 = ["iteration "]
        header2.extend([key for key in dict_out if key in msg2_list])
        msg2 = [str(current_iter)]
        msg2.extend([str(val) for key, val in dict_out.items() if key in msg2_list])

        if current_iter == 1:
            self.writer.writerow(header)
            self.writer2.writerow(header2)
        self.writer.writerow(msg1)
        self.writer2.writerow(msg2)
        self.csvfile2.flush()
        self.csvfile.flush()

    def compute_sos_jacobian(self):

        # dobjective/dfunction_df
        f_manager = self.func_manager

        # -- update function values
        for f in self.function_dict.keys():
            fvalue_df = self.get_sosdisc_inputs(f)
            self.check_isnan_inf(f, fvalue_df)
            # conversion dataframe > array:
            f_arr = self.convert_df_to_array(f, fvalue_df)
            # update func manager with value as array
            f_manager.update_function_value(f, f_arr)

        # -- build aggregation functions
        f_manager.build_aggregated_functions(eps=1e-3)  # alpha=3

        var_list = []
        for variable_name in var_list:
            f_manager.compute_dobjective_dweight(variable_name)

        inputs_dict = self.get_sosdisc_inputs()
        grad_value_l = {}

        grad_val_compos = {}
        value_gh_l = []
        value_ghk_l = []
        for variable_name in self.func_manager.functions.keys():
            input_var_value = inputs_dict[variable_name]
            grad_value_l[variable_name] = {}
            grad_val_compos[variable_name] = {}
            var_f_type = self.func_manager.functions[variable_name][self.FTYPE]
            var_aggr_type = self.func_manager.functions[variable_name][self.AGGR_TYPE]
            weight = self.func_manager.functions[variable_name][self.WEIGHT]
            if var_f_type == self.OBJECTIVE:

                if isinstance(input_var_value, np.ndarray):
                    n = len(
                        self.func_manager.functions[variable_name][self.VALUE])

                    if self.func_manager.functions[variable_name][self.AGGR_TYPE] == self.AGGR_TYPE_SUM:

                        grad_value = weight * np.ones(n)
                    elif self.func_manager.functions[variable_name][self.AGGR_TYPE] == self.AGGR_TYPE_SMAX:
                        grad_value = float(weight) * \
                                     np.array(get_dsmooth_dvariable(
                                         self.func_manager.functions[variable_name][self.VALUE]))
                    else:
                        raise Exception(f"Unknown aggr type {self.func_manager.functions[variable_name][self.AGGR_TYPE]}")

                    self.set_partial_derivative(
                        'objective_lagrangian', variable_name, 100.0 * np.atleast_2d(grad_value))
                    self.set_partial_derivative(
                        'objective', variable_name, np.atleast_2d(grad_value))
                else:
                    for col_name in input_var_value.columns:
                        if col_name != 'years':
                            n = len(
                                self.func_manager.functions[variable_name][self.VALUE])

                            if self.func_manager.functions[variable_name][self.AGGR_TYPE] == self.AGGR_TYPE_SUM:

                                grad_value = weight * np.ones(n)
                            elif self.func_manager.functions[variable_name][self.AGGR_TYPE] == self.AGGR_TYPE_SMAX:

                                grad_value = float(weight) * \
                                             np.array(get_dsmooth_dvariable(
                                                 self.func_manager.functions[variable_name][self.VALUE]))

                            self.set_partial_derivative_for_other_types(
                                ('objective',), (variable_name, col_name), grad_value)

                            self.set_partial_derivative_for_other_types(
                                ('objective_lagrangian',), (variable_name, col_name), 100.0 * grad_value)

            elif var_f_type == self.INEQ_CONSTRAINT:
                if isinstance(input_var_value, pd.DataFrame):
                    if var_aggr_type == FunctionManager.INEQ_NEGATIVE_WHEN_SATIFIED:
                        for col_name in input_var_value.columns:
                            if col_name != 'years':
                                value = input_var_value[col_name].values * weight
                                func_value = keep_positive_only(value)
                                grad_value = derivative_d_keep_positive_only(value)
                                value_gh_l.append(func_value)
                                grad_value_l[variable_name][col_name] = grad_value
                    elif var_aggr_type == FunctionManager.INEQ_POSITIVE_WHEN_SATIFIED:
                        for col_name in input_var_value.columns:
                            if col_name != 'years':
                                value = input_var_value[col_name] * weight
                                func_value = keep_negative_only(value)
                                grad_value = derivative_d_keep_negative_only(value)
                                value_gh_l.append(func_value)
                                grad_value_l[variable_name][col_name] = grad_value
                    elif var_aggr_type == FunctionManager.INEQ_NEGATIVE_WHEN_SATIFIED_AND_SQUARE_IT:
                        for col_name in input_var_value.columns:
                            if col_name != 'years':
                                value = input_var_value[col_name] * weight
                                func_value = keep_positive_only_square(value)
                                grad_value = derivative_d_keep_positive_only_square(value)
                                value_gh_l.append(func_value)
                                grad_value_l[variable_name][col_name] = grad_value
                    elif var_aggr_type == FunctionManager.INEQ_POSITIVE_WHEN_SATIFIED_AND_SQUARE_IT:
                        for col_name in input_var_value.columns:
                            if col_name != 'years':
                                value = input_var_value[col_name] * weight
                                func_value = keep_negative_only(value) ** 2
                                grad_value = derivative_d_keep_negative_only(value) * 2 * value
                                value_gh_l.append(func_value)
                                grad_value_l[variable_name][col_name] = grad_value
                    else:
                        for col_name in input_var_value.columns:
                            if col_name != 'years':
                                # h: cst_func_ineq, g: smooth_max, f: smooth_max
                                # g(h(x)) for each variable , f([g(h(x1), g(h(x2))])
                                # weight
                                value = input_var_value[col_name] * weight
                                # h'(x)
                                grad_value = self.get_dfunc_ineq_dvariable(value)
                                # h(x)
                                func_ineq = f_manager.cst_func_ineq(value)
                                # g(h(x))
                                value_gh_l.append(
                                    smooth_maximum(func_ineq))
                                # g'(h(x))
                                grad_val_compos[variable_name][col_name] = get_dsmooth_dvariable(
                                    func_ineq)
                                # g'(h(x)) * h'(x)
                                grad_val_compos_l = np.array(
                                    grad_value) * np.array(grad_val_compos[variable_name][col_name])

                                grad_value_l[variable_name][col_name] = grad_val_compos_l

                elif isinstance(input_var_value, np.ndarray):
                    value = input_var_value * weight
                    # h'(x)
                    grad_value = self.get_dfunc_ineq_dvariable(
                        value)
                    # h(x)
                    func_ineq = f_manager.cst_func_ineq(
                        value)
                    # g(h(x))
                    value_gh_l.append(smooth_maximum(func_ineq))
                    # g'(h(x))
                    grad_val_compos[variable_name] = get_dsmooth_dvariable(
                        func_ineq)
                    # g'(h(x)) * h'(x)
                    grad_val_compos_l = np.array(
                        grad_value) * np.array(grad_val_compos[variable_name])

                    grad_value_l[variable_name] = grad_val_compos_l

                else:
                    raise Exception(
                        'Gradients for constraints which are not dataframes or arrays are not yet implemented')
            elif var_f_type == self.EQ_CONSTRAINT:
                if isinstance(input_var_value, pd.DataFrame):
                    for col_name in input_var_value.columns:
                        if col_name != 'years':
                            # k: sqrt(x**2) h: cst_func_eq, g : smooth_max, f: smooth_max
                            # g(h(k(x))) for each variable , f([g(h(k(x1)), g(h(k(x2)))])
                            # weight
                            value = np.array(input_var_value[col_name].values * weight)
                            if self.func_manager.functions[variable_name][
                                self.AGGR_TYPE] == self.func_manager.AGGR_TYPE_DELTA:
                                # k(x)
                                k_cst = np.sqrt(compute_func_with_exp_min(value ** 2, 1e-15))
                                # k'(x)
                                dk_dcst = 2 * np.ones(len(value)) * value * \
                                          compute_dfunc_with_exp_min(value ** 2, 1e-15).reshape(value.shape) / (
                                                  2 * k_cst)
                                # h(k(x))
                                h_k = f_manager.cst_func_ineq(
                                    k_cst)
                                # h'(k(x))
                                dh_dcst = self.get_dfunc_ineq_dvariable(
                                    k_cst)

                            elif self.func_manager.functions[variable_name][
                                self.AGGR_TYPE] == self.func_manager.AGGR_TYPE_LIN_TO_QUAD:
                                k_cst = f_manager.cst_func_eq_lintoquad(
                                    value)
                                dk_dcst = self.get_dfunc_eq_lin_to_quad_dvariable(value)

                                h_k = k_cst
                                dh_dcst = 1.
                            else:
                                k_cst = np.sqrt(np.sign(value) * value)
                                # k'(x)
                                dk_dcst = np.sign(value) * np.ones(len(value)) / (
                                        2 * k_cst)
                                # h(k(x))
                                h_k = f_manager.cst_func_ineq(
                                    k_cst)
                                # h'(k(x))
                                dh_dcst = self.get_dfunc_ineq_dvariable(
                                    k_cst)

                            # g(h(k(x)))
                            value_ghk_l.append(
                                smooth_maximum(h_k))

                            # g'(h(k(x)))
                            grad_val_compos[variable_name][col_name] = get_dsmooth_dvariable(
                                h_k)
                            # g'(h(k(x))) * h'(k(x)) * k'(x)
                            grad_val_compos_l = np.array(grad_val_compos[variable_name][col_name]) * np.array(
                                dh_dcst) * np.array(dk_dcst)

                            grad_value_l[variable_name][col_name] = grad_val_compos_l

                elif isinstance(input_var_value, np.ndarray):
                    # k: sqrt(x**2) h: cst_func_eq, g : smooth_max, f: smooth_max
                    # g(h(k(x))) for each variable , f([g(h(k(x1)), g(h(k(x2)))])
                    # weight
                    value = input_var_value * weight

                    if self.func_manager.functions[variable_name][
                        self.AGGR_TYPE] == self.func_manager.AGGR_TYPE_DELTA:
                        # k(x)
                        k_cst = np.sqrt(compute_func_with_exp_min(value ** 2, 1e-15))
                        # k'(x)
                        dk_dcst = 2 * np.ones(len(value)) * value * \
                                  compute_dfunc_with_exp_min(value ** 2, 1e-15).reshape(value.shape) / (
                                          2 * k_cst)
                        # h(k(x))
                        h_k = f_manager.cst_func_ineq(
                            k_cst)
                        # h'(k(x))
                        dh_dcst = self.get_dfunc_ineq_dvariable(
                            k_cst)

                    elif self.func_manager.functions[variable_name][
                        self.AGGR_TYPE] == self.func_manager.AGGR_TYPE_LIN_TO_QUAD:
                        k_cst = f_manager.cst_func_eq_lintoquad(
                            value)
                        dk_dcst = self.get_dfunc_eq_lin_to_quad_dvariable(value)

                        h_k = k_cst
                        dh_dcst = 1.
                    else:
                        k_cst = np.sqrt(np.sign(value) * value)
                        # k'(x)
                        dk_dcst = np.sign(value) * np.ones(len(value)) / (
                                2 * k_cst)
                        # h(k(x))
                        h_k = f_manager.cst_func_ineq(
                            k_cst)
                        # h'(k(x))
                        dh_dcst = self.get_dfunc_ineq_dvariable(
                            k_cst)

                    # g(h(k(x)))
                    value_ghk_l.append(
                        smooth_maximum(h_k))

                    # g'(h(k(x)))
                    grad_val_compos[variable_name] = get_dsmooth_dvariable(
                        h_k)
                    # g'(h(k(x))) * h'(k(x)) * k'(x)
                    grad_val_compos_l = np.array(grad_val_compos[variable_name]) * np.array(
                        dh_dcst) * np.array(dk_dcst)

                    grad_value_l[variable_name] = grad_val_compos_l
                else:
                    raise Exception(
                        'Gradients for constraints which are not dataframes or arrays are not yet implemented')

        dict_grad_eq = {}
        grad_val_ineq = {}
        grad_val_eq = {}
        # g'(h(x)) * h'(x)
        if self.func_manager.aggr_mod_ineq == 'smooth_max' and len(value_gh_l) != 0:
            grad_val_ineq = get_dsmooth_dvariable(value_gh_l)
        if self.func_manager.aggr_mod_ineq == 'smooth_max' and len(value_ghk_l) != 0:
            grad_val_eq = get_dsmooth_dvariable(value_ghk_l)
        i = 0
        j = 0
        for variable_name in self.func_manager.functions.keys():
            weight = self.func_manager.functions[variable_name][self.WEIGHT]
            var_f_type = self.func_manager.functions[variable_name][self.FTYPE]
            if var_f_type == self.INEQ_CONSTRAINT:

                input_var_value = inputs_dict[variable_name]
                if isinstance(input_var_value, np.ndarray):

                    if self.func_manager.aggr_mod_ineq == 'smooth_max':
                        grad_lagr_val = np.array(grad_value_l[variable_name]) * grad_val_ineq[i]
                        grad_ineq_val = np.array(grad_value_l[variable_name]) * grad_val_ineq[i]
                    else:
                        grad_lagr_val = np.array(grad_value_l[variable_name])
                        grad_ineq_val = np.array(grad_value_l[variable_name])

                    self.set_partial_derivative(
                        'objective_lagrangian', variable_name,
                        np.atleast_2d(weight * 100.0 * grad_lagr_val))
                    self.set_partial_derivative(
                        'ineq_constraint', variable_name,
                        np.atleast_2d(weight * grad_ineq_val))

                    i = i + 1

                elif isinstance(input_var_value, pd.DataFrame):
                    for col_name in input_var_value.columns:
                        if col_name != 'years':
                            if self.func_manager.aggr_mod_ineq == 'smooth_max':
                                grad_lagr_val = np.array(grad_value_l[variable_name][col_name]) * grad_val_ineq[i]
                                grad_ineq_val = np.array(grad_value_l[variable_name][col_name]) * grad_val_ineq[i]
                            else:
                                grad_lagr_val = np.array(grad_value_l[variable_name][col_name])
                                grad_ineq_val = np.array(grad_value_l[variable_name][col_name])

                            self.set_partial_derivative_for_other_types(
                                ('objective_lagrangian',),
                                (variable_name, col_name),
                                weight * 100.0 * grad_lagr_val)
                            self.set_partial_derivative_for_other_types(
                                ('ineq_constraint',), (variable_name, col_name),
                                weight * grad_ineq_val)
                            i = i + 1

            elif var_f_type == self.EQ_CONSTRAINT:
                input_var_value = inputs_dict[variable_name]
                if isinstance(input_var_value, np.ndarray):
                    if self.func_manager.aggr_mod_eq == 'smooth_max':
                        grad_lagr_val = np.array(grad_value_l[variable_name]) * grad_val_eq[j]
                        grad_eq_val = np.array(grad_value_l[variable_name]) * grad_val_eq[j]
                    else:

                        grad_lagr_val = np.array(grad_value_l[variable_name])
                        grad_eq_val = np.array(grad_value_l[variable_name])

                    self.set_partial_derivative(
                        'objective_lagrangian', variable_name,
                        np.atleast_2d(weight * 100.0 * grad_lagr_val))
                    self.set_partial_derivative(
                        'eq_constraint', variable_name,
                        np.atleast_2d(weight * grad_eq_val))
                    j = j + 1

                elif isinstance(input_var_value, pd.DataFrame):

                    for col_name in input_var_value.columns:
                        if col_name != 'years':
                            if self.func_manager.aggr_mod_eq == 'smooth_max':
                                grad_lagr_val = np.array(grad_value_l[variable_name][col_name]) * grad_val_eq[j]
                                grad_eq_val = np.array(grad_value_l[variable_name][col_name]) * grad_val_eq[j]
                            else:
                                grad_lagr_val = np.array(grad_value_l[variable_name][col_name])
                                grad_eq_val = np.array(grad_value_l[variable_name][col_name])

                            self.set_partial_derivative_for_other_types(
                                ('objective_lagrangian',), (variable_name, col_name),
                                weight * 100.0 * grad_lagr_val)
                            self.set_partial_derivative_for_other_types(
                                ('eq_constraint',), (variable_name, col_name),
                                weight * grad_eq_val)
                            j = j + 1

    def get_dfunc_ineq_dvariable(self, value_df, eps=1e-3):
        """
        Get dineq_dvariable
        """
        grad_value = []
        # for col in value_df.columns:
        # if col != 'years':
        valcol = value_df

        cst_result = np.zeros_like(valcol)
        # ---get value of espilon2
        smooth_log = self.get_sosdisc_inputs('smooth_log')
        eps2 = self.get_sosdisc_inputs('eps2')
        for iii, val in enumerate(valcol):
            if smooth_log and val > eps2:
                # res00 + 2 * np.log(val)
                res = 2.0 / val
            elif val > eps:
                # res = res0 + val ** 2 - eps ** 2
                res = 2.0 * val
            elif val < -250:
                # res = 0.0
                res = 0.0
            else:
                # res = eps * (np.exp(val) - 1.)
                res = eps * np.exp(val)
            cst_result[iii] = res
        grad_value.extend(cst_result)

        return grad_value

    def get_dfunc_eq_lin_to_quad_dvariable(self, value_df, eps=1e-3):
        """
        Get dineq_dvariable for lin_to_quad case
        """
        grad_value = []
        # for col in value_df.columns:
        # if col != 'years':
        valcol = value_df

        dcst_result = np.zeros_like(valcol)
        # ---get value of espilon2
        smooth_log = self.get_sosdisc_inputs('smooth_log')
        eps2 = self.get_sosdisc_inputs('eps2')
        for iii, val in enumerate(valcol):
            if val > eps:
                # res = res0 + val ** 2 - eps ** 2
                dres = 2.0 * val
            elif -eps < val < 0.0:
                # res = eps * (np.exp(-val) - 1.)
                dres = -eps * np.exp(-val)
            elif val < -eps:
                # res= res0 + (-val) - eps
                dres = -1.
            else:
                # res = eps * (np.exp(val) - 1.)
                dres = eps * np.exp(val)
            dcst_result[iii] = dres
        grad_value.extend(dcst_result)
        return grad_value

    def check_isnan_inf(self, key_, value_):
        """
        Check if there is any NaN or inf values in dataframe
        """

        if isinstance(value_, pd.DataFrame):
            try:
                if value_.isin([np.inf, -np.inf]).any().any():
                    logging.warning(f'{key_} has inf values')
                if value_.isin([np.nan]).any().any():
                    logging.warning(f'{key_} has NaN values')
            except AttributeError as error:
                logging.warning(
                    f'func_managerdiscipline::check_isnan_inf : {str(error)}')

    def convert_df_to_array(self, func_name, val_df):
        '''
        Non-generic method that returns an array of the only (excluding years) column in val_df
        '''

        if isinstance(val_df, ndarray):
            arr = val_df
        else:
            if 'years' in val_df.columns:
                val_df = val_df.drop('years', axis=1)
            if self.INDEX in self.function_dict[func_name]:
                val_df = val_df.loc[self.function_dict[func_name][self.INDEX]]
            if self.COMPONENT in self.function_dict[func_name]:
                val_df = val_df[self.function_dict[func_name][self.COMPONENT]]

            if isinstance(val_df, (pd.Series, pd.DataFrame)):
                arr = val_df.to_numpy().flatten()
            else:
                arr = asarray([val_df])

        if arr.dtype == object:
            arr = arr.astype(float64)
        return arr

    def get_chart_filter_list(self):

        chart_filters = []
        chart_list = ['lagrangian objective', 'aggregated objectives',
                      'objectives', 'ineq_constraints', 'eq_constraints', 'objective (colored)']
        optim_output_df = self.get_sosdisc_outputs(self.OPTIM_OUTPUT_DF)
        if optim_output_df[self.INEQ_CONSTRAINT].empty:
            chart_list.remove('ineq_constraints')
        if optim_output_df[self.EQ_CONSTRAINT].empty:
            chart_list.remove('eq_constraints')
        if optim_output_df[self.INEQ_CONSTRAINT].empty and optim_output_df[self.EQ_CONSTRAINT].empty:
            chart_list.remove('objective (colored)')
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))
        return chart_filters

    def get_post_processing_list(self, filters=None):

        # For the outputs, making a graph for block fuel vs range and blocktime vs
        # range

        instanciated_charts = []
        charts = []

        func_df = self.get_sosdisc_inputs('function_df')

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values
        if 'objective (colored)' in charts:
            if not self.get_sosdisc_outputs(self.OPTIM_OUTPUT_DF)[self.OBJECTIVE].empty and not \
                    self.get_sosdisc_outputs(self.OPTIM_OUTPUT_DF)[self.INEQ_CONSTRAINT].empty:
                optim_output_df = deepcopy(self.get_sosdisc_outputs(self.OPTIM_OUTPUT_DF))
                new_chart = self.get_chart_obj_constraints_iterations(func_df, optim_output_df, [self.OBJECTIVE],
                                                                      'objective (colored)')
                instanciated_charts.append(new_chart)

        chart_list = ['lagrangian objective', 'aggregated objectives',
                      'objectives', 'ineq_constraints', 'eq_constraints', ]

        for chart in chart_list:
            new_chart = None
            optim_output_df = self.get_sosdisc_outputs(self.OPTIM_OUTPUT_DF)
            parameters_df, obj_list, ineq_list, eq_list = self.get_parameters_df(
                func_df)
            if chart in charts:
                if chart == 'lagrangian objective':
                    new_chart = self.get_chart_lagrangian_objective_iterations(
                        optim_output=optim_output_df,
                        main_parameters=parameters_df.loc[[
                            self.OBJECTIVE_LAGR]],
                        sub_parameters=parameters_df.loc[[
                            self.OBJECTIVE, self.INEQ_CONSTRAINT, self.EQ_CONSTRAINT]],
                        name=chart)
                elif chart == 'aggregated objectives':
                    new_chart = self.get_chart_aggregated_iterations(
                        optim_output=optim_output_df,
                        main_parameters=parameters_df.loc[[
                            self.OBJECTIVE, self.INEQ_CONSTRAINT, self.EQ_CONSTRAINT]],
                        objectives=parameters_df.loc[obj_list],
                        ineq_constraints=parameters_df.loc[ineq_list],
                        eq_constraints=parameters_df.loc[eq_list], name=chart)
                elif chart == 'objectives':
                    new_chart = self.get_chart_parameters_mod_iterations(optim_output_df,
                                                                         parameters_df.loc[obj_list],
                                                                         name=chart)
                elif chart == 'ineq_constraints':
                    is_ineq_constraints = 'ineq_constraint' in func_df.loc[
                        ~np.isclose(func_df['weight'], 0.0), 'ftype'].values.tolist()
                    if is_ineq_constraints:
                        new_chart = self.get_chart_parameters_mod_iterations(optim_output_df,
                                                                             parameters_df.loc[ineq_list],
                                                                             name=chart)
                elif chart == 'eq_constraints':
                    is_eq_constraints = 'eq_constraint' in func_df.loc[
                        ~np.isclose(func_df['weight'], 0.0), 'ftype'].values.tolist()
                    if is_eq_constraints:
                        new_chart = self.get_chart_parameters_mod_iterations(optim_output_df,
                                                                             parameters_df.loc[eq_list],
                                                                             name=chart)
                if new_chart is not None:
                    instanciated_charts.append(new_chart)
        return instanciated_charts

    def get_parameters_df(self, func_df):
        """
        Function to explore func_df and arrange the parameters info to plot into a dataframe
        Get ['variable', 'parent', 'weight', 'aggr'] for aggregated objectives and constraints as well as for _mod
        Inputs : func_df
        Outputs : parameters_df, mod_lists
        """
        parameters_dict = {}
        # Handle _mod
        obj_list = []
        ineq_list = []
        eq_list = []
        mod_columns_dict = {self.PARENT: None,
                            self.WEIGHT: 1.0, self.AGGR_TYPE: self.AGGR_TYPE_SUM}
        for i, row in func_df.iterrows():
            mod_parameter_dict = {}
            mod_parameter_dict[self.VARIABLE] = row[self.VARIABLE] + '_mod'
            if self.PARENT in row.index:
                if row[self.PARENT] in [np.nan]:
                    mod_parameter_dict[self.PARENT] = None
                else:
                    mod_parameter_dict[self.PARENT] = row[self.PARENT]
            else:
                mod_parameter_dict[self.PARENT] = None
            for column in mod_columns_dict.keys():
                if column in row.index:
                    mod_parameter_dict[column] = row[column]
                else:
                    mod_parameter_dict[column] = mod_columns_dict[column]
            if 'objective' in row['ftype']:
                obj_list.append(mod_parameter_dict[self.VARIABLE])
            elif 'ineq_constraint' in row['ftype']:
                ineq_list.append(mod_parameter_dict[self.VARIABLE])
            elif 'eq_constraint' in row['ftype']:
                eq_list.append(mod_parameter_dict[self.VARIABLE])
            parameters_dict[mod_parameter_dict[self.VARIABLE]
            ] = mod_parameter_dict

        # Aggregated objectives and constraints
        dict_aggr_obj = {self.VARIABLE: self.OBJECTIVE,
                         self.PARENT: self.OBJECTIVE_LAGR,
                         self.WEIGHT: 1.0, self.AGGR_TYPE: self.AGGR_TYPE_SUM}
        parameters_dict[dict_aggr_obj[self.VARIABLE]] = dict_aggr_obj
        dict_aggr_ineq = {self.VARIABLE: self.INEQ_CONSTRAINT,
                          self.PARENT: self.OBJECTIVE_LAGR,
                          self.WEIGHT: 1.0, self.AGGR_TYPE: self.AGGR_TYPE_SMAX}
        parameters_dict[dict_aggr_ineq[self.VARIABLE]] = dict_aggr_ineq
        dict_aggr_eq = {self.VARIABLE: self.EQ_CONSTRAINT,
                        self.PARENT: self.OBJECTIVE_LAGR,
                        self.WEIGHT: 1.0, self.AGGR_TYPE: self.AGGR_TYPE_SMAX}
        parameters_dict[dict_aggr_eq[self.VARIABLE]] = dict_aggr_eq

        # Lagrangian objective
        dict_lagrangian = {self.VARIABLE: self.OBJECTIVE_LAGR, self.PARENT: None,
                           self.WEIGHT: 1.0, self.AGGR_TYPE: self.AGGR_TYPE_SUM}
        parameters_dict[dict_lagrangian[self.VARIABLE]] = dict_lagrangian
        parameters_df = pd.DataFrame(parameters_dict).transpose()

        return parameters_df, obj_list, ineq_list, eq_list

    def get_chart_lagrangian_objective_iterations(self, optim_output, main_parameters, sub_parameters=[],
                                                  name='lagrangian objective'):
        """
        Function to create the post proc of aggregated objectives and constraints
        A dropdown menu is used to select between:
            -"Simple" : Simple scatter+line of lagrangian objective
            -Detailed Contribution" : scatter+line of aggregated (sum*100) lagrangian objective + summed area of individual contributions (*100)
        Inputs: main_parameters (lagrangian objective) name, list of sub-parameters (aggregated objectives, inequality constraints and
        equality constraints) names and name of the plot
        Ouput: instantiated plotly chart
        """
        chart_name = f'{name} wrt iterations'
        fig = go.Figure()
        for parameter in main_parameters[self.VARIABLE]:
            y = [value[0] for value in optim_output[parameter].values]
            if 'complex' in str(type(y[0])):
                y = [np.real(value[0])
                     for value in optim_output[parameter].values]
            fig.add_trace(go.Scatter(x=list(optim_output['iteration'].values),
                                     y=list(y), name=parameter, visible=True))
        for parameter in sub_parameters[self.VARIABLE]:
            y = [value[0] * 100 for value in optim_output[parameter].values]
            if 'complex' in str(type(y[0])):
                y = [np.real(value[0])
                     for value in optim_output[parameter].values]
            if sum(y) == 0:
                continue
            fig.add_trace(go.Scatter(x=list(optim_output['iteration'].values),
                                     y=list(y), stackgroup='group', mode='none', name=parameter + ' (x100)',
                                     visible=False))
        fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                          xaxis_title='n iterations', yaxis_title=f'value of {name}')

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{'visible': [False if scatter['stackgroup']
                                                        == 'group' else True for scatter in fig.data]}, ],
                            label="Simple",
                            method="restyle"
                        ),
                        dict(
                            args=[{'visible': [True for _ in fig.data]}, ],
                            label="Detailed Contribution",
                            method="restyle"
                        )
                    ]),
                    direction='down',
                    type='dropdown',
                    pad={"r": 0, "t": 0},
                    showactive=True,
                    active=0,
                    x=1.0,
                    y=1.01,
                    yanchor='bottom',
                    xanchor='right'
                ),
                dict(
                    buttons=list([
                        dict(
                            args=[{"yaxis.type": "linear"}],
                            label="Linear",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis.type": "log"}],
                            label="Log",
                            method="relayout"
                        ),
                    ]),
                    type="buttons",
                    direction="right",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=0,
                    x=0.0,
                    xanchor="left",
                    y=1.01,
                    yanchor="bottom"
                ),
            ]
        )

        new_chart = InstantiatedPlotlyNativeChart(
            fig, chart_name=chart_name, default_title=True)
        return new_chart

    def get_chart_aggregated_iterations(self, optim_output, main_parameters, objectives={}, ineq_constraints={},
                                        eq_constraints={}, name='aggregated'):
        """
        Function to create the post proc of aggregated objectives and constraints
        A dropdown menu is used to select between:
            -"All Aggregated" : Simple scatter+line of all the aggregated values
            -"Objective - Detailed" : scatter+line of aggregated (sum) objective + summed area of individual contributions
            -"Ineq Constraint - Detailed" : scatter+line of aggregated (smax) inequality constraint + area of individual contributions
            -"Eq Constraint - Detailed": scatter+line of aggregated (smax) equality constraint + area of individual contributions
        Inputs: main_parameters (aggregated) names, list of objectives, inequality constraints and equality constraints names,
        and name of the plot
        Ouput: instantiated plotly chart
        """
        chart_name = f'{name} wrt iterations'
        fig = go.Figure()
        for parameter in main_parameters[self.VARIABLE]:
            if 'objective' in parameter:
                customdata = ['aggr', 'obj']
            elif 'ineq_constraint' in parameter:
                customdata = ['aggr', 'ineq']
            elif 'eq_constraint' in parameter:
                customdata = ['aggr', 'eq']
            else:
                raise Exception(f"Can't get customdata for parameter {parameter}")
            y = [value[0] for value in optim_output[parameter].values]
            if 'complex' in str(type(y[0])):
                y = [np.real(value[0])
                     for value in optim_output[parameter].values]
            fig.add_trace(go.Scatter(x=list(optim_output['iteration'].values),
                                     y=list(y), name=parameter, customdata=customdata, visible=True))

        if len(objectives) > 0:
            for parameter in objectives[self.VARIABLE]:
                customdata = ['mod', 'obj']
                y = [value[0] for value in optim_output[parameter].values]
                if 'complex' in str(type(y[0])):
                    y = [np.real(value[0])
                         for value in optim_output[parameter].values]
                if sum(y) == 0:
                    continue
                fig.add_trace(go.Scatter(x=list(optim_output['iteration'].values),
                                         y=list(y), stackgroup='group_obj', mode='none',
                                         name=parameter, customdata=customdata, visible=False))

        if len(ineq_constraints) > 0:
            for i, parameter in enumerate(ineq_constraints[self.VARIABLE]):
                customdata = ['mod', 'ineq']
                y = [value[0] for value in optim_output[parameter].values]
                if 'complex' in str(type(y[0])):
                    y = [np.real(value[0])
                         for value in optim_output[parameter].values]
                if sum(y) == 0:
                    continue
                fig.add_trace(go.Scatter(x=list(optim_output['iteration'].values),
                                         y=list(y), stackgroup='group_ineq' + str(i), mode='none',
                                         name=parameter, customdata=customdata, visible=False))

        if len(eq_constraints) > 0:
            for i, parameter in enumerate(eq_constraints[self.VARIABLE]):
                customdata = ['mod', 'eq']
                y = [value[0] for value in optim_output[parameter].values]
                if 'complex' in str(type(y[0])):
                    y = [np.real(value[0])
                         for value in optim_output[parameter].values]
                if sum(y) == 0:
                    continue
                fig.add_trace(go.Scatter(x=list(optim_output['iteration'].values),
                                         y=list(y), stackgroup='group_eq' + str(i), mode='none',
                                         name=parameter, customdata=customdata, visible=False))
        fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                          xaxis_title='n iterations', yaxis_title=f'value of {name}')

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{'visible': [True if scatter['customdata'][0]
                                                       == 'aggr' else False for scatter in fig.data]}, ],
                            label="All Aggregated",
                            method="restyle"
                        ),
                        dict(
                            args=[{'visible': [True if scatter['customdata'][1]
                                                       == 'obj' else False for scatter in fig.data]}, ],
                            label="Objective - Detailed",
                            method="restyle"
                        ),
                        dict(
                            args=[{'visible': [True if scatter['customdata'][1]
                                                       == 'ineq' else False for scatter in fig.data]}, ],
                            label="Ineq Constraint - Detailed",
                            method="restyle"
                        ),
                        dict(
                            args=[{'visible': [True if scatter['customdata'][1]
                                                       == 'eq' else False for scatter in fig.data]}, ],
                            label="Eq Constraint - Detailed",
                            method="restyle"
                        ),
                    ]),
                    direction='down',
                    type='dropdown',
                    pad={"r": 0, "t": 0},
                    showactive=True,
                    active=0,
                    x=1.0,
                    y=1.01,
                    yanchor='bottom',
                    xanchor='right'
                ),
                dict(
                    buttons=list([
                        dict(
                            args=[{"yaxis.type": "linear"}],
                            label="Linear",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis.type": "log"}],
                            label="Log",
                            method="relayout"
                        ),
                    ]),
                    type="buttons",
                    direction="right",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=0,
                    x=0.0,
                    xanchor="left",
                    y=1.01,
                    yanchor="bottom"
                ),
            ]
        )
        new_chart = InstantiatedPlotlyNativeChart(
            fig, chart_name=chart_name, default_title=True)
        return new_chart

    def get_chart_parameters_mod_iterations(self, optim_output, parameters_df, name):
        """
        Function to create the post proc of objectives and constraints mod.
        First all the values are put in a dataframe, then the parent-children links are 'calculated'.
        Then it traces the grouped _mod and adds value, weight, aggregation type and children as hovertext.
        Finally a dropdown menu is used to select the level to be displayed.
        Inputs: parameters_dict[variable,parents,children,weights,aggr_type] and name of the plot
        Output: instantiated plotly chart
        """
        chart_name = f'{name} wrt iterations'
        fig = go.Figure()
        # Remove entries with weight = 0.0
        weight0_index = parameters_df[parameters_df[self.WEIGHT]
                                      == '0.0'].index
        parameters_df.drop(weight0_index, inplace=True)
        # Add value column
        parameters_df['value'] = [
            [0.0 for _ in range(optim_output.shape[0])] for _ in range(parameters_df.shape[0])]

        for parameter in parameters_df[self.VARIABLE]:
            y = [value[0] for value in optim_output[parameter].values]
            if 'complex' in str(type(y[0])):
                y = [np.real(value[0])
                     for value in optim_output[parameter].values]
            parameters_df.loc[parameters_df[self.VARIABLE]
                              == parameter, 'value'] = {parameter: y}
        # Remove entries with values = 0.0
        parameters_df['isnull'] = parameters_df['value'].apply(
            lambda x: all(val == 0.0 for val in x))
        value0_index = parameters_df[parameters_df['isnull']].index
        parameters_df.drop(value0_index, inplace=True)
        parameters_df.drop(columns=['isnull'], inplace=True)
        # Create entries for all the groups in dataframe
        # Find all parents by level (to calculate level-by-level)
        n_level = int(max(np.append(0, np.asarray([0 if parent is None else len(parent.split('-'))
                                                   for parent in parameters_df[self.PARENT]]))))
        level_list = dict.fromkeys(range(n_level), [])
        parent_split = np.asarray([[] if parent is None else parent.split('-')
                                   for parent in parameters_df[self.PARENT]], dtype=object)
        for i in reversed(range(n_level)):
            i_lvl = []
            for parent in parent_split:
                if len(parent) - 1 >= i:
                    i_lvl += [parent[i], ]
            level_list[i] = list(set(i_lvl))
        # parents_list = list(set(np.asarray([parent.split(
        # '-') for parent in parameters_df[self.PARENT].to_list()]).flatten()))

        for parent_level in level_list.values():
            for parent in parent_level:
                # if parent isn't in df
                if parent not in parameters_df[self.VARIABLE]:
                    parent_list = []
                    parameters_df['match'] = parameters_df[self.PARENT].apply(
                        lambda x: 'Match' if parent in str(x) else 'Mismatch')
                    for full_parent in parameters_df.loc[parameters_df['match'] == 'Match', self.PARENT]:
                        if full_parent is None:
                            continue
                        for i, split_full_parent in enumerate(full_parent.split('-')):
                            if parent == split_full_parent and i > 0:  # if parent has a parent
                                parent_list.append(
                                    full_parent.split('-')[i - 1])
                    parent_parent = None
                    if len(list(set(parent_list))) > 0:
                        parent_parent = parent_list[0]
                    children_list = parameters_df.loc[parameters_df['match']
                                                      == 'Match', self.VARIABLE]
                    parameters_df.drop(columns=['match'], inplace=True)
                    value = [sum(v) for v in zip(
                        *parameters_df.loc[children_list, 'value'])]
                    dict_parent = {self.VARIABLE: parent,
                                   self.PARENT: parent_parent,
                                   self.WEIGHT: 1.0, self.AGGR_TYPE: self.AGGR_TYPE_SUM,
                                   'value': [value]}
                    parameters_df = pd.concat([parameters_df,
                        pd.DataFrame(dict_parent, index=[parent])], axis=0)

        for row in parameters_df.iterrows():
            vis = False
            y = row[1]['value']
            if 'complex' in str(type(y[0])):
                y = [np.real(value)
                     for value in row[1]['value']]
            hovertemplate = '<br>X: %{x}' + '<br>Y: %{y:.2e}' + \
                            '<br>weight: %{customdata[0]}' + \
                            '<br>aggr_type: %{customdata[1]}'
            is_mod = True if '_mod' in row[1][self.VARIABLE] else False
            vis = True if row[1][self.PARENT] is None else False
            customdata = [[str(row[1][self.WEIGHT]) for _ in range(len(y))],
                          [row[1][self.AGGR_TYPE] for _ in range(len(y))],
                          [row[1][self.PARENT] for _ in range(len(y))],
                          [is_mod for _ in range(len(y))]]
            fig.add_trace(go.Scatter(x=list(optim_output['iteration'].values),
                                     y=list(y), name=row[1][self.VARIABLE], customdata=list(np.asarray(customdata).T),
                                     hovertemplate=hovertemplate,
                                     visible=vis))

        fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                          xaxis_title='n iterations', yaxis_title=f'value of {name}')

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list(
                        [dict(
                            args=[
                                {'visible': [True if scatter['customdata'][0][2]
                                                     == group else False for scatter in fig.data]},
                            ],
                            label='grouped',
                            method="update"
                        ) for group in list(set(parameters_df[self.PARENT])) if group is None] +
                        [dict(
                            args=[
                                {'visible': [scatter['customdata'][0][3]
                                             for scatter in fig.data]},
                            ],
                            label='All',
                            method="update"
                        )] +
                        [dict(
                            args=[
                                {'visible': [True if scatter['customdata'][0][2]
                                                     == group else False for scatter in fig.data]},
                            ],
                            label='group ' + str(group),
                            method="update"
                        ) for group in list(set(parameters_df[self.PARENT])) if group is not None]
                    ),
                    direction='down',
                    type='dropdown',
                    pad={"r": 0, "t": 0},
                    showactive=True,
                    active=0,
                    x=1.0,
                    y=1.01,
                    yanchor='bottom',
                    xanchor='right'
                ),
                dict(
                    buttons=list([
                        dict(
                            args=[{"yaxis.type": "linear"}],
                            label="Linear",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis.type": "log"}],
                            label="Log",
                            method="relayout"
                        ),
                    ]),
                    type="buttons",
                    direction="right",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=0,
                    x=0.0,
                    xanchor="left",
                    y=1.01,
                    yanchor="bottom"
                ),
            ]
        )

        new_chart = InstantiatedPlotlyNativeChart(
            fig, chart_name=chart_name, default_title=True)
        return new_chart

    def get_chart_obj_constraints_iterations(self, func_df, optim_output, objectives, name):
        """
        Function to create a summary post proc of the optim problem
        In black : the aggregated objective
        In colorscale from green to red : the sum of all the constraints (green == negative values)
        Additionnal information such as the name and value of the dominant constraint are shown in the hovertext
        Inputs: objective name, name of the plot and boolean for log scale
        Ouput: instantiated plotly chart
        """

        chart_name = 'objective wrt iterations with constraints (colored)'
        fig = go.Figure()
        x = optim_output['iteration'].values
        for obj in objectives:
            y = [value[0] for value in optim_output[obj].values]
            if 'complex' in str(type(y[0])):
                y = [np.real(value[0])
                     for value in optim_output[obj].values]
            fig.add_trace(go.Scatter(x=list(x), y=list(
                y), name=obj, line=dict(color='black')))
        func_dict = {row[self.VARIABLE] + '_mod': row['ftype']
                     for i, row in func_df.iterrows()}

        for col in optim_output.columns:
            if col not in ['iteration', ]:
                optim_output[col] = optim_output[col].apply(
                    lambda x: x[0].real)

        ineq_constraints = optim_output[[
            key for key, value in func_dict.items() if value in [self.INEQ_CONSTRAINT]]].astype(float).reset_index(
            drop=True)
        eq_constraints = optim_output[[
            key for key, value in func_dict.items() if value in [self.EQ_CONSTRAINT]]].astype(float).reset_index(
            drop=True)

        ineq_constraints_sum = ineq_constraints.sum(axis=1).fillna(0)
        ineq_constraints_max = ineq_constraints.max(axis=1).fillna(0)
        try:
            ineq_constraints_max_col = ineq_constraints.idxmax(axis=1)
        except:
            ineq_constraints_max_col = None

        eq_constraints_sum = eq_constraints.sum(axis=1).fillna(0)
        eq_constraints_max = eq_constraints.max(axis=1).fillna(0)
        try:
            eq_constraints_max_col = eq_constraints.idxmax(axis=1)
        except:
            eq_constraints_max_col = None
        tot_constraints = pd.concat([ineq_constraints, eq_constraints], axis=1)
        tot_constraints_sum = ineq_constraints_sum + eq_constraints_sum
        tot_constraints_max = [max(row) for index, row in pd.concat(
            [ineq_constraints_max, eq_constraints_max], axis=1).iterrows()]
        try:
            tot_constraints_max_col = [row.idxmax()
                                       for index, row in tot_constraints.iterrows()]
            tot_constraints_max = [row[tot_constraints_max_col[index]]
                                   for index, row in tot_constraints.iterrows()]
        except:
            tot_constraints_max_col = [
                0 for index, row in tot_constraints.iterrows()]
            tot_constraints_max = [0 for index,
            row in tot_constraints.iterrows()]
        fm_ineq = [value
                   for value in optim_output[self.INEQ_CONSTRAINT].values]
        fm_eq = [value
                 for value in optim_output[self.EQ_CONSTRAINT].values]
        tot_constraints_text = [zipped for zipped in zip(
            fm_ineq, fm_eq, tot_constraints_max_col, tot_constraints_max, tot_constraints_sum)]
        hover_text = [
            'Iteration : {}<br />INEQ constraint : {}<br />EQ constraint : {}<br />Max constraint name : {}<br />Max constraint value : {}<br />Summed constraint: {}'.format(
                i + 1, t[0], t[1], t[2], t[3], t[4]) for i, t in enumerate(tot_constraints_text)]

        def f_log10(value):
            if value < 0:
                log10_value = -np.log10(abs(value)) - 1.0
            else:
                log10_value = np.log10(value) + 1.0
            return log10_value

        values = [value if abs(value) <= 1.0 else f_log10(value)
                  for value in tot_constraints_sum]

        xmap = list(np.linspace(x[0] - 0.5, x[-1] + 0.5, len(x) + 1))
        dy = max(y) - min(y)
        y0 = min(y) + dy / 2
        fig.add_trace(go.Heatmap(x=xmap, y0=y0, dy=dy, z=[values, ], hoverinfo='text', hovertext=[hover_text, ],
                                 colorscale=['green', 'white', 'red'], opacity=0.5,
                                 zmid=0.0,
                                 colorbar={"title": 'total constraints (sum) [symlog]',
                                           'x': 1.1, 'yanchor': "middle", 'titleside': 'right'}))

        fig.update_layout(title={'text': chart_name, 'x': 0.5, 'y': 1.0, 'xanchor': 'center', 'yanchor': 'top'},
                          xaxis_title='n iterations', yaxis_title=f'value of {name}')
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"yaxis.type": "linear"}],
                            label="Linear",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis.type": "log"}],
                            label="Log",
                            method="relayout"
                        ),
                    ]),
                    type="buttons",
                    direction="right",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=0,
                    x=0.0,
                    xanchor="left",
                    y=1.01,
                    yanchor="bottom"
                ),
            ]
        )
        new_chart = InstantiatedPlotlyNativeChart(
            fig, chart_name=chart_name, default_title=True)
        return new_chart

    def __build_mod_names(self, f):
        ''' returns the functions out names
        '''
        return f + self.MOD_SUFFIX

    def check_value_range(self, fvalue_df, fname: str, ftype: str):
        """Log some warnings if value is not between 0 and 1"""
        if isinstance(fvalue_df, np.ndarray):
            if fvalue_df.max() > 10:
                self.logger.warning(f"{ftype} {fname} maximum is above 10 ({fvalue_df.max()}). Ideally all its values should be between 0 and 1")
            if fvalue_df.min() < -10:
                self.logger.warning(f"{ftype} {fname} minimum is lower than -10 ({fvalue_df.min()}). Ideally all its values should be between 0 and 1")
