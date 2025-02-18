'''
Copyright 2024 Capgemini

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
from typing import TYPE_CHECKING, Union

import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

if TYPE_CHECKING:
    from sostrades_optimization_plugins.models.differentiable_model import (
        DifferentiableModel,
    )


class AutodifferentiedDisc(SoSWrapp):
    """Discipline which model is a DifferentiableModel"""
    GRADIENTS = "gradient"
    coupling_inputs = []  # inputs verified during jacobian test
    coupling_outputs = []  # outputs verified during jacobian test
    autoconfigure_gradient_variables: bool = True

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.model: Union[DifferentiableModel, None] = None

    def collect_var_for_dynamic_setup(self, variable_names: Union[str, list[str]]):
        """easy method for setup sos dynamic variable gathering"""
        values_dict = {}
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        go = set(self.get_data_in().keys()).issuperset(variable_names)
        if go:
            values_dict = {vn: self.get_sosdisc_inputs(vn) for vn in variable_names}
            go = not any(val is None for val in values_dict.values())

        return values_dict, go

    def run(self):

        # todo : remove filtration later when we will be able to collect only non-numerical inputs
        inputs = self.get_non_numerical_inputs()
        inputs_filtered = {key: value for key, value in inputs.items() if value is not None}
        self.model.set_inputs(inputs_filtered)
        self.model.compute()
        outputs = self.model.get_all_variables()
        self.store_sos_outputs_values(outputs)

    def get_non_numerical_inputs(self):
        inputs = self.get_sosdisc_inputs()
        return {key: value for key, value in inputs.items() if value is not None}

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """

        if self.autoconfigure_gradient_variables:
            self._auto_configure_jacobian_variables()
        # dataframes variables
        all_inputs_dict = {**self.DESC_IN, **self.inst_desc_in}
        all_outputs_dict = {**self.DESC_OUT, **self.inst_desc_out}
        coupling_dataframe_input = list(filter(lambda x: all_inputs_dict[x]['type'] == 'dataframe', self.coupling_inputs))
        coupling_dataframe_output = list(filter(lambda x: all_outputs_dict[x]['type'] == 'dataframe', self.coupling_outputs))
        other_coupling_inputs = list(set(self.coupling_inputs) - set(coupling_dataframe_input))
        other_coupling_outputs = list(set(self.coupling_outputs) - set(coupling_dataframe_output))

        all_inputs_model_path = other_coupling_inputs
        for c_i_df in coupling_dataframe_input:
            all_inputs_model_path.extend(self.model.get_df_input_dotpaths(df_inputname=c_i_df))

        all_inputs_model_path = list(filter(lambda x: not x.endswith(":years"), all_inputs_model_path))

        all_outputs_model_path = other_coupling_outputs
        for c_o_df in coupling_dataframe_output:
            all_outputs_model_path.extend(self.model.get_df_output_dotpaths(df_outputname=c_o_df))
        all_outputs_model_path = list(filter(lambda x: not x.endswith(":years"), all_outputs_model_path))

        def handle_gradients_wrt_inputs(output_path: str, gradients: dict):
            arg_output = (output_path,)
            if ':' in output_path:
                arg_output = tuple(output_path.split(':'))

            for input_path, grad_input_value in gradients.items():
                arg_input = (input_path,)
                if ':' in input_path:
                    arg_input = tuple(input_path.split(':'))
                if len(grad_input_value.shape) == 0:
                    grad_input_value = np.array([[grad_input_value]])
                self.set_partial_derivative_for_other_types(arg_output, arg_input, grad_input_value)

        for output_path in all_outputs_model_path:
            gradients = self.model.compute_partial(output_name=output_path, input_names=all_inputs_model_path)
            handle_gradients_wrt_inputs(output_path=output_path, gradients=gradients)

    def _auto_configure_jacobian_variables(self):
        self.coupling_inputs = []
        all_inputs_dict = {**self.DESC_IN, **self.inst_desc_in}
        for varname, vardescr in all_inputs_dict.items():
            if self.GRADIENTS in vardescr and vardescr[self.GRADIENTS]:
                self.coupling_inputs.append(varname)

        self.coupling_outputs = []
        all_outputs_dict = {**self.DESC_OUT, **self.inst_desc_out}
        for varname, vardescr in all_outputs_dict.items():
            if self.GRADIENTS in vardescr and vardescr[self.GRADIENTS]:
                self.coupling_outputs.append(varname)

    def pimp_string(self, val: str):
        val = val.replace("_", ' ')
        val = val.capitalize()
        return val
