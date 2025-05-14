'''
Copyright 2025/02/17 Capgemini

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
import warnings
from os.path import dirname

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

from sostrades_optimization_plugins.models.autodifferentiated_discipline import (
    AutodifferentiedDisc,
)
from sostrades_optimization_plugins.tools.discipline_tester import (
    discipline_test_function,
)

warnings.filterwarnings("ignore")


class GenericDisciplinesTestClass(unittest.TestCase):
    """Ethanol Fuel jacobian test class"""

    name = 'Test'
    override_dump_jacobian = False
    show_graphs = False
    jacobian_test = True
    mod_path = "to_fill"
    pickle_directory = dirname(__file__)
    ns_dict = {}
    pickle_prefix = ""
    model_name = "to_fill"
    inputs_dicts = {}

    # GRADIENTS TUNING: usefull for Autodifferentiated disciplines.
    # When self configuration of gradient variable is enabled, the discipline uses autodifferentiation to
    # evaluate all coupling output wrt all coupling input. The trouble is that is is quite slow and a significant
    # combinations of I/O have null gradients. Activating the following class variable stores all the null gradient
    # combinations for all tests of the discipline then stores all the combinations that were null in every tests.
    # the discipline then drop a json cache file of the combination to avoid in real usages.

    gradients_tuning: bool = False

    _null_gradients_dict_list: dict[str: list[dict[str: list[str]]]] = []
    _disciplines: dict[str: SoSWrapp] = {}

    def tearDown(self) -> None:
        if hasattr(self, "stream_name"):
            self.pickle_prefix = getattr(self, "stream_name")
        inputs = self.get_inputs_dict()
        inputs.update(self.inputs_dicts)

        self._disciplines[self.mod_path] = discipline_test_function(
            module_path=self.mod_path,
            model_name=self.model_name,
            gradients_tuning=self.gradients_tuning,
            name=self.name,
            jacobian_test=self.jacobian_test,
            show_graphs=self.show_graphs,
            inputs_dict=inputs,
            namespaces_dict=self.ns_dict,
            pickle_directory=self.pickle_directory,
            pickle_name=f'{self.pickle_prefix}_{self.model_name}.pkl',
            override_dump_jacobian=self.override_dump_jacobian
        )
        if isinstance(self._disciplines[self.mod_path], AutodifferentiedDisc) and self.gradients_tuning:
            if self.mod_path not in self._null_gradients_dict_list:
                self._null_gradients_dict_list[self.mod_path] = []
            self._null_gradients_dict_list[self.mod_path].append(self._disciplines[self.mod_path].model.null_gradients_tuning)

    @classmethod
    def setUpClass(cls) -> None:
        cls._null_gradients_dict_list = {}
    @classmethod
    def handle_gradient_tuning(cls):
        if cls.gradients_tuning:
            for mod_path, null_gradients_list in cls._null_gradients_dict_list.items():
                always_null_gradients = cls.intersect_multiple_dicts(null_gradients_list)
                cls._disciplines[mod_path].dump_null_gradients_cache(always_null_gradients)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.handle_gradient_tuning()

    def get_inputs_dict(self) -> dict:
        raise NotImplementedError("Must be overloaded")


    @staticmethod
    def intersect_multiple_dicts(dicts: list[dict[str:dict[str: bool]]]):
        concat_result = {}

        for dict in dicts:
            for output in dict.keys():
                if output not in concat_result:
                    concat_result[output] = {}
                for input, null_gradient_bool in dict[output].items():
                    if input not in concat_result[output]:
                        concat_result[output][input] = []
                    concat_result[output][input].append(null_gradient_bool)

        always_null_gradients = {}
        for output in concat_result.keys():
            for input, list_is_null_bools in concat_result[output].items():
                if all(list_is_null_bools):
                    if output not in always_null_gradients:
                        always_null_gradients[output] = []
                    always_null_gradients[output].append(input)


        return always_null_gradients

"""
a1 = {
    "o1": {"i1": True, "i2": False}
}

a2 = {
    "o1": {"i1": True, "i2": False ,"i3": True, "i4": True},
    "o2": {"i1": False, "i2": False ,"i3": True},
}

a3 = {
    "o1": {"i1": True, "i2": False ,"i3": True, "i4": False}
}

result = GenericDisciplinesTestClass.intersect_multiple_dicts([a1, a2, a3])
assert 'o1' in result
assert 'i1' in result['o1']
assert 'i3' in result['o1']
assert 'i2' not in result['o1']
assert 'o2' in result
assert 'i3' in result['o2']
assert 'i1' not in result['o2']
assert 'i2' not in result['o2']
"""
