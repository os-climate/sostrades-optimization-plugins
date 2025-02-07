'''
Copyright 2025/02/06 Capgemini
Modifications on 2023/06/14-2024/06/24 Copyright 2023 Capgemini

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

    def tearDown(self) -> None:
        if hasattr(self, "stream_name"):
            self.pickle_prefix = getattr(self, "stream_name")

        discipline_test_function(
            module_path=self.mod_path,
            model_name=self.model_name,
            name=self.name,
            jacobian_test=self.jacobian_test,
            show_graphs=self.show_graphs,
            inputs_dict=self.get_inputs_dict(),
            namespaces_dict=self.ns_dict,
            pickle_directory=self.pickle_directory,
            pickle_name=f'{self.pickle_prefix}_{self.model_name}.pkl',
            override_dump_jacobian=self.override_dump_jacobian
        )

    def get_inputs_dict(self) -> dict:
        raise NotImplementedError("Must be overloaded")

