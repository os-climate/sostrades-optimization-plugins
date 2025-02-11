"""
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
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Union

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

import autograd
import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
import pandas as pd

ArrayLike = Union[list[float], npt.NDArray[np.float64]]
InputType = Union[float, int, ArrayLike, pd.DataFrame]
OutputType = Union[float, ArrayLike]


class DifferentiableModel:
    """A base class for differentiable models.

    This class provides a framework for creating models that can be differentiated
    with respect to their inputs. It handles input setting, output computation,
    and gradient/Jacobian calculations.

    Attributes:
        inputs (dict): A dictionary to store input values.
        outputs (dict): A dictionary to store computed output values.
        parameters (dict): A dictionary to store model parameters.
        output_types (dict): A dictionary to store output types.
    """

    def __init__(
        self,
        flatten_dfs: bool = True,
        ad_backend: str = "autograd",
        overload_numpy: bool = True,
        numpy_ns: str = "np",
    ) -> None:
        """
        Initialize the model.

        Args:
            flatten_dfs: If True, DataFrames will be flattened into separate arrays
                        with keys as 'dataframe_name:column_name'.
                        If False, DataFrames will be converted to dictionaries of arrays.
            ad_backend: The backend to use for automatic differentiation. Defaults to "autograd".
            overload_numpy: If True, the numpy namespace will be replaced with the
                            appropriate namespace for the backend. Defaults to True.
            numpy_ns: The namespace to use for numpy if overload_numpy is True. Defaults to 'np'.

        """
        if ad_backend == "jax" and not HAS_JAX:
            error_msg = "JAX not installed. Please install JAX to use JAX backend."
            raise ValueError(error_msg)

        self._ad_backend = ad_backend

        # Overload numpy namespace if required / requested
        if overload_numpy:
            self.numpy_ns = numpy_ns
            self.np = anp if self._ad_backend == "autograd" else jnp
        else:
            self.np = np

        # Prepare ad_backend functions
        if ad_backend == "autograd":
            self.__grad = autograd.grad
            self.__jacobian = autograd.jacobian
        elif ad_backend == "jax":
            self.__grad = jax.grad
            self.__jacobian = jax.jacobian

        self.dataframes_outputs_colnames: dict[str : list[str]] = {}
        self.dataframes_inputs_colnames: dict[str : list[str]] = {}

        self.inputs: dict[str, Union[float, np.ndarray, dict[str, np.ndarray]]] = {}
        self.outputs: dict[str, Union[float, np.ndarray, dict[str, np.ndarray]]] = {}

        self._params = {}
        self._output_types = {}
        self.temp_variables = {}

        self.flatten_dfs = flatten_dfs

        # Internal variables
        self._for_grad: bool = False

        # Default methods
        self.compute_partial = self.compute_partial_bwd

    def _reset_outputs(self):
        self.outputs = {}
        self.temp_variables = {}
        self._output_types = {}
        self.dataframes_outputs_colnames: dict[str : list[str]] = {}
        self.dataframes_inputs_colnames: dict[str : list[str]] = {}

    def _reset(self):
        self._reset_outputs()
        self.inputs = {}

    @property
    def parameters(self) -> dict[str, float, np.ndarray, dict]:
        """Get the current parameters of the model.

        Returns:
            dict: A dictionary of parameter names and their values.

        """
        return self.get_parameters()

    @parameters.setter
    def parameters(self, value: dict[str, float, np.ndarray, dict]) -> None:
        """Set the parameters of the model.

        Args:
            value (dict): A dictionary of parameter names and their values.

        """
        self.set_parameters(value)

    def set_parameters(self, params: dict[str, float, np.ndarray]) -> None:
        """Set the parameters of the model.

        Args:
            params (dict): A dictionary of parameter names and their values.

        """
        self._params = params

    def get_parameters(self) -> dict[str, float, np.ndarray]:
        """Retrieve the current parameters of the model.

        Returns:
            dict: A dictionary of parameter names and their values.

        """
        return self._params

    def set_inputs(self, inputs_in: dict[str, InputType]) -> None:
        """Set the input values for the model.

        Args:
            inputs_in (dict): A dictionary containing input names and their values.
                Can contain nested dictionaries with DataFrames or arrays.

        Raises:
            TypeError: If a DataFrame input contains non-numeric data.
            ValueError: If an input array has more than 2 dimensions.

        Examples:
            >>> inputs = {
            ...     'simple_array': [1, 2, 3],
            ...     'level1': {
            ...         'df1': pd.DataFrame({
            ...             'A': [1, 2, 3],
            ...             'B': [4, 5, 6]
            ...         }),
            ...         'level2': {
            ...             'df2': pd.DataFrame({
            ...                 'C': [7, 8, 9],
            ...                 'D': [10, 11, 12]
            ...             }),
            ...             'array': [1, 2, 3]
            ...         }
            ...     }
            ... }

            With self.flatten_dfs = True:
            >>> model.set_inputs(inputs)
            # Results in:
            # self.inputs = {
            ...     'simple_array': array([1, 2, 3]),
            ...     'level1': {
            ...         'df1:A': array([1, 2, 3]),
            ...         'df1:B': array([4, 5, 6]),
            ...         'level2': {
            ...             'df2:C': array([7, 8, 9]),
            ...             'df2:D': array([10, 11, 12]),
            ...             'array': array([1, 2, 3])
            ...         }
            ...     }
            ... }

            With self.flatten_dfs = False:
            >>> model.set_inputs(inputs)
            # Results in:
            # self.inputs = {
            ...     'simple_array': array([1, 2, 3]),
            ...     'level1': {
            ...         'df1': {
            ...             'A': array([1, 2, 3]),
            ...             'B': array([4, 5, 6])
            ...         },
            ...         'level2': {
            ...             'df2': {
            ...                 'C': array([7, 8, 9]),
            ...                 'D': array([10, 11, 12])
            ...             },
            ...             'array': array([1, 2, 3])
            ...         }
            ...     }
            ... }
        """
        self._reset()
        self.dataframes_inputs_colnames = {}
        self.inputs = self._process_input_dict(inputs_in)

    def _process_input_dict(self, input_dict: dict, parent_key: str = "") -> dict:
        """Recursively process input dictionary to handle nested structures.

        Args:
            input_dict (dict): Dictionary to process
            parent_key (str): Parent key for nested structures

        Returns:
            dict: Processed dictionary with flattened DataFrames if self.flatten_dfs is True

        Examples:
            >>> nested_dict = {
            ...     'level1': {
            ...         'df1': pd.DataFrame({
            ...             'A': [1, 2, 3]
            ...         }),
            ...         'array': [4, 5, 6]
            ...     }
            ... }

            >>> model._process_input_dict(nested_dict)  # with flatten_dfs = True
            {
                'level1': {
                    'df1:A': array([1, 2, 3]),
                    'array': array([4, 5, 6])
                }
            }
        """
        processed_inputs = {}

        for key, value in input_dict.items():
            current_key = f"{parent_key}:{key}" if parent_key else key

            if isinstance(value, dict):
                processed_inputs[key] = self._process_input_dict(value, key)

            elif isinstance(value, pd.DataFrame):
                if not all(np.issubdtype(dtype, np.number) for dtype in value.dtypes):
                    msg = f"DataFrame '{current_key}' contains non-numeric data."
                    raise TypeError(msg)

                if self.flatten_dfs:
                    self.dataframes_inputs_colnames[current_key] = list(value.columns)
                    df_dict = {
                        f"{key}:{col}": value[col].to_numpy() for col in value.columns
                    }
                    processed_inputs.update(df_dict)
                else:
                    processed_inputs[key] = {
                        col: value[col].to_numpy() for col in value.columns
                    }

            elif isinstance(value, pd.Series):
                processed_inputs[key] = value.to_numpy()

            elif isinstance(value, (list, np.ndarray)):
                if len(np.array(value).shape) > 2:
                    msg = f"Input '{current_key}' has too many dimensions; only 1D or 2D arrays allowed."
                    raise ValueError(msg)
                processed_inputs[key] = np.array(value)

            else:
                processed_inputs[key] = value

        return processed_inputs

    def set_output(self, key: str, value) -> None:
        """Set the output key with value."""
        self.outputs[key] = value

    def set_output_types(self, output_types: dict[str, str]) -> None:
        """Set the types of the output variables.

        Args:
            output_types (dict): A dictionary of output names and their types.

        """
        self.output_types = output_types

    def get_output_df_names(self) -> dict:
        """Retreive."""

        result = defaultdict(list)

        if self.flatten_dfs:
            # Find all unique base names in flattened outputs
            for key in self.outputs:
                if ":" not in key:
                    continue
                base, child = key.split(":")
                result[base].append(child)

        # Check for dictionary outputs
        for key, value in self.outputs.items():
            if isinstance(value, dict):
                result[key] = list(value.keys())

        return dict(result)

    def get_dataframe(
        self,
        name: str,
        get_from: str = "outputs",
    ) -> pd.DataFrame | None:
        """Retrieve a specific DataFrame from outputs or inputs based on its name.

        Works with both dictionary outputs and flattened outputs.

        Args:
            name: Name of the DataFrame to retrieve
            get_from: Source of the DataFrame, either "outputs" or "inputs"

        Returns:
            DataFrame if it can be constructed from outputs or inputs, None otherwise

        """
        if get_from == "inputs":
            source = self.inputs
        elif get_from == "outputs":
            source = self.outputs
        else:
            source = self.outputs

        # First check if there's a direct dictionary output with this name
        if (
            name in source
            and isinstance(source[name], dict)
            and all(isinstance(v, np.ndarray) for v in source[name].values())
        ):
            return pd.DataFrame(source[name])

        # If using flatten_dfs, check for columns with this base name
        # if self.flatten_dfs:
        prefix = f"{name}:"
        columns = {}
        columns_in_sources = list(
            map(
                lambda x: x[0],
                list(
                    filter(
                        lambda item: item[0].startswith(prefix)
                        and isinstance(item[1], np.ndarray),
                        source.items(),
                    )
                ),
            )
        )
        for col_with_prefix in columns_in_sources:
            col_name = col_with_prefix[
                len(prefix) :
            ]  # Remove the prefix to get column name
            columns[col_name] = source[col_with_prefix]

        if columns:  # Only create DataFrame if we found matching columns
            return pd.DataFrame(columns)
        else:
            a = 1

        return None

    def get_base_name(self, get_from: str = "outputs"):
        if get_from == "inputs":
            source = self.inputs
        elif get_from == "outputs":
            source = self.outputs
        else:
            source = self.outputs

        return {key.split(":", 1)[0] for key in source if ":" in key}

    def get_dataframes(self, get_from: str = "outputs") -> dict[str, pd.DataFrame]:
        """Convert all suitable outputs or inputs to pandas DataFrames.

        Args:
            get_from: Source of the DataFrame, either "outputs" or "inputs".
                Defaults to "outputs".

        Returns:
            Dictionary of DataFrames reconstructed from outputs

        """
        result = {}

        if get_from == "inputs":
            source = self.inputs
        elif get_from == "outputs":
            source = self.outputs
        else:
            source = self.outputs

        if self.flatten_dfs:
            # Find all unique base names in flattened outputs
            base_names = self.get_base_name(get_from=get_from)
            for base_name in base_names:
                df = self.get_dataframe(base_name)
                if df is not None:
                    result[base_name] = df

        # Check for dictionary outputs
        for key, value in source.items():
            if isinstance(value, dict):
                df = self.get_dataframe(key)
                if df is not None:
                    result[key] = df

        return result

    def get_all_variables(self, get_from: str = "outputs") -> dict[str, Any]:
        """Retrieves all variables (in or out) while converting dataframes"""
        result = self.get_dataframes(get_from=get_from)
        self.dataframes_outputs_colnames = self.get_output_df_names()
        if get_from == "inputs":
            source = self.inputs
        elif get_from == "outputs":
            source = self.outputs
        else:
            source = self.outputs

        for key, value in source.items():
            if ":" not in key:
                result[key] = value

        return result

    def compute(self, *args: InputType) -> OutputType:
        """Compute the model outputs based on inputs passed as arguments."""
        self._compute(*args)

    def _compute(self, *args: InputType) -> OutputType:
        """Compute the model outputs based on inputs passed as arguments.

        This method should be overridden by subclasses.

        Args:
            *args: Variable length argument list of input values.

        Returns:
            OutputType: The computed output.

        Raises:
            NotImplementedError: If not implemented in a subclass.

        """
        msg = "Subclasses must implement the compute method."
        raise NotImplementedError(msg)

    def get_outputs(self) -> dict[str, OutputType]:
        """Retrieve the computed outputs.

        Returns:
            dict: A dictionary of output names and their computed values.

        """
        return self.outputs

    def _create_wapped_compute_bwd(
        self,
        output_name: str,
        input_names: Union[str, list] = None,
        all_inputs: bool = False,
    ) -> Callable | list[Callable]:
        """Create wrapped compute functions for a specific output and inputs.

        Args:
            output_name (str): The name of the output.
            input_names (str, list): The name of the input. Can be a list of inputs.
            all_inputs (bool): Whether to compute the derivative with respect to all
            inputs.

        Returns:
            (Callable, list[Callable]): A single wrapped compute function or a list of
            the wrapped compute functions for each input.

        """
        # Make sure either input_names or all_inputs is provided
        if input_names is None and all_inputs is False:
            msg = "Either input_names or all_inputs must be provided."
            raise ValueError(msg)

        # Ensure input_names is a list
        if isinstance(input_names, str):
            input_names = [input_names]

        wrapped_computes = None

        if all_inputs:

            def wrapped_compute(args: InputType) -> OutputType:
                temp_inputs = deepcopy(self.inputs)
                self.inputs = args
                self._for_grad = True
                self.compute()
                self._for_grad = False
                self.inputs = temp_inputs
                return self.outputs[output_name]

            wrapped_computes = wrapped_compute

        else:
            wrapped_computes = []

            for input_name in input_names:
                if isinstance(self.inputs[input_name], dict):

                    def wrapped_compute(
                        *args: InputType,
                        input_name: str = input_name,
                    ) -> OutputType:
                        temp_inputs = deepcopy(self.inputs)
                        for i, col in enumerate(self.inputs[input_name].keys()):
                            self.inputs[input_name][col] = args[i]
                        self._for_grad = True
                        self.compute()
                        self._for_grad = False
                        self.inputs = temp_inputs
                        return self.outputs[output_name]
                else:

                    def wrapped_compute(
                        arg: InputType,
                        input_name: str = input_name,
                    ) -> OutputType:
                        temp_inputs = deepcopy(self.inputs)
                        self.inputs[input_name] = arg
                        self._for_grad = True
                        self.compute()
                        self._for_grad = False
                        self.inputs = temp_inputs
                        return self.outputs[output_name]

                wrapped_computes.append(wrapped_compute)

        return wrapped_computes

    def compute_partial_bwd(
        self, output_name: str, input_names: str | list, all_inputs: bool = False
    ) -> (
        npt.NDArray[np.float64]
        | dict[str, npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]]
    ):
        """Compute the partial derivative of an output with respect to an input or all inputs.

        Args:
            output_name (str): The name of the output to compute the derivative for.
            input_names (Union[str, list]): The name or list of names of the input(s)
                                            with respect to which the derivative is computed.
            all_inputs (bool): Flag indicating whether to compute the derivative with respect
                               to all inputs at once.

        Returns:
            Union[npt.NDArray[np.float64], dict[str, Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]]]:
                The computed partial derivative(s) as a NumPy ndarray or a dictionary of arrays.

        """
        # pylint: disable=E1120

        is_single = False
        if isinstance(input_names, str):
            is_single = True
            input_names = [input_names]

        result = {}

        # Create wrapped compute functions making sure only asking for all inputs if using jax
        wrapped_computes = self._create_wapped_compute_bwd(
            output_name,
            input_names,
            all_inputs=all_inputs if self._ad_backend == "jax" else False,
        )

        # If all_inputs is True, compute the jacobian using all inputs at once
        if all_inputs:
            wrapped_compute = wrapped_computes
            jacobian_func = self.__jacobian(wrapped_compute)
            result = jacobian_func(self.inputs)

        else:  # If not, compute the jacobian for each input
            for wrapped_compute, input_name in zip(wrapped_computes, input_names):
                self._reset_outputs()
                if isinstance(self.inputs[input_name], dict):  # For DataFrame inputs
                    jacobians = {}
                    argnum_kword = (
                        "argnum" if self._ad_backend == "autograd" else "argnums"
                    )
                    for col in self.inputs[input_name]:
                        jacobian_func = self.__jacobian(
                            wrapped_compute,
                            **{
                                argnum_kword: list(
                                    self.inputs[input_name].keys()
                                ).index(
                                    col,
                                ),
                            },
                        )
                        jacobians[col] = jacobian_func(
                            *self.inputs[input_name].values()
                        )

                    result[input_name] = jacobians

                else:  # For other inputs
                    jacobian_func = self.__jacobian(wrapped_compute)
                    result[input_name] = jacobian_func(self.inputs[input_name])

            if is_single:
                return result[input_names[0]]

        return result

    def compute_partial_all_inputs(self, output_name: str) -> dict:
        """Compute the Jacobian of the model output with respect to all inputs.

        Computes the Jacobian of the model output with respect to all inputs. This
        is useful for computing the Jacobian when the model has multiple inputs and
        you want to get the Jacobian with respect to all of them at once.

        Args:
            output_name (str):The name of the output of the model for which to compute
            the Jacobian.

        Returns:
            result (dict): A dictionary where the keys are the names of the inputs and
            the values are the Jacobians of the output with respect to the inputs.

        """
        if self._ad_backend == "autograd":
            result = {}
            for key in self.inputs:
                partial = self.compute_partial(output_name, key)
                result[key] = partial
        else:  # JAX
            result = self.compute_partial(
                output_name, list(self.inputs.keys()), all_inputs=True
            )

        return result

    def _create_wrapped_compute_array(
        self, output_name: str, input_names: list[str] = None
    ) -> Callable:
        """Creates a wrapped compute function that accepts a single array for multiple inputs.

        Args:
            output_name (str): The name of the output.
            input_names (List[str]): List of input names to include in the wrapper.
                If None, all inputs are used.

        Returns:
            Callable: A wrapped compute function that accepts a single 1D numpy array.
        """
        if input_names is None:
            input_names = list(self.inputs.keys())

        # Get the shapes once to avoid repeated calls
        _, shapes, _ = self._inputs_to_array(input_names)

        def wrapped_compute(flat_array: np.ndarray):
            # Store original state
            temp_inputs = deepcopy(self.inputs)
            # temp_outputs = deepcopy(self.outputs)

            # Convert flat array back to dictionary and update inputs
            restored_dict = self._array_to_dict(flat_array, input_names, shapes)
            for key, value in restored_dict.items():
                self.inputs[key] = value

            # Compute and get result
            self.compute()
            return_value = self.outputs[output_name]

            # Restore original state
            self.inputs = temp_inputs
            # self.outputs = temp_outputs

            return return_value

        return wrapped_compute

    def _inputs_to_array(self, keys):
        """
        Convert selected inputs items into a 1D numpy array.

        Args:
            keys (list): List of keys to include in the array

        Returns:
            tuple: (concatenated array, list of shapes, total length)
        """
        arrays = []
        shapes = []
        total_length = 0

        for key in keys:
            if isinstance(self.inputs[key], float):
                arr = np.array([self.inputs[key]])
                shapes.append(arr.shape)
            else:
                arr = self.inputs[key].reshape(-1)  # Flatten the array
                shapes.append(self.inputs[key].shape)

            total_length += len(arr)
            arrays.append(arr)

        return np.concatenate(arrays), shapes, total_length

    def _array_to_dict(self, array, keys, shapes):
        """
        Convert 1D array back to dictionary with original shapes.

        Args:
            array (np.ndarray): 1D array containing all values
            keys (list): List of keys in the same order as dict_to_array
            shapes (list): Original shapes of arrays from dict_to_array

        Returns:
            dict: Dictionary with reshaped arrays
        """
        result = {}
        start_idx = 0

        for key, shape in zip(keys, shapes):
            size = np.prod(shape)
            arr = array[start_idx : start_idx + size]
            result[key] = arr.reshape(shape)
            start_idx += size

        return result

    def compute_partial_multiple(
        self, output_name: str, input_names: Union[str, list]
    ) -> Union[
        npt.NDArray[np.float64],
        dict[str, Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]],
    ]:
        """Computes the partial derivative of an output with respect to an input or all inputs.

        Args:
            output_name (str): The name of the output.
            input_names (str): The name of the input.

        Returns:
            Union[npt.NDArray[np.float64], Dict[str, Union[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]]]:
                The computed partial derivative(s).
        """
        # pylint: disable=E1120

        wrapped_compute = self._create_wrapped_compute_array(output_name, input_names)

        inputs_array, shapes, _ = self._inputs_to_array(input_names)

        jacobian_func = self.__jacobian(wrapped_compute)
        jac_array = jacobian_func(inputs_array)

        # Convert Jacobian array back to dictionary format
        output_shape = self.outputs[output_name].shape
        result = {}
        start_idx = 0

        for key, shape in zip(input_names, shapes):
            size = np.prod(shape)
            # Reshape the Jacobian slice for this input
            # Combine output shape with input shape
            full_shape = output_shape + shape
            jac_slice = jac_array[:, start_idx : start_idx + size].reshape(full_shape)
            result[key] = jac_slice
            start_idx += size

        return result

    def compute_partial_numeric(
        self,
        output_name: str,
        input_name: str,
        method: str = "complex_step",
        epsilon: float = 1e-8,
    ) -> dict[str, Union[np.ndarray, float, bool]]:
        """Compute the partial derivative of an output with respect to an input using a numerical method.

        Args:
            output_name (str): The name of the output to compute the derivative for.
            input_name (str): The name of the input with respect to which the derivative is computed.
            method (Literal["finite_differences", "complex_step"]): The numerical method to use.
        """

        def finite_difference(f, x, i, eps=epsilon):
            x_plus = x.copy()
            x_plus[i] += eps
            return (f(x_plus) - f(x)) / eps

        def complex_step(f, x, i, eps=epsilon):
            x_complex = x.copy().astype(complex)
            x_complex[i] += 1j * eps
            return np.imag(f(x_complex)) / eps

        # Prepare for numerical approximation
        wrapped_compute = self._create_wapped_compute_bwd(output_name, input_name)[0]

        if isinstance(self.inputs[input_name], dict):
            numerical = {}
            for col, value in self.inputs[input_name].items():
                if np.isscalar(value):

                    def f(x, col: str = col) -> Callable:
                        temp_inputs = self.inputs[input_name].copy()
                        temp_inputs[col] = x
                        return wrapped_compute(*temp_inputs.values())

                    if method == "finite_differences":
                        numerical[col] = finite_difference(f, np.array([value]), 0)
                    else:  # complex_step
                        numerical[col] = complex_step(f, np.array([value]), 0)
                else:  # array input
                    numerical[col] = np.zeros(
                        value.shape + self.outputs[output_name].shape
                    )
                    for i in np.ndindex(value.shape):

                        def f(x, col: str = col) -> Callable:
                            temp_inputs = self.inputs[input_name].copy()
                            temp_inputs[col] = x
                            return wrapped_compute(*temp_inputs.values())

                        if method == "finite_differences":
                            numerical[col][i] = finite_difference(f, value, i)
                        else:  # complex_step
                            numerical[col][i] = complex_step(f, value, i)
        else:
            value = self.inputs[input_name]
            if np.isscalar(value):
                if method == "finite_differences":
                    numerical = finite_difference(wrapped_compute, np.array([value]), 0)
                else:  # complex_step
                    numerical = complex_step(wrapped_compute, np.array([value]), 0)
            else:  # array input
                numerical = np.zeros(value.shape + self.outputs[output_name].shape)
                for i in np.ndindex(value.shape):
                    if method == "finite_differences":
                        numerical[i] = finite_difference(wrapped_compute, value, i)
                    else:  # complex_step
                        numerical[i] = complex_step(wrapped_compute, value, i)

        return numerical

    def check_partial(
        self,
        output_name: str,
        input_name: str,
        method: str = "complex_step",
        epsilon: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> dict[str, Union[np.ndarray, float, bool]]:
        """Compare the partial derivative computed by compute_partial with a numerical approximation
        for a specific input-output pair, handling array inputs correctly.

        Args:
            output_name (str): The name of the output.
            input_name (str): The name of the input.
            method (Literal["finite_differences", "complex_step"]): The numerical method to use.
            epsilon (float): Step size for numerical approximation.
            rtol (float): Relative tolerance for comparison.
            atol (float): Absolute tolerance for comparison.

        Returns:
            Dict[str, Union[np.ndarray, float, bool]]: A dictionary containing the analytical derivative,
            numerical approximation, maximum absolute error, maximum relative error, and whether the
            results are within tolerance.

        """

        # Get the analytical partial derivative
        analytical = self.compute_partial(output_name, input_name)

        # Get the numerical partial derivative
        numerical = self.compute_partial_numeric(
            output_name, input_name, method=method, epsilon=epsilon
        )

        # Ensure analytical and numerical have the same shape
        if isinstance(analytical, dict):
            for col in analytical:
                analytical[col] = np.atleast_1d(analytical[col])
                numerical[col] = np.atleast_1d(numerical[col])
        else:
            analytical = np.atleast_1d(analytical)
            numerical = np.atleast_1d(numerical)

        if isinstance(numerical, np.ndarray):
            numerical = numerical.T
        elif isinstance(numerical, dict):
            numerical = {k: v.T for k, v in numerical.items()}

        # Compute errors
        if isinstance(analytical, dict):
            abs_error = {
                col: np.abs(analytical[col] - numerical[col]) for col in analytical
            }
            rel_error = {
                col: abs_error[col] / (np.abs(analytical[col]) + 1e-15)
                for col in analytical
            }
            max_abs_error = max(np.max(err) for err in abs_error.values())
            max_rel_error = max(np.max(err) for err in rel_error.values())
            within_tolerance = all(
                np.allclose(analytical[col], numerical[col], rtol=rtol, atol=atol)
                for col in analytical
            )
        else:
            abs_error = np.abs(analytical - numerical)
            rel_error = abs_error / (np.abs(analytical) + 1e-15)
            max_abs_error = np.max(abs_error)
            max_rel_error = np.max(rel_error)
            within_tolerance = np.allclose(analytical, numerical, rtol=rtol, atol=atol)

        return {
            "analytical": analytical,
            "numerical": numerical,
            "max_absolute_error": float(max_abs_error),
            "max_relative_error": float(max_rel_error),
            "within_tolerance": within_tolerance,
        }

    def get_df_input_dotpaths(self, df_inputname: str) -> dict[str : list[str]]:
        """Get dataframe inputs dotpaths.

        Returns:
            _type_: _description_
        """
        return [
            f"{df_inputname}:{colname}"
            for colname in self.dataframes_inputs_colnames[df_inputname]
        ]

    def get_df_output_dotpaths(self, df_outputname: str) -> dict[str : list[str]]:
        return [
            f"{df_outputname}:{colname}"
            for colname in self.dataframes_outputs_colnames[df_outputname]
        ]

    def get_colnames_output_dataframe(
        self, df_name: str, expect_years: bool = False, full_path: bool = False
    ):
        """Retrieves column names for a specific output DataFrame.

        Args:
            df_name (str): Name of the DataFrame.
            expect_years (bool): If True, excludes the 'years' column from the result.
                Defaults to False.
            full_path (bool): If True, returns full column paths including DataFrame name.
                Defaults to False.

        Returns:
            list[str]: List of column names or full column paths.
        """
        columns_names = list(
            filter(lambda key: key.startswith(f"{df_name}:"), self.outputs.keys())
        )
        if expect_years:
            columns_names.remove(f"{df_name}:years")
        if not full_path:
            columns_names = [col.replace(f"{df_name}:", "") for col in columns_names]
        return columns_names

    def get_cols_output_dataframe(self, df_name: str, expect_years: bool = False):
        """
        Retrieve column values for a specific output DataFrame.

        Args:
            df_name (str): Name of the DataFrame.
            expect_years (bool): If True, excludes the 'years' column from the result.
                Defaults to False.

        Returns:
            list[np.ndarray]: List of column values as numpy arrays.
        """
        columns_names = self.get_colnames_output_dataframe(
            df_name=df_name, expect_years=expect_years, full_path=True
        )
        columns = [self.outputs[col] for col in columns_names]
        return columns

    def get_colnames_input_dataframe(
        self, df_name: str, expect_years: bool = False, full_path: bool = False
    ):
        columns_names = list(
            filter(lambda key: key.startswith(f"{df_name}:"), self.inputs.keys())
        )
        if expect_years:
            columns_names.remove(f"{df_name}:years")
        if not full_path:
            columns_names = [col.replace(f"{df_name}:", "") for col in columns_names]
        return columns_names

    def get_cols_input_dataframe(self, df_name: str, expect_years: bool = False):
        columns_names = self.get_colnames_input_dataframe(
            df_name=df_name, expect_years=expect_years, full_path=True
        )
        columns = [self.inputs[col] for col in columns_names]
        return columns

    def sum_cols(self, cols: list[np.ndarray | ArrayLike]) -> ArrayLike:
        """
        Perform summation of arrays in an autograd-compatible manner.

        Args:
            cols (list[np.ndarray | ArrayLike]): List of arrays to sum.

        Returns:
            ArrayLike: Sum of all input arrays.

        Note:
            This method ensures compatibility with automatic differentiation by avoiding
            direct numpy sum operations.
        """
        sum_result = cols[0] * 0.0 + 0.0
        for col in cols:
            sum_result = sum_result + col
        return sum_result

    @staticmethod
    def _df_to_dict(df: pd.DataFrame, parent_name: str = None) -> dict:
        """Convert a dataframe into a dictionary of numpy arrays.

        Args:
            df (pd.DataFrame): Dataframe to convert.
            parent_name (str, optional): Parent name to prefix column names with.
                If provided, keys will be formatted as 'parent_name:column_name'.
                Defaults to None.

        Returns:
            dict: Dictionary of numpy arrays where keys are column names
                (optionally prefixed with parent_name) and values are
                numpy arrays of the column data
        """
        if parent_name is not None:
            return {f"{parent_name}:{col}": df[col].to_numpy() for col in df.columns}
        return {col: df[col].to_numpy() for col in df.columns}
