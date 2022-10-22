from typing import Callable

import numpy as np


class Layer:
    weights: np.ndarray
    bias: np.ndarray
    learning_rate: float
    activation_function: Callable[[float], float]
    differential_of_activation_function: Callable[[float], float]

    def __init__(
            self,
            output_size: int,
            input_size: int,
            activation_function: Callable[[float], float],
            differential_of_activation_function: Callable[[float], float],
            learning_rate: float
    ) -> None:
        self.differential_of_activation_function = differential_of_activation_function
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.learning_rate = learning_rate

    def propagate_forward(self, inputs: np.ndarray) -> np.ndarray:
        raw_output: np.ndarray = (np.dot(self.weights, inputs) + self.bias)[0]
        return np.vectorize(self.activation_function)(raw_output)

    def propagate_backward_output(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        output: np.ndarray = self.propagate_forward(inputs)
        error: np.ndarray = target - output

        delta_weights: np.ndarray = inputs * error
        delta_bias: np.ndarray = error * np.ones(self.bias.shape)

        self.weights += delta_weights * self.learning_rate
        self.bias += delta_bias * self.learning_rate
        """

        # Calculating:-
        # d(error total) / d(weight) =
        # [d(error total) / d(corresponding output)] *
        # [d(corresponding output) / d(corresponding raw output)] *
        # [d(corresponding raw output) / d(corresponding weight)]

        # ------ IMPORTANT --------
        # We are using mean squared error here

        output: np.ndarray = self.propagate_forward(inputs)
        raw_output: np.ndarray = np.dot(self.weights, inputs)
        original_weights: np.ndarray = self.weights

        for output_index in range(len(self.weights)):
            current_target: float = target[output_index]
            current_output: float = output[output_index]
            current_raw_output: float = raw_output[output_index]
            current_bias: float = self.bias[output_index]

            # [d(error total) / d(corresponding output)] =
            # [d(mean(squared errors)) / d(corresponding output)] =
            # [1/len(errors) * 2 * (target - output) ** (2-1) ) * d(-output) / d(output)] =
            # [(2/len(len(weights)) * (output - target)]
            d_error_by_d_output: float = (2 / len(self.weights)) * (current_output - current_target)

            # [d(corresponding output) / d(corresponding raw output)] =
            d_output_by_d_raw_output: float = self.differential_of_activation_function(current_raw_output)

            for input_index in range(len(inputs)):
                current_weight: float = self.weights[output_index][input_index]
                current_input: float = inputs[input_index]

                # d(corresponding raw output) / d(corresponding weight) =
                # d( w1 * in1 + w2 * in2 + ... + b ) / d(corresponding weight) =
                # corresponding input
                d_raw_output_by_d_weight: float = current_input

                d_error_by_d_weight: float = d_error_by_d_output * \
                    d_output_by_d_raw_output * \
                    d_raw_output_by_d_weight

                new_weight: float = current_weight - self.learning_rate * d_error_by_d_weight
                self.weights[output_index][input_index] = new_weight

            # [d(error) / d(bias) =
            # [d(error) / d(output)] *
            # [d(output) / d(raw output)] *
            # [d(raw output) / d(bias)]
            # But d(waw output) / d(bias) = 1
            # => [d(error) / d(bias) =
            # [d(error) / d(output)] *
            # [d(output) / d(raw output)]
            d_error_by_d_bias: float = d_error_by_d_output * d_output_by_d_raw_output
            new_bias: float = current_bias - self.learning_rate * d_error_by_d_bias
            self.bias[output_index] = new_bias
        return original_weights

    # def propagate_backward_hidden(
    #         self,
    #         inputs: np.ndarray,
    #         target: np.ndarray,
    #         weights_previous: np.ndarray
    # ) -> np.ndarray:
