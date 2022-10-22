import matplotlib.pyplot as plt
import numpy as np

from activation_function import identity, differential_of_identity
from data import DataHandler
from layer import Layer


def plot(layers: list[Layer], data: np.ndarray) -> None:
    point: np.ndarray
    for point in data:
        previous_layer_output: np.ndarray = point
        for layer in layers:
            previous_layer_output = layer.propagate_forward(previous_layer_output)
        is_above: bool = (previous_layer_output[0] >= 0.5)
        color: str
        if is_above:
            color = "green"
        else:
            color = "red"
        plt.plot(point[0], point[1], marker="o", markersize=5, markeredgecolor=color, markerfacecolor=color)


def main() -> None:
    data_handler: DataHandler = DataHandler(slope=3, c=0)
    # input_layer: Layer = Layer(
    #     input_size=2,
    #     output_size=3,
    #     activation_function=identity,
    #     differential_of_activation_function=differential_of_identity,
    #     learning_rate=0.0001
    # )
    output_layer: Layer = Layer(
        input_size=2,
        output_size=1,
        activation_function=identity,
        differential_of_activation_function=differential_of_identity,
        learning_rate=0.05
    )

    data_handler.plot(data_handler.get_random_data(1000))
    data_handler.plot_line()
    plt.show()

    plot([output_layer], data_handler.get_random_data(1000))
    data_handler.plot_line()
    plt.show()

    for point in data_handler.get_random_data(500):
        output_layer.propagate_backward_output(point, np.array([data_handler.get_value(point)]))

    plot([output_layer], data_handler.get_random_data(1000))
    data_handler.plot_line()
    plt.show()

    while True:
        print("Type:-")
        print(
            "Green" if output_layer.propagate_forward(np.array([
                float(input()),
                float(input())
            ])) == 1 else "Red"
        )


if __name__ == '__main__':
    main()
