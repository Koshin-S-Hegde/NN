import numpy as np
import matplotlib.pyplot as plt


class DataHandler:
    slope: float
    c: float

    def __init__(self, slope: float = 1, c: float = 0) -> None:
        self.slope = slope
        self.c = c

    def get_value(self, point: np.ndarray) -> int:
        value: int = np.sign(point[1] - self.slope * point[0] - self.c)
        return 1 if value > 0 else 0

    @staticmethod
    def get_random_data(size: int) -> np.ndarray:
        return np.random.uniform(low=-10, high=10, size=(size, 2))

    def plot(self, data: np.ndarray) -> None:
        positive_points: list = list()
        negative_points: list = list()

        for point in data:
            if self.get_value(point) == 1:
                positive_points.append(point)
            else:
                negative_points.append(point)

        positive_x: list = [point[0] for point in positive_points]
        positive_y: list = [point[1] for point in positive_points]
        negative_x: list = [point[0] for point in negative_points]
        negative_y: list = [point[1] for point in negative_points]

        plt.scatter(positive_x, positive_y, c="green")
        plt.scatter(negative_x, negative_y, c="red")

    @staticmethod
    def plot_axes() -> None:
        plt.axhline(0, c="black")
        plt.axvline(0, c="black")

    def plot_line(self) -> None:
        plt.axline((0, self.c), slope=self.slope, c="yellow")
