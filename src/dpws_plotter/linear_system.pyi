from numpy.typing import NDArray
from sympy import MutableDenseMatrix
from typing import List, Tuple, Union, overload

import numpy as np


type Num_NDArray = Union[
    NDArray[np.number],
    Tuple[Num_NDArray, Num_NDArray],
    List[Num_NDArray],
]

class Linear2DSystem:
    def __init__(self, A: Num_NDArray|str, b: Num_NDArray|str, symbols: str='x y'):
        self.A: np.ndarray = None
        self.b: np.ndarray = None
        self.sym_A: MutableDenseMatrix = None
        self.sym_b: MutableDenseMatrix = None
        self.sym_X: MutableDenseMatrix = None
        self.eigenvectors = None
        self.singularity = None

    def print(self) -> None: ...
    def symrepr(self) -> MutableDenseMatrix: ...
    def eval(self, xy: Num_NDArray) -> Num_NDArray: ...
    def plot(self,
        x_range: Num_NDArray = (-10,10),
        y_range: Num_NDArray = (-10,10),
        n_grid: int = 25, density: float = 1): ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @overload
    def __call__(self, X: Num_NDArray) -> Num_NDArray: ...
    @overload
    def __call__(self, X: MutableDenseMatrix) -> MutableDenseMatrix: ...
    def __add__(self, other: MutableDenseMatrix) -> MutableDenseMatrix: ...
    def __sub__(self, other: MutableDenseMatrix) -> MutableDenseMatrix: ...
    def __mul__(self, other: MutableDenseMatrix) -> MutableDenseMatrix: ...
    def __truediv__(self, other: MutableDenseMatrix) -> MutableDenseMatrix: ...
