from typing import Union

import matplotlib.pyplot as plt
import numpy.linalg as nplg
import numpy.typing as npt
import numpy as np
import sympy as sp

type NumArray = npt.NDArray[np.number]

class Linear2DSystem:
    def __init__(self, A: Union[NumArray|str], b: Union[NumArray|str], symbols = 'x y'):
        """ Creates a Lienar Dinamical System 2x2:
        dX/dt = A*X + b
        """
        if type(A) is str:
            A = np.asmatrix(A)
        if type(b) is str:
            b = np.asmatrix(b)

        self.A = np.matrix(A)
        self.b = np.matrix(b)
        # check argument dimensions
        if self.A.shape != (2, 2):
            raise ValueError
        if self.b.shape != (2, 1):
            raise ValueError
        
        self.sym_A = sp.Matrix(A)
        self.sym_b = sp.Matrix(b)

        x, y = sp.symbols(symbols)
        self._X = sp.Matrix([[x],[y]])
        self._x = x
        self._y = y
    
    def __repr__(self):
        return self.sym_A*self._X + self.b

    def __call__(self, X):
        if type(X) is list or type(X) is tuple:
            X = np.array(X)
        X = X.reshape(2,1)
        if X is NumArray:
            return self.A*X + self.b
        return self.sym_A*X + self.b

    def eval(self, X: NumArray):
        """Return flatten np.array of F(X)
        """
        return np.array(self.__call__(X)).flatten()
    
    def plot(self, x_rage = (-10, 10), y_rage = (-10, 10), n_grid = 30):
        # draw singular points
        x_min, x_max = x_rage
        y_min, y_max = y_rage
        plt.grid(True)

        try:
            x0, y0 = np.array(self.A.I*(-self.b)) # calcula a singularidade
            singular_pts = True
            print(f'Singularidade: ({x0[0]},{y0[0]})')
        except nplg.LinAlgError: 
            x0, y0 = np.array(-self.b).flatten() # por padrao (0,0) é singular para os casos degenerados
            singular_pts = False
        
        # Draw streamplot
        # Create a grid for the vector field
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        X, Y = np.meshgrid(x_grid, y_grid)

        U = np.zeros(X.shape)
        V = np.zeros(Y.shape)
        # Evaluate the vector field at each grid point
        for i in range(n_grid):
            for j in range(n_grid):
                X_vec = np.array([X[i, j], Y[i, j]])
                dx_dt = self.eval(X_vec)
                U[i, j] = dx_dt[0]
                V[i, j] = dx_dt[1]
        
        plt.streamplot(X, Y, U, V, color='b', linewidth=0.7, density=1.5, arrowstyle='->', arrowsize=1.5)

        # Draw autovects lines
        print('Autovalores:')
        for lbd, mul, vecs in self.sym_A.eigenvects():
            print(f'\tlbd: {lbd}, mul: {mul}:')
            # Se não tem ponto singular unico
            # entao se o autovalor é zero o vetor indica as singularidade
            # se nao for, então pula para o proximo vetor
            if (not singular_pts) and lbd: continue

            t = sp.symbols('t')

            for a, b in vecs:
                print(f'\t\tVetor: ({a}, {b})')
                if np.abs(a) > np.abs(b):
                    # x cresce mais
                    t1 = (x_min - x0)/a
                    t2 = (x_max - x0)/a
                else:
                    # y cresce mais
                    t1 = (y_min - y0)/b
                    t2 = (y_max - y0)/b
                # Isso nao garante que saia do frame range
                # Mas se sair é pq o ponto singular é mal posicionado
                # dai ele ajuda melhorando o frame_range    
                plt.plot([x0 + a*t2, x0 + a*t1], [y0 + b*t2, y0 + b*t1], 'k--')
                if singular_pts:
                    plt.plot(x0, y0, 'ro')
        plt.show(block=False)

    def __add__(self, other):
        return self.__repr__() + other
    def __sub__(self, other):
        return self.__repr__() - other
    def __mul__(self, other):
        return self.__repr__() * other
    def __truediv__(self, other):
        return self.__repr__() / other
    def __floordiv__(self, other):
        return self.__repr__() // other

